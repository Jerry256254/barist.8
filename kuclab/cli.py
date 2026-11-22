from __future__ import annotations

import argparse
import json
import sys

from .infer import SafetyModel

CAT_PRIORITY = ["sexual_abuse", "self_harm", "violence", "weapons_explosives", "drugs",
                "cybercrime_fraud", "jailbreak_injection", "illegal_marketplace", "other_unsafe",
                "hate_harassment"]  # hate last: catch-all


def _cat_rank(c: str) -> int:
    return CAT_PRIORITY.index(c) if c in CAT_PRIORITY else 99

from .langdetect import identity_positive, detect_lang, must_catch_hit, review_hit, is_support_context, SUPPORT_LINK





BANNER = "Barist.7 - napis text, Enter. 'exit' pro konec."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barist.7",
        description="Barist.7: tiny multilingual safety classifier. Prints safe or unsafe.",
    )
    parser.add_argument("text", nargs="*", help="text to classify; omit to read stdin or use --repl")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--json", action="store_true", help="print JSON with score instead of bare 0/1")
    parser.add_argument("--score", action="store_true", help="print label and score")
    parser.add_argument("--repl", action="store_true", help="interactive mode")
    return parser


def run_repl(model: SafetyModel) -> int:
    print(BANNER)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().lower() in {"exit", "quit", ":q"}:
            return 0
        if not line.strip():
            continue
        verdict = model.predict(line)
        print(verdict.label)


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


class HubClassifier:
    """The published model from the Hub, loaded with plain transformers."""

    def __init__(self, repo: str = "KucLab/Barist.7", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    def predict(self, text: str) -> tuple[str, float, list[str], str | None, str | None, bool]:
        """Returns (label, p_unsafe, categories, override, needs_support, needs_review)."""
        import torch.nn.functional as F

        n_labels = int(self.model.config.num_labels)
        temp = float(getattr(self.model.config, "temperature", 1.0) or 1.0)
        with self._torch.no_grad():
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=int(self.model.config.max_position_embeddings))
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            logits = self.model(**enc).logits[0] / max(temp, 1e-6)
            probs = logits.sigmoid() if n_labels > 2 else logits.softmax(-1)
        p = float(probs[1])
        lang = detect_lang(text)
        lang_thr = getattr(self.model.config, "lang_thresholds", None) or {}
        thr = float(lang_thr.get(lang, lang_thr.get("default", 0.5)))
        label = "unsafe" if p >= thr else "safe"
        cats = []
        hit = must_catch_hit(text)  # Barist.7 hard pre-filter
        if hit:
            label = "unsafe"
        help_ctx = is_support_context(text)
        support = help_ctx or "self_harm" in hit
        cat_block: list[str] = []
        if n_labels > 2:
            id2label = self.model.config.id2label
            cat_thr = getattr(self.model.config, "cat_thresholds", None) or {}
            cat_exp = set(getattr(self.model.config, "cat_experimental", []) or [])
        cat_block_enabled = bool(getattr(self.model.config, "category_block", False))  # Barist.7: default OFF until head gate
        if n_labels > 2:
            if cat_block_enabled and not help_ctx:  # hard block only when head gate passed; help-other never blocked
                for i in range(2, n_labels):
                    if id2label[i] in ("sexual_abuse", "self_harm") and float(probs[i]) >= 0.3:
                        cat_block.append(id2label[i])
            # Barist.7: model self_harm >= 0.3 -> needs_support (not a block)
            if n_labels > 2 and "self_harm" in id2label.values():
                i_sh0 = [i for i, v in id2label.items() if v == "self_harm"][0]
                if float(probs[i_sh0]) >= 0.3:
                    support = True
            if cat_block:
                label = "unsafe"
            if help_ctx and not hit and not cat_block:
                label = "safe"  # help-seeking context is never blocked
            if identity_positive(text) and not hit:
                label = "safe"  # Barist.7: identity-positive waiver
            if label == "unsafe":
                fired = [(float(probs[i]), id2label[i]) for i in range(2, n_labels)
                         if float(probs[i]) >= float(cat_thr.get(id2label[i], 0.5)) and id2label[i] not in cat_exp]
                fired_set = {name for _, name in fired} | set(hit) | set(cat_block)
                fired.sort(key=lambda x: (_cat_rank(x[1]), -x[0]))
                if not fired_set:  # unsafe must always carry >=1 category: top-1 by score
                    top = max(((float(probs[i]), id2label[i]) for i in range(2, n_labels)), key=lambda x: x[0])
                    fired = [top]
                    fired_set = {top[1]}
                cats = [name for _, name in fired[:2]] + [c for c in (hit + cat_block) if c not in {n for _, n in fired[:2]}]
        if cat_block_enabled and not support and n_labels > 2 and "self_harm" in id2label.values():
            i_sh = [i for i, v in id2label.items() if v == "self_harm"][0]
            if float(probs[i_sh]) >= 0.3:
                support = True
        override = "must_catch" if hit else ("category_block" if cat_block else None)
        review = bool(review_hit(text))
        return label, p, cats, override, (SUPPORT_LINK if support else None), review


def run_ui(predict, source: str) -> int:
    """Small interactive terminal UI: phrase in, safe/unsafe + score out."""
    print(f"Barist.7 - interactive terminal UI  [{source}]")
    print("Type a phrase, Enter to classify. 'exit' to quit.\n")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if text.lower() in {"exit", "quit", ":q"}:
            return 0
        if not text:
            continue
        label, prob, cats, override, support, review = predict(text)
        colored = _color(f"{label:6s}", "31" if label == "unsafe" else "32")
        extra = f"  [{', '.join(cats)}]" if cats else ""
        if override:
            extra += f"  (override: {override})"
        if support:
            extra += f"  (needs_support: {support})"
        if review:
            extra += "  (needs_review)"
        print(f"{colored} {prob:.4f}{extra} | {text}")


def ui_main(argv: list[str] | None = None) -> int:
    """Entry point for the `barist.7` command."""
    parser = argparse.ArgumentParser(
        prog="barist.7",
        description="Barist.7 terminal UI: classify text as safe/unsafe (+ categories). "
        "Run without arguments for the interactive UI.",
    )
    parser.add_argument("text", nargs="*", help="phrase to classify once; omit to open the interactive UI")
    parser.add_argument("--repo", default="KucLab/Barist.7", help="Hub repo id of the model")
    parser.add_argument("--checkpoint", default=None, help="use a local training checkpoint instead of the Hub model")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--json", action="store_true", help="one-shot JSON output")
    args = parser.parse_args(argv)

    if args.checkpoint:
        local = SafetyModel(checkpoint=args.checkpoint, device=args.device)

        def predict(text: str) -> tuple[str, float, list[str], str | None, str | None, bool]:
            v = local.predict(text)
            return ("unsafe" if v.is_unsafe else "safe"), v.probability, v.categories or [], \
                getattr(v, "override", None), getattr(v, "needs_support", None), getattr(v, "needs_review", False)

        source = str(args.checkpoint)
    else:
        print(f"loading {args.repo} ...", file=sys.stderr)
        clf = HubClassifier(args.repo, device=args.device)
        predict = clf.predict
        source = f"{args.repo} ({clf.device})"

    text = " ".join(args.text).strip()
    if text:
        label, prob, cats, override, support, review = predict(text)
        if args.json:
            print(json.dumps({"label": label, "unsafe": label == "unsafe", "score": round(prob, 4),
                              **({"categories": cats} if cats else {}),
                              **({"override": override} if override else {}),
                              **({"needs_support": support} if support else {}),
                              **({"needs_review": True} if review else {})}))
        else:
            colored = _color(f"{label:6s}", "31" if label == "unsafe" else "32")
            extra = f"  [{', '.join(cats)}]" if cats else ""
            if override:
                extra += f"  (override: {override})"
            if support:
                extra += f"  (needs_support: {support})"
            if review:
                extra += "  (needs_review)"
            print(f"{colored} {prob:.4f}{extra}")
        return 0
    return run_ui(predict, source)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    text = " ".join(args.text).strip()
    if not text and not args.repl and not sys.stdin.isatty():
        text = sys.stdin.read().strip()

    model = SafetyModel(checkpoint=args.checkpoint, threshold=args.threshold, device=args.device)

    if args.repl or (not text and sys.stdin.isatty()):
        return run_repl(model)

    if not text:
        parser.error("no input text provided")

    verdict = model.predict(text)
    if args.json:
        print(json.dumps(verdict.as_dict()))
    elif args.score:
        print(f"{verdict.label} {verdict.probability:.4f}")
    else:
        print(verdict.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def ui_main_b8(argv: list[str] | None = None) -> int:
    """Entry point for the `barist.8` command (HFShield: e5-small encoder + heads)."""
    parser = argparse.ArgumentParser(
        prog="barist.8",
        description="Barist.8 terminal UI: classify text as safe/unsafe (+ 15 categories). "
        "Run without arguments for the interactive UI.",
    )
    parser.add_argument("text", nargs="*", help="phrase to classify once; omit to open the interactive UI")
    parser.add_argument("--repo", default="KucLab/Barist.8", help="Hub repo id of the model")
    parser.add_argument("--model-dir", default=None, help="local Barist.8 bundle dir (shield_config.json) instead of the Hub")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--json", action="store_true", help="one-shot JSON output")
    args = parser.parse_args(argv)

    from .hf_model import HFShield

    if args.model_dir:
        sh = HFShield(args.model_dir, device=args.device)
        source = f"{args.model_dir} ({sh.device})"
    else:
        from huggingface_hub import snapshot_download
        print(f"loading {args.repo} ...", file=sys.stderr)
        d = snapshot_download(args.repo)
        sh = HFShield(d, device=args.device)
        source = f"{args.repo} ({sh.device})"

    def predict(text: str):
        v = sh.predict(text)
        return (v["label"], v["score"], v.get("categories") or [], v.get("override"),
                v.get("needs_support"), bool(v.get("needs_review")))

    text = " ".join(args.text).strip()
    if text:
        label, prob, cats, override, support, review = predict(text)
        if args.json:
            print(json.dumps({"label": label, "unsafe": label == "unsafe", "score": prob,
                              **({"categories": cats} if cats else {}),
                              **({"override": override} if override else {}),
                              **({"needs_support": support} if support else {}),
                              **({"needs_review": True} if review else {})}))
        else:
            colored = _color(f"{label:6s}", "31" if label == "unsafe" else "32")
            extra = f"  [{', '.join(cats)}]" if cats else ""
            if override:
                extra += f"  (override: {override})"
            if support:
                extra += f"  (needs_support: {support})"
            if review:
                extra += "  (needs_review)"
            print(f"{colored} {prob:.4f}{extra}")
        return 0
    return run_ui(predict, source)
