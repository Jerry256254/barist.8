"""Barist.8 inference: pretrained multilingual encoder (e5-small) + linear heads.
Same Verdict shape as kuclab.infer, plus the regex safety net and identity waiver."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from .langdetect import (detect_lang, must_catch_hit, review_hit, is_support_context,
                         identity_positive, SUPPORT_LINK)

CAT_PRIORITY = ["sexual_minors", "self_harm", "self_harm_intent", "self_harm_instructions",
                "grooming", "violence", "illicit_violent", "harassment_threatening",
                "hate_threatening", "sexual", "harassment", "hate", "violence_graphic",
                "illicit", "jailbreak_injection"]


def _cat_rank(c):
    return CAT_PRIORITY.index(c) if c in CAT_PRIORITY else 99


# regex pre-filter still speaks the old 10-category names; map to the 15 new ones
HIT2NEW = {"sexual_abuse": "sexual_minors", "self_harm": "self_harm",
           "violence": "violence", "weapons_explosives": "illicit_violent",
           "drugs": "illicit", "cybercrime_fraud": "illicit",
           "jailbreak_injection": "jailbreak_injection", "illegal_marketplace": "illicit",
           "hate_harassment": "hate", "other_unsafe": "illicit"}


class HFShield:
    def __init__(self, shield_dir, device=None, threshold=0.5):
        shield_dir = Path(shield_dir)
        cfg = json.loads((shield_dir / "shield_config.json").read_text())
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        base_src = str(shield_dir) if (shield_dir / "config.json").exists() else cfg["base"]
        self.tok = AutoTokenizer.from_pretrained(base_src)
        self.base = AutoModel.from_pretrained(base_src).to(self.device).eval()
        heads = load_file(str(shield_dir / "heads.safetensors"))
        hidden = self.base.config.hidden_size
        self.head_bin = torch.nn.Linear(hidden, 2).to(self.device)
        self.head_cats = torch.nn.Linear(hidden, len(cfg["categories"])).to(self.device)
        self.head_bin.load_state_dict({"weight": heads["head_bin.weight"], "bias": heads["head_bin.bias"]})
        self.head_cats.load_state_dict({"weight": heads["head_cats.weight"], "bias": heads["head_cats.bias"]})
        self.head_bin.eval(); self.head_cats.eval()
        self.cats = cfg["categories"]
        self.threshold = float(cfg.get("threshold", threshold))
        self.cat_thr = float(cfg.get("cat_threshold", 0.5))
        self.max_len = int(cfg.get("max_len", 256))

    @torch.no_grad()
    def raw(self, text):
        enc = self.tok(text[:4000], truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        h = self.base(**enc).last_hidden_state[:, 0]
        p = torch.softmax(self.head_bin(h), -1)[0, 1].item()
        cats = torch.sigmoid(self.head_cats(h))[0].cpu().tolist()
        return p, cats

    def predict(self, text: str) -> dict:
        p, cat_scores = self.raw(text)
        cats = self.cats
        hit = must_catch_hit(text)
        review = review_hit(text)
        help_ctx = is_support_context(text)
        id_pos = identity_positive(text)
        is_unsafe = p >= self.threshold
        if hit:
            is_unsafe = True
        if help_ctx and not hit:
            is_unsafe = False
        if id_pos and not hit:
            is_unsafe = False
        support = help_ctx or "self_harm" in hit
        if cats and cat_scores[cats.index("self_harm")] >= 0.3:
            support = True
        cat_block = []  # category hard block stays OFF (head gate)
        score_map = dict(zip(cats, cat_scores))
        flagged = []
        if is_unsafe:
            hit_new = [HIT2NEW.get(h, h) for h in hit]
            fired = [c for c, s in zip(cats, cat_scores) if s >= self.cat_thr]
            flagged = sorted(set(fired) | set(hit_new), key=lambda c: (_cat_rank(c), -score_map.get(c, 1.0)))[:2]
            if not flagged:
                flagged = [max(cats, key=lambda c: score_map[c])]
        return {
            "label": "unsafe" if is_unsafe else "safe",
            "label_id": "1" if is_unsafe else "0",
            "unsafe": is_unsafe,
            "override": "must_catch" if hit else None,
            "needs_support": SUPPORT_LINK if support else None,
            "needs_review": True if review else None,
            "score": round(p, 4),
            "category_scores": [round(c, 4) for c in cat_scores],
            "categories": flagged if is_unsafe else [],
        }
