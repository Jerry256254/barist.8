#!/usr/bin/env python3
"""Barist.8 export: merge e5-small base + heads into a standard transformers model dir,
plus ONNX export (binary+cat logits) and a CPU benchmark.

Usage: python scripts/export_b8.py --shield checkpoints_b8 --out /tmp/bundle_b8
"""
import argparse, json, time, sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shield", default="checkpoints_b8")
    ap.add_argument("--out", default="/tmp/bundle_b8")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    shield = Path(args.shield)
    cfg = json.loads((shield / "shield_config.json").read_text())
    heads = load_file(str(shield / "heads.safetensors"))
    cats = cfg["categories"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # multi-label category model (standard transformers artifact)
    model = AutoModelForSequenceClassification.from_pretrained(
        (str(Path(shield)) if (Path(shield)/"config.json").exists() else cfg["base"]), num_labels=len(cats), problem_type="multi_label_classification")
    model.classifier.weight = torch.nn.Parameter(heads["head_cats.weight"])
    model.classifier.bias = torch.nn.Parameter(heads["head_cats.bias"])
    model.config.id2label = {i: c for i, c in enumerate(cats)}
    model.config.label2id = {c: i for i, c in enumerate(cats)}
    model.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(str(Path(shield)) if (Path(shield)/"config.json").exists() else cfg["base"])
    tok.save_pretrained(out)

    # binary head as a small extra file + metadata
    from safetensors.torch import save_file
    save_file({"weight": heads["head_bin.weight"], "bias": heads["head_bin.bias"]},
              str(out / "binary_head.safetensors"))
    meta = dict(cfg)
    meta["binary_head"] = "binary_head.safetensors"
    meta["usage"] = ("cats: AutoModelForSequenceClassification(num_labels=15, sigmoid). "
                     "binary: linear head on [CLS] from binary_head.safetensors. "
                     "Full pipeline with regex safety net: kuclab.hf_model.HFShield.")
    (out / "shield_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("merged model saved to", out, flush=True)

    # ONNX export (encoder + both heads, single graph, outputs: p_unsafe, cat_logits)
    import numpy as np
    try:
        # custom: export base encoder + both heads as one graph
        class Enc(torch.nn.Module):
            def __init__(self, base, hb, hc):
                super().__init__()
                self.base = base
                self.hb = hb
                self.hc = hc
            def forward(self, input_ids, attention_mask):
                h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
                return self.hb(h), self.hc(h)
        enc = Enc(model.bert, torch.nn.Linear(model.config.hidden_size, 2), torch.nn.Linear(model.config.hidden_size, len(cats)))
        enc.hb.weight = torch.nn.Parameter(heads["head_bin.weight"])
        enc.hb.bias = torch.nn.Parameter(heads["head_bin.bias"])
        enc.hc.weight = torch.nn.Parameter(heads["head_cats.weight"])
        enc.hc.bias = torch.nn.Parameter(heads["head_cats.bias"])
        enc.eval()
        dummy = tok("test", return_tensors="pt")
        torch.onnx.export(enc, (dummy["input_ids"], dummy["attention_mask"]),
                          str(out / "barist8.onnx"), input_names=["input_ids", "attention_mask"],
                          output_names=["bin_logits", "cat_logits"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
                          opset_version=17, dynamo=False)
        print("onnx exported", flush=True)
    except Exception as e:
        print("onnx export failed:", e, flush=True)

    # CPU latency benchmark of the HF model
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import pipeline
    clf = pipeline("text-classification", model=str(out), device=-1, function_to_apply="sigmoid", top_k=None)
    texts = ["Kde koupit chleba", "Chci vidět nahé děti", "Dobrou noc", "Zabiju ho zítra po škole"] * 25
    t0 = time.time()
    for t in texts:
        clf(t)
    dt = time.time() - t0
    print(f"CPU latency: {dt / len(texts) * 1000:.1f} ms/text ({len(texts)} texts)", flush=True)
    (out / "bench.json").write_text(json.dumps({"cpu_ms_per_text": round(dt / len(texts) * 1000, 2), "n": len(texts)}))


if __name__ == "__main__":
    main()
