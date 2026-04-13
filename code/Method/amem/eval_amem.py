import os
import json
import glob
import argparse
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)

def tok(s): return nltk.word_tokenize(str(s).lower()) if s is not None else []

def f1(gold, pred):
    g, p = set(tok(gold)), set(tok(pred))
    if not g or not p: return 0.0
    inter = len(g & p)
    pr, rc = inter / len(p), inter / len(g)
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0

def bleu1(gold, pred):
    g, p = tok(gold), tok(pred)
    if not g or not p: return 0.0
    return sentence_bleu([g], p, weights=(1,0,0,0), smoothing_function=SmoothingFunction().method1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--use_concise", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.result_dir, "sample_*", "qa_result.json")))
    details, by_cat = [], {}

    for fp in files:
        sample_id = os.path.basename(os.path.dirname(fp))
        data = json.load(open(fp, "r", encoding="utf-8"))
        for x in data:
            gold = x.get("answer", "")
            pred = x.get("concise_response", "") if args.use_concise else x.get("response", "")
            cat = x.get("category", None)

            s_f1 = f1(gold, pred)
            s_b1 = bleu1(gold, pred)

            details.append({
                "sample_id": sample_id,
                "question": x.get("question", ""),
                "category": cat,
                "gold": gold,
                "pred": pred,
                "f1": s_f1,
                "bleu1": s_b1
            })

            by_cat.setdefault(cat, {"f1": [], "bleu1": [], "count": 0})
            by_cat[cat]["f1"].append(s_f1)
            by_cat[cat]["bleu1"].append(s_b1)
            by_cat[cat]["count"] += 1

    out = {
        "overall": {
            "count": len(details),
            "avg_f1": float(np.mean([d["f1"] for d in details])) if details else 0.0,
            "avg_bleu1": float(np.mean([d["bleu1"] for d in details])) if details else 0.0
        },
        "by_category": {
            str(k): {
                "count": v["count"],
                "avg_f1": float(np.mean(v["f1"])) if v["f1"] else 0.0,
                "avg_bleu1": float(np.mean(v["bleu1"])) if v["bleu1"] else 0.0
            } for k, v in by_cat.items()
        },
        "details": details
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.output_json}")

if __name__ == "__main__":
    main()