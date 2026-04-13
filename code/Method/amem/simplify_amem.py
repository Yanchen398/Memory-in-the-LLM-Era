import os
import json
import glob
import time
import argparse
from openai import OpenAI

def simplify_one(client, model_name, question, response, retry=3):
    prompt = f"""Simplify the answer to the shortest direct fact.

Question: {question}
Original answer: {response}

Rules:
- Keep only core fact.
- If date/time question, output only date/time phrase.
- No explanation.
"""
    for _ in range(retry):
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You simplify answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=64,
                timeout=30
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[WARN] simplify failed: {e}")
            time.sleep(2)
    return ""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True)
    p.add_argument("--base_url", default="http://localhost:8005/v1")
    p.add_argument("--api_key", default="fake_key")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    files = sorted(glob.glob(os.path.join(args.result_dir, "sample_*", "qa_result.json")))
    print(f"Found {len(files)} files")

    for fp in files:
        data = json.load(open(fp, "r", encoding="utf-8"))
        for item in data:
            q = item.get("question", "")
            resp = item.get("response", "")
            item["concise_response"] = simplify_one(client, args.model_name, q, resp)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"updated: {fp}")

if __name__ == "__main__":
    main()