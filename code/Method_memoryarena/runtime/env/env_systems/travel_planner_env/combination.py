import argparse
import json
import glob
import os
import re
from tqdm import tqdm


def parse_plan_text(result_text):
    """用正则直接解析 plan 文本，返回结构化的 dict list"""
    if not result_text or result_text in ["", None, "Max Token Length Exceeded."]:
        return None
    
    days = []
    day_pattern = r"Day\s*(\d+):(.*?)(?=Day\s*\d+:|$)"
    for day_match in re.finditer(day_pattern, result_text, re.DOTALL):
        day_idx = int(day_match.group(1))
        day_content = day_match.group(2)
        
        day_obj = {'day': day_idx}
        for line in day_content.strip().split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                day_obj[key] = val.strip()
        days.append(day_obj)
    
    return days if days else None


def combine(model_name, output_dir, submission_file_dir, mode="sole_planning"):
    files = sorted(
        glob.glob(f'{output_dir}/generated_plan_*.json'),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    os.makedirs(submission_file_dir, exist_ok=True)

    if mode == 'baseline':
        key = f'{model_name}_baseline_results'
    else:
        key = f'{model_name}_sole-planning_results'

    group_list = []

    for filepath in tqdm(files, desc="Combining"):
        data_idx = int(filepath.split('_')[-1].split('.')[0])
        generated_plan = json.load(open(filepath))

        if key not in generated_plan:
            raise KeyError(f"Key '{key}' not found in {filepath}. Keys: {list(generated_plan.keys())}")

        persons = []
        for round_item in generated_plan[key]:
            result_text = round_item.get('result', '')
            parsed_plan = parse_plan_text(result_text)
            persons.append({
                "person_idx": round_item.get('person_idx'),
                "name": round_item.get('name'),
                "query": round_item.get('query'),
                "plan": parsed_plan
            })

        all_results_raw = generated_plan.get('all_results', [])
        all_results = []
        for round_plans in all_results_raw:
            round_parsed = []
            for plan_item in round_plans:
                result_text = plan_item.get('result', '')
                parsed_plan = parse_plan_text(result_text)
                round_parsed.append({
                    "person_idx": plan_item.get('person_idx'),
                    "name": plan_item.get('name'),
                    "query": plan_item.get('query'),
                    "plan": parsed_plan
                })
            all_results.append(round_parsed)

        group_list.append({
            "id": data_idx,
            "persons": persons,
            "all_results": all_results
        })

    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    output_file = f'{submission_file_dir}/{safe_model_name}_submission.jsonl'
    with open(output_file, 'w', encoding='utf-8') as w:
        for group in group_list:
            w.write(json.dumps(group, ensure_ascii=False) + "\n")

    total_persons = sum(len(g["persons"]) for g in group_list)
    print(f"Saved {len(group_list)} groups ({total_persons} persons) to {output_file}")
    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--submission_file_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=['sole_planning', 'baseline'])
    args = parser.parse_args()
    combine(args.model_name, args.output_dir, args.submission_file_dir, args.mode)
