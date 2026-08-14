import json
import difflib
import os
import csv
from collections import defaultdict
from datetime import datetime
import argparse
from .data_loader import load_travel_data

SIM_TH = 0.7

def similarity(a, b):
    if not a or not b:
        return 0.0
    a, b = str(a).strip().lower(), str(b).strip().lower()
    if a == "-" and b == "-":
        return 1.0
    m = min(len(a), len(b))
    return difflib.SequenceMatcher(None, a[:m], b[:m]).ratio() if m else 0.0

def load_ground_truth():
    gt = {}
    base_plans = {}
    group_person_count = {}

    for row in load_travel_data():
        data_idx = row["id"]
        base_person = row.get("base_person")
        if base_person:
            base_plans[data_idx] = base_person["daily_plans"]

        persons = row.get("answers", [])
        group_person_count[data_idx] = len(persons)
        for person in persons:
            person_idx = person["round_idx"]
            gt[(data_idx, person_idx)] = person["daily_plans"]

    return gt, base_plans, group_person_count

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_submissions(submission_path):
    sub_final = {}
    for row in _read_jsonl(submission_path):
        data_idx = row["id"]
        for person in row.get("persons", []):
            person_idx = person.get("person_idx")
            if person_idx is not None:
                sub_final[(data_idx, person_idx)] = person.get("plan")
    return sub_final

SLOTS = ["breakfast", "lunch", "dinner", "accommodation", "transportation", "attraction"]

def get_day(plan, day_idx):
    if not plan:
        return None
    for d in plan:
        if d.get("day") == day_idx or d.get("days") == day_idx:
            return d
    return None

def find_constraint_slots(gt, base_plans, data_idx, person_idx):
    base_plan = base_plans.get(data_idx)
    this_plan = gt.get((data_idx, person_idx))
    if not base_plan or not this_plan:
        return set()
    
    constraint_slots = set()
    for day_obj in this_plan:
        day_idx = day_obj.get("days") or day_obj.get("day")
        base_day = get_day(base_plan, day_idx)
        if not base_day:
            continue
        for slot in SLOTS:
            base_val = base_day.get(slot, "-")
            this_val = day_obj.get(slot, "-")
            if similarity(base_val, this_val) < SIM_TH:
                constraint_slots.add((day_idx, slot))
    return constraint_slots

def check_slot_pass(gt_plan, sub_plan, day_idx, slot):
    if not gt_plan or not sub_plan:
        return False
    gt_day = get_day(gt_plan, day_idx)
    sub_day = get_day(sub_plan, day_idx)
    if not gt_day or not sub_day:
        return False
    expected = gt_day.get(slot, "-")
    actual = sub_day.get(slot, "-")
    return similarity(expected, actual) >= SIM_TH

def check_person_full_pass(gt_plan, sub_plan):
    if not gt_plan or not sub_plan:
        return False
    for gt_day in gt_plan:
        day_idx = gt_day.get("days") or gt_day.get("day")
        sub_day = get_day(sub_plan, day_idx)
        if not sub_day:
            return False
        for slot in SLOTS:
            expected = gt_day.get(slot, "-")
            actual = sub_day.get(slot, "-")
            if similarity(expected, actual) < SIM_TH:
                return False
    return True


def evaluate(submission_path, model_name="unknown", memory_system="none", global_csv=None):
    gt, base_plans, group_person_count = load_ground_truth()
    sub_final = load_submissions(submission_path)

    # SR: Group (all persons pass)
    group_all_pass_count = 0
    total_groups = 0

    # PS: Overall Person Full Pass Rate
    total_persons_global = 0
    full_pass_persons_global = 0

    # SPS: Data-level Avg Constraint Rate
    data_avg_constraint_rates = []

    gt_data_indices = set(k[0] for k in gt.keys())
    sub_data_indices = set(k[0] for k in sub_final.keys())
    data_indices = sorted(gt_data_indices & sub_data_indices)

    for data_idx in data_indices:
        person_indices = sorted(k[1] for k in sub_final.keys() if k[0] == data_idx)
        total_groups += 1

        all_persons_pass = True
        data_person_constraint_rates = []

        for person_idx in person_indices:
            gt_plan = gt.get((data_idx, person_idx))
            sub_plan = sub_final.get((data_idx, person_idx))

            constraint_slots = find_constraint_slots(gt, base_plans, data_idx, person_idx)

            person_constraint_passed = 0
            person_constraint_total = 0

            if gt_plan:
                for gt_day in gt_plan:
                    day_idx = gt_day.get("days") or gt_day.get("day")
                    for slot in SLOTS:
                        has_constraint = (day_idx, slot) in constraint_slots
                        if has_constraint:
                            passed = check_slot_pass(gt_plan, sub_plan, day_idx, slot)
                            person_constraint_total += 1
                            if passed:
                                person_constraint_passed += 1

            if person_constraint_total > 0:
                data_person_constraint_rates.append(person_constraint_passed / person_constraint_total)

            # PS
            person_pass = check_person_full_pass(gt_plan, sub_plan)
            total_persons_global += 1
            if person_pass:
                full_pass_persons_global += 1
            else:
                all_persons_pass = False

        # SR
        if all_persons_pass:
            group_all_pass_count += 1

        # SPS
        if data_person_constraint_rates:
            data_avg_constraint_rates.append(sum(data_person_constraint_rates) / len(data_person_constraint_rates))

    ps = full_pass_persons_global / total_persons_global * 100 if total_persons_global else 0
    sps = sum(data_avg_constraint_rates) / len(data_avg_constraint_rates) * 100 if data_avg_constraint_rates else 0
    sr = group_all_pass_count / total_groups * 100 if total_groups else 0

    print(f"\n{'='*50}")
    print(f"{'EVALUATION':^50}")
    print(f"{'='*50}")
    print(f"Data:       HuggingFace (ZexueHe/memoryarena, group_travel_planner)")
    print(f"Submission: {submission_path}")
    print(f"Groups:     {total_groups}")
    print(f"{'='*50}")
    print(f"  PS:   {full_pass_persons_global:>4} / {total_persons_global:<4} = {ps:.2f}%")
    print(f"  SPS:  {sps:.2f}%")
    print(f"  SR:   {group_all_pass_count:>4} / {total_groups:<4} = {sr:.2f}%")
    print(f"{'='*50}\n")

    if global_csv:
        csv_dir = os.path.dirname(global_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'memory_system': memory_system,
            'total_groups': total_groups,
            'PS': f"{ps:.2f}",
            'SPS': f"{sps:.2f}",
            'SR': f"{sr:.2f}",
        }

        file_exists = os.path.exists(global_csv)
        file_empty = not file_exists or os.path.getsize(global_csv) == 0
        fieldnames = list(row.keys())

        with open(global_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if file_empty:
                writer.writeheader()
            writer.writerow(row)

        print(f"Appended to global CSV: {global_csv}")

    return {"ps": ps, "sps": sps, "sr": sr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "unknown"))
    parser.add_argument("--memory_system", type=str, default=os.environ.get("MEMORY_SYSTEM", "none"))
    parser.add_argument("--global_csv", type=str, default=None)
    args = parser.parse_args()

    evaluate(
        submission_path=args.submission_path,
        model_name=args.model_name,
        memory_system=args.memory_system,
        global_csv=args.global_csv,
    )
