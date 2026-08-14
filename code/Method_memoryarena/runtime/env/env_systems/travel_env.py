import os
import re
import json
import difflib
from typing import Any, Dict, List, Optional, Tuple
from .base_env import BaseEnvironment
from .travel_planner_env.data_loader import load_travel_data

ENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "travel_planner_env")

SLOTS = ["current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]

SIM_TH = 0.7

HINT_SIM_TH = 0.9


def _similarity(a, b):
    if not a or not b:
        return 0.0
    a, b = str(a).strip().lower(), str(b).strip().lower()
    if a == "-" and b == "-":
        return 1.0
    m = min(len(a), len(b))
    return difflib.SequenceMatcher(None, a[:m], b[:m]).ratio() if m else 0.0




def _get_day(plan, day_idx):
    if not plan:
        return None
    for d in plan:
        if d.get("day") == day_idx or d.get("days") == day_idx:
            return d
    return None


def _parse_person_plan_from_result(result_text, name):
    pattern = rf"===\s*{re.escape(name)}'s Plan\s*===(.*?)(?====|$)"
    match = re.search(pattern, result_text, re.DOTALL)
    if not match:
        return []
    plan_text = match.group(1).strip()
    days = []
    day_pattern = r"Day\s*(\d+):(.*?)(?=Day\s*\d+:|$)"
    for day_match in re.finditer(day_pattern, plan_text, re.DOTALL):
        day_idx = int(day_match.group(1))
        day_content = day_match.group(2)
        day_obj = {'day': day_idx}
        for line in day_content.strip().split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                day_obj[key] = val.strip()
        days.append(day_obj)
    return days


def _format_judge_answer(name, gt_plans):
    lines = [f"Possible answer for {name}:"]
    for day in gt_plans:
        day_idx = day.get('days') or day.get('day')
        lines.append(f"Day {day_idx}:")
        lines.append(f"Current City: {day.get('current_city', '-')}")
        lines.append(f"Transportation: {day.get('transportation', '-')}")
        lines.append(f"Breakfast: {day.get('breakfast', '-')}")
        lines.append(f"Attraction: {day.get('attraction', '-')}")
        lines.append(f"Lunch: {day.get('lunch', '-')}")
        lines.append(f"Dinner: {day.get('dinner', '-')}")
        lines.append(f"Accommodation: {day.get('accommodation', '-')}")
        lines.append("")
    return "\n".join(lines)


def _format_judge_hint(name, model_plan, gt_plans):
    lines = [f"Feedback for {name}:"]
    lines.append("The following slots need correction:")
    has_error = False
    for gt_day in gt_plans:
        day_idx = gt_day.get('days') or gt_day.get('day')
        model_day = _get_day(model_plan, day_idx)
        for slot in SLOTS:
            expected = gt_day.get(slot, '-')
            actual = model_day.get(slot, '-') if model_day else '-'
            if _similarity(expected, actual) < HINT_SIM_TH:
                slot_display = slot.replace('_', ' ').title()
                lines.append(f"- Day {day_idx}, {slot_display}")
                has_error = True
    if not has_error:
        lines.append("- None (all correct)")
    return "\n".join(lines)


class TravelPlannerEnvironment(BaseEnvironment):
    """Travel Planner environment for trip planning tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.data = load_travel_data()
        self.data_by_id = {item['id']: item for item in self.data}
        self.judgement_mode = self.config.get("judgement_mode", "none")

        self.history: List[Dict] = []
        self._current_observation: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.current_group: Optional[Dict] = None

        print("TravelPlannerEnvironment initialized with config")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.history = []
        self.step_count = 0

        if seed is not None and seed in self.data_by_id:
            self.current_group = self.data_by_id[seed]
        elif seed is not None and 0 <= seed < len(self.data):
            self.current_group = self.data[seed]
        else:
            self.current_group = self.data[0] if self.data else {}

        self._current_observation = {
            "group_id": self.current_group.get("id"),
            "base_person": self.current_group.get("base_person"),
            "questions": self.current_group.get("questions", []),
            "answers": self.current_group.get("answers", []),
            "step": self.step_count,
        }
        return self._current_observation

    def step(
        self,
        action: Any,
        ground_truth: Any = None,
        need_judge: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = None
        info: Dict[str, Any] = {}

        if need_judge and ground_truth is not None:
            if isinstance(ground_truth, dict):
                name = ground_truth.get("name", "")
                gt_plans = ground_truth.get("daily_plans", [])
                judgement_mode = ground_truth.get("judgement_mode", self.judgement_mode)
            elif isinstance(ground_truth, list):
                name = ""
                gt_plans = ground_truth
                judgement_mode = self.judgement_mode
            else:
                name = ""
                gt_plans = []
                judgement_mode = self.judgement_mode

            model_plan = _parse_person_plan_from_result(action, name) if name else []
            reward = self._evaluate_slots(model_plan, gt_plans)

            if judgement_mode == "hint":
                info["judgement"] = _format_judge_hint(name, model_plan, gt_plans)
            elif judgement_mode == "answer":
                info["judgement"] = _format_judge_answer(name, gt_plans)

        self.history.append({
            "step": self.step_count,
            "action": action[:200] if isinstance(action, str) else action,
            "reward": reward,
        })

        self._current_observation = {
            "step": self.step_count,
            "final": action,
            "reward": reward,
        }
        return self._current_observation, reward, info

    def _evaluate_slots(self, model_plan, gt_plans):
        if not gt_plans:
            return False
        for gt_day in gt_plans:
            day_idx = gt_day.get("days") or gt_day.get("day")
            sub_day = _get_day(model_plan, day_idx)
            if not sub_day:
                return False
            for slot in SLOTS:
                expected = gt_day.get(slot, "-")
                actual = sub_day.get(slot, "-")
                if _similarity(expected, actual) < SIM_TH:
                    return False
        return True

    def judge(self, plan_text: str, ground_truth: Any) -> bool:
        if isinstance(ground_truth, dict):
            gt_plans = ground_truth.get("daily_plans", [])
            name = ground_truth.get("name", "")
        elif isinstance(ground_truth, list):
            gt_plans = ground_truth
            name = ""
        else:
            return False

        model_plan = _parse_person_plan_from_result(plan_text, name)
        return self._evaluate_slots(model_plan, gt_plans)

    def get_observation(self) -> Dict[str, Any]:
        return self._current_observation or self.reset()

    def close(self):
        self.history = []
        self._current_observation = None
        self.current_group = None
