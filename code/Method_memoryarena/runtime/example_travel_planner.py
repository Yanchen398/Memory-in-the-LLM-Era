"""
Travel Planner - Memory-Augmented Agent Benchmark
Follows the same pattern as example_math.py:
    env_client.reset → memory.wrap → agent.act → env_client.step → agent.build_memory_entry → memory.add

Requires:
    - Environment server running: python env/env_server.py
    - Memory server running:      python memory/server.py  (if using a memory system)
"""

import os
import re
import json
import time
import uuid
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from agent.travel_planner import TravelPlannerAgent
from env.env_client import EnvironmentClient


# ---------------------------------------------------------------------------
# Memory system factory (same as original sole_planning.py)
# ---------------------------------------------------------------------------

def get_memory_system(memory_system_name: str, user_id: str, server_url: str = "http://0.0.0.0:8000"):
    if memory_system_name == "none":
        return None
    from Method_memoryarena.client import MemoryClient
    return MemoryClient(user_id=user_id, memory_system_name=memory_system_name, base_url=server_url)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            data.append(json.loads(line))
    return data


def parse_all_plans(result_text, queries):
    all_results = []
    pattern = r'===\s*([^=]+?)\'s Plan\s*==='
    parts = re.split(pattern, result_text)

    name_plan_pairs = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            name = parts[i].strip()
            plan = parts[i + 1].strip()
            name_plan_pairs[name] = plan

    for idx, round_item in enumerate(queries, start=1):
        name = round_item.get('name', f'Person{idx}')
        query = round_item['query']
        result = name_plan_pairs.get(name, "")
        all_results.append({
            'person_idx': idx,
            'name': name,
            'query': query,
            'result': result,
        })

    return all_results


def format_person_plan(name, daily_plans):
    lines = [f"=== {name}'s Plan ==="]
    for day in daily_plans:
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--judgement_mode", type=str, default="hint",
                        choices=["answer", "hint", "none"])
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--memory_system", type=str, default="none",
                        choices=["none", "long_context", "mirix", "letta", "mem0", "mem0-g",
                                 "rag", "bm25", "text-embedding-3-small", "cognee", "memorag",
                                 "graphrag", "lightmem", "amem", "reasoningbank", "zep"])
    parser.add_argument("--server_url", type=str, default="http://0.0.0.0:8000")
    parser.add_argument("--env_server_url", type=str, default="http://0.0.0.0:8001")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SOLE PLANNING - Agent Mode")
    print("=" * 60)
    print(f"Model:         {args.model_name}")
    print(f"Data file:     {args.data_file}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Judgement:     {args.judgement_mode}")
    print(f"Memory system: {args.memory_system}")
    print(f"Memory URL:    {args.server_url}")
    print(f"Env URL:       {args.env_server_url}")
    print(f"Max steps:     {args.max_steps}")
    print("=" * 60 + "\n")

    query_data_list = load_line_json_data(args.data_file)
    print(f"Loaded {len(query_data_list)} data items")
    if query_data_list:
        print('First item keys:', list(query_data_list[0].keys()))

    agent = TravelPlannerAgent(
        model_name=args.model_name,
        max_steps=args.max_steps,
    )

    task_id = str(uuid.uuid4())
    env_client = EnvironmentClient(
        task_id=task_id,
        env_name="travel_planner",
        base_url=args.env_server_url,
        env_config={
            "judgement_mode": args.judgement_mode,
        },
    )

    start_time = time.time()

    for query_data in tqdm(query_data_list):
        data_idx = query_data['id']
        output_file = os.path.join(args.output_dir, f'generated_plan_{data_idx}.json')
        if os.path.exists(output_file):
            print(f"[Data {data_idx}] Output file already exists, skipping: {output_file}")
            continue

        # --- Reset env for this group ---
        obs = env_client.reset(seed=data_idx)

        # --- Initialize memory system for this group ---
        memory_user_id = f"data_{data_idx}_{args.model_name}_{args.memory_system}"
        memory_system = get_memory_system(args.memory_system, memory_user_id, args.server_url)

        if memory_system:
            print(f"\n[Memory] Initialized {args.memory_system} for user_id: {memory_user_id}")

        multi_queries = obs["questions"]
        ground_truth_list = obs.get("answers", [])

        gt_by_round = {}
        for gt in ground_truth_list:
            gt_by_round[gt['round_idx']] = gt

        all_round_results = []
        all_results = []
        all_scratchpads = []
        total_rounds = len(multi_queries)
        accumulated_plan_text = ""
        round_judgements = {}

        agent.reset()

        # --- Handle base_person ---
        base_person = obs.get("base_person")
        if base_person:
            base_name = base_person['name']
            base_query = base_person['query']
            base_plan = format_person_plan(base_name, base_person['daily_plans'])
            accumulated_plan_text = base_plan
            print(f"[Data {data_idx}] Base person set: {base_name}")

            if memory_system:
                base_chunk = json.dumps({
                    "name": base_name,
                    "query": base_query,
                    "is_base_person": True,
                    "final_plan": base_plan,
                }, ensure_ascii=False)
                memory_system.add(base_chunk)

                if args.memory_system in ['mem0', 'mem0-g']:
                    print("[Memory] Waiting 10s for Mem0 indexing...")
                    time.sleep(50)
                print(f"[Memory] Added base person to memory ({len(base_chunk)} chars)")
                print(f"[Memory] Base person query: {base_query[:100]}...")
                print(f"[Memory] Base chunk content:\n{base_chunk[:1000]}{'...' if len(base_chunk) > 1000 else ''}")
            else:
                agent.set_base_person(base_name, base_query, base_plan)

        subquery_times = {}

        # --- Loop over persons ---
        for round_item in multi_queries:
            subquery_start = time.time()

            single_query = round_item['query']
            round_idx = round_item['round_idx']
            name = round_item.get('name', f'User{round_idx}')

            print(f"\n{'=' * 50}")
            print(f"[Data {data_idx}, Round {round_idx}, Name: {name}] Starting Agent...")
            print(f"{'=' * 50}")
            print(f"Query: {single_query[:200]}...")

            # 1. Memory wrap
            memory_context = None
            if memory_system:
                wrapped_prompt = memory_system.wrap_user_prompt(single_query)
                memory_context = wrapped_prompt.split("</memory_context>")[0] + "</memory_context>"
                memory_context = memory_context.strip()

                print(f"\n[Memory] === Retrieved Context for {name} ===")
                print(f"[Memory] Context length: {len(memory_context)} chars")
                print(f"[Memory] Content:\n{memory_context[:2000]}{'...' if len(memory_context) > 2000 else ''}")
                print(f"[Memory] === End Retrieved Context ===")

            # 2. Agent act (full ReAct loop inside)
            agent.prepare_for_person(
                name=name,
                round_idx=round_idx,
                include_previous_plans=(memory_system is None),
                memory_context=memory_context,
            )
            action = agent.act(single_query)

            accumulated_plan_text += f"\n\n{action}" if accumulated_plan_text else action

            agent_result = agent.last_result

            print(f"\n{'=' * 50}")
            print(f"[Data {data_idx}, Round {round_idx}, Name: {name}] RESULT:")
            print(f"{'=' * 50}\n")
            print(f"{action[:1000]}..." if len(action) > 1000 else action)
            print(f"\nSteps: {agent_result.total_steps}, Success: {agent_result.success}")

            # 3. Env step (evaluate via env_server)
            gt = gt_by_round.get(round_idx)
            if gt and 'daily_plans' in gt:
                ground_truth = {
                    "name": name,
                    "daily_plans": gt['daily_plans'],
                    "judgement_mode": args.judgement_mode,
                }
                result = env_client.step(
                    action, ground_truth=ground_truth, need_judge=True,
                )
            else:
                result = env_client.step(action)

            observation = result["observation"]
            reward = result["reward"]
            info = result.get("info", {})

            round_result = {
                'round': round_idx,
                'name': name,
                'query': single_query,
                'result': action,
            }
            all_round_results.append(round_result)

            scratchpad_dict = agent.get_scratchpad_dict()
            all_scratchpads.append({
                'round': round_idx,
                'name': name,
                'scratchpad': scratchpad_dict,
                'total_steps': agent_result.total_steps,
                'success': agent_result.success,
                'error_message': agent_result.error_message,
            })

            parsed_this_round = parse_all_plans(accumulated_plan_text, multi_queries[:round_idx])
            all_results.append(parsed_this_round)

            judgement = info.get("judgement")
            if judgement:
                round_judgements[round_idx] = judgement

            # 4. Build memory entry and add to memory
            if memory_system:
                obs_with_judgement = {**(observation or {}), "judgement": judgement}
                memory_entry = agent.build_memory_entry(
                    task=single_query,
                    action=action,
                    observation=obs_with_judgement,
                    reward=reward,
                )
                memory_system.add(memory_entry)

                if args.memory_system in ['mem0', 'mem0-g']:
                    print(f"[Memory] Waiting for Mem0 indexing...")
                    time.sleep(60)
                print(f"\n[Memory] === Added Trace for {name} ===")
                print(f"[Memory] Chunk size: {len(memory_entry)} chars")
                print(f"[Memory] Scratchpad steps: {len(scratchpad_dict)}")
                print(f"[Memory] Has judgement: {judgement is not None}")
                if judgement:
                    judge_preview = judgement[:500] + "..." if len(judgement) > 500 else judgement
                    print(f"[Memory] Judgement:\n{judge_preview}")
                print(f"[Memory] Chunk content:\n{memory_entry[:2000]}{'...' if len(memory_entry) > 2000 else ''}")
                print(f"[Memory] === End Trace ===\n")

            # Non-memory: feed cumulative judge feedback to agent
            if round_idx < total_rounds and memory_system is None:
                all_feedback = [
                    round_judgements[r]
                    for r in sorted(round_judgements.keys())
                    if r <= round_idx
                ]
                if all_feedback:
                    combined_judge = "\n\n".join(all_feedback)
                    agent.add_judge_feedback(combined_judge)
                    print(f"[Added judge ({args.judgement_mode}) for all travelers up to round {round_idx}]")

            subquery_time = time.time() - subquery_start
            subquery_times[round_idx] = subquery_time
            print(f"\n[Timing] Data {data_idx}, Round {round_idx}, {name}: {subquery_time:.2f}s")

        # --- Save output ---
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        final_parsed_results = all_results[-1] if all_results else []

        for person in final_parsed_results:
            person_idx = person.get('person_idx')
            if person_idx is not None and person_idx in subquery_times:
                person['subquery_time'] = round(subquery_times[person_idx], 2)

        result_to_save = {
            'metadata': {
                'model_name': args.model_name,
                'judgement_mode': args.judgement_mode,
                'memory_system': args.memory_system,
                'mode': 'agent',
                'max_steps': args.max_steps,
            },
            f'{args.model_name}_sole-planning_results': final_parsed_results,
            'all_results': all_results,
            'scratchpads': all_scratchpads,
        }

        print(f"\n{'=' * 60}")
        print(f"[Data {data_idx}] GROUP SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total persons: {total_rounds}")
        print(f"Memory system: {args.memory_system}")
        print(f"Saving to: {output_file}")
        print(f"{'=' * 60}\n")

        with open(output_file, 'w') as f:
            json.dump(result_to_save, f, indent=4, ensure_ascii=False)

    # --- Cleanup ---
    env_client.close()

    # --- Usage stats ---
    duration_seconds = time.time() - start_time
    duration_minutes = duration_seconds / 60

    usage = agent.get_usage_stats()
    usage['duration_seconds'] = round(duration_seconds, 2)
    usage['duration_minutes'] = round(duration_minutes, 2)

    print(f"\n=== Usage Stats ===")
    print(f"Total input tokens: {usage['total_input_tokens']}")
    print(f"Total output tokens: {usage['total_output_tokens']}")
    print(f"Total cost: ${usage['total_cost']:.4f}")
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")

    stats_dir = os.path.join(args.output_dir, "stats_results")
    os.makedirs(stats_dir, exist_ok=True)
    usage_file = os.path.join(stats_dir, "usage_stats.json")
    with open(usage_file, 'w') as f:
        json.dump(usage, f, indent=2)
    print(f"Usage stats saved to: {usage_file}")
    print("Done!")


if __name__ == "__main__":
    main()
