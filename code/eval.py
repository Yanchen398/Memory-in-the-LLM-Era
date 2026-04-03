import argparse
import json
import logging
import os
import time

import nltk
import numpy as np
import transformers

from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()
DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"

DATASET_CONFIGS = {
    "loco": {
        "display_name": "LOCOMO",
        "file_prefix": "locomo",
        "result_root": "Result/LOCOMO",
        "categories": [1, 2, 3, 4],
        "response_file": "result_simplified.json",
        "judged_filename": "{frame}_{file_prefix}_judged.json",
        "statistics_txt_filename": "{frame}_{file_prefix}_statistics.txt",
        "statistics_json_filename": "{frame}_{file_prefix}_statistics.json",
        "include_context_tokens": False,
        "valid_category_hint": "(1,2,3,4)",
        "method_choices": [
            "amem",
            "memochat",
            "memoryos",
            "memos",
            "memtree",
            "zep",
            "mem0",
            "mem0g",
            "memorybank",
            "memgpt",
            "sota",
        ],
    },
    "lme": {
        "display_name": "LONGMEMEVAL",
        "file_prefix": "longmemeval",
        "result_root": "Result/LONGMEMEVAL",
        "categories": [
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
        ],
        "response_file": "result_simplified.json",
        "judged_filename": "{frame}_{file_prefix}_judged.json",
        "statistics_txt_filename": "{frame}_{file_prefix}_statistics.txt",
        "statistics_json_filename": "{frame}_{file_prefix}_statistics.json",
        "include_context_tokens": True,
        "valid_category_hint": "['multi-session', 'temporal-reasoning', 'knowledge-update', 'single-session-user', 'single-session-assistant', 'single-session-preference']",
        "method_choices": [
            "amem",
            "memochat",
            "memoryos",
            "memos",
            "memtree",
            "zep",
            "mem0",
            "mem0g",
            "memorybank",
            "memgpt",
            "sota",
        ],
    },
}


try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")

sentence_model_name = None
sentence_model = None


def load_sentence_model(model_name):
    global sentence_model
    global sentence_model_name

    if sentence_model is not None and sentence_model_name == model_name:
        return sentence_model

    sentence_model = SentenceTransformer(model_name)
    sentence_model_name = model_name
    print(f"SentenceTransformer model loaded successfully: {model_name}")
    return sentence_model


def calculate_rouge_scores(gold_answer, response):
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Failed to calculate ROUGE scores: {e}")
    return metrics


def calculate_bleu_scores(gold_tokens, response_tokens):
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    try:
        smoothing = SmoothingFunction().method1
        weights = [
            (1, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (0.33, 0.33, 0.33, 0),
            (0.25, 0.25, 0.25, 0.25),
        ]
        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")
    return metrics


def calculate_meteor_score_value(gold_tokens, response_tokens):
    try:
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        print(f"Failed to calculate METEOR score: {e}")
        return 0.0


def calculate_semantic_similarity(gold_answer, response, embedding_model_name):
    try:
        model = load_sentence_model(embedding_model_name)
        gold_embedding = model.encode([gold_answer], show_progress_bar=False)[0]
        response_embedding = model.encode([response], show_progress_bar=False)[0]
        return 1 - cosine(gold_embedding, response_embedding)
    except Exception as e:
        print(f"Failed to calculate semantic similarity: {e}")
        return 0.0


def calculate_f1_score(gold_tokens, response_tokens):
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)
        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0
        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


def calculate_nlp_metrics(
    gold_answer,
    response,
    context="",
    include_context_tokens=False,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
):
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    metrics = {}

    if include_context_tokens:
        metrics["context_tokens"] = len(nltk.word_tokenize(context)) if context else 0

    gold_tokens = nltk.word_tokenize(gold_answer.lower())
    response_tokens = nltk.word_tokenize(response.lower())
    metrics["lexical"] = {
        "f1": calculate_f1_score(gold_tokens, response_tokens),
        "meteor": calculate_meteor_score_value(gold_tokens, response_tokens),
    }
    metrics["lexical"].update(calculate_rouge_scores(gold_answer, response))
    metrics["lexical"].update(calculate_bleu_scores(gold_tokens, response_tokens))

    metrics["semantic"] = {}
    metrics["semantic"]["similarity"] = calculate_semantic_similarity(
        gold_answer,
        response,
        embedding_model_name,
    )
    _, _, f1 = bert_score(
        [gold_answer], [response], lang="en", rescale_with_baseline=True, verbose=False
    )
    metrics["semantic"]["bert_f1"] = f1.item() if f1 is not None else 0.0

    return metrics


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def flatten_scores(nlp_metrics):
    scores = {}
    if "lexical" in nlp_metrics:
        scores.update(nlp_metrics["lexical"])
    if "semantic" in nlp_metrics:
        scores.update(nlp_metrics["semantic"])
    return scores


def calculate_statistics(all_grades, categories):
    statistics = {
        "sample_statistics": {},
        "category_statistics": {cat: [] for cat in categories},
        "overall_statistics": {},
        "summary": {},
    }

    for sample_id, graded_responses in all_grades.items():
        sample_stats = {
            "total_questions": len(graded_responses),
            "category_counts": {cat: 0 for cat in categories},
            "category_scores": {cat: [] for cat in categories},
        }

        for response in graded_responses:
            category = response.get("category")
            nlp_metrics = response.get("nlp_metrics", {})
            if category not in categories:
                continue
            sample_stats["category_counts"][category] += 1
            scores = flatten_scores(nlp_metrics)
            sample_stats["category_scores"][category].append(scores)
            statistics["category_statistics"][category].append(scores)

        sample_stats["category_averages"] = {}
        for cat in categories:
            if sample_stats["category_scores"][cat]:
                avg_scores = {}
                metrics_names = sample_stats["category_scores"][cat][0].keys()
                for metric in metrics_names:
                    values = [s[metric] for s in sample_stats["category_scores"][cat] if metric in s]
                    avg_scores[metric] = np.mean(values) if values else 0.0
                sample_stats["category_averages"][cat] = avg_scores
            else:
                sample_stats["category_averages"][cat] = {}

        statistics["sample_statistics"][sample_id] = sample_stats

    statistics["overall_category_averages"] = {}
    for cat in categories:
        if statistics["category_statistics"][cat]:
            avg_scores = {}
            metrics_names = statistics["category_statistics"][cat][0].keys()
            for metric in metrics_names:
                values = [s[metric] for s in statistics["category_statistics"][cat] if metric in s]
                avg_scores[metric] = np.mean(values) if values else 0.0
            statistics["overall_category_averages"][cat] = avg_scores
        else:
            statistics["overall_category_averages"][cat] = {}

    all_scores = []
    for cat_scores in statistics["category_statistics"].values():
        all_scores.extend(cat_scores)

    if all_scores:
        overall_avg = {}
        metrics_names = all_scores[0].keys()
        for metric in metrics_names:
            values = [s[metric] for s in all_scores if metric in s]
            overall_avg[metric] = np.mean(values) if values else 0.0
        statistics["overall_statistics"] = overall_avg

    total_questions = sum(len(responses) for responses in all_grades.values())
    category_totals = {cat: len(statistics["category_statistics"][cat]) for cat in categories}
    statistics["summary"] = {
        "total_samples": len(all_grades),
        "total_questions": total_questions,
        "category_totals": category_totals,
        "questions_per_sample": total_questions / len(all_grades) if all_grades else 0,
    }
    return statistics


def save_statistics_table(statistics, output_path, dataset_label, categories):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{dataset_label} EVALUATION DETAILED STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        summary = statistics["summary"]
        f.write(f"Total Samples: {summary['total_samples']}\n")
        f.write(f"Total Questions: {summary['total_questions']}\n")
        f.write(f"Average Questions per Sample: {summary['questions_per_sample']:.2f}\n\n")

        f.write("Questions by Category:\n")
        for cat, count in summary["category_totals"].items():
            f.write(f"  Category {cat}: {count} questions\n")
        f.write("\n")

        f.write("OVERALL AVERAGE SCORES BY CATEGORY\n")
        f.write("-" * 40 + "\n")
        for cat in categories:
            f.write(f"\nCategory {cat}:\n")
            if statistics["overall_category_averages"].get(cat):
                for metric, score in statistics["overall_category_averages"][cat].items():
                    f.write(f"  {metric}: {score:.4f}\n")
            else:
                f.write("  No data available\n")

        f.write("\nOVERALL AVERAGE SCORES (ALL CATEGORIES)\n")
        f.write("-" * 40 + "\n")
        if statistics["overall_statistics"]:
            for metric, score in statistics["overall_statistics"].items():
                f.write(f"{metric}: {score:.4f}\n")
        else:
            f.write("No data available\n")

        f.write("\nSAMPLE-WISE STATISTICS\n")
        f.write("-" * 40 + "\n")
        for sample_id, sample_stats in statistics["sample_statistics"].items():
            f.write(f"\nSample: {sample_id}\n")
            f.write(f"  Total Questions: {sample_stats['total_questions']}\n")
            f.write("  Category Distribution:\n")
            for cat, count in sample_stats["category_counts"].items():
                f.write(f"    Category {cat}: {count} questions\n")

            f.write("  Average Scores by Category:\n")
            for cat in categories:
                if sample_stats["category_averages"].get(cat):
                    f.write(f"    Category {cat}:\n")
                    for metric, score in sample_stats["category_averages"][cat].items():
                        f.write(f"      {metric}: {score:.4f}\n")
                elif sample_stats["category_counts"][cat] > 0:
                    f.write(f"    Category {cat}: No scores available\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF STATISTICS\n")
        f.write("=" * 80 + "\n")


def build_retrieved_context(retrieved, include_context_tokens):
    if not include_context_tokens:
        return ""
    if isinstance(retrieved, list):
        return " ".join(str(item) for item in retrieved)
    if isinstance(retrieved, dict):
        merged = []
        for values in retrieved.values():
            if isinstance(values, list):
                merged.extend(str(item) for item in values)
        return " ".join(merged)
    return str(retrieved) if retrieved else ""


def process_group_responses(sample_id, sample_data, dataset_config, embedding_model_name):
    graded_responses = []
    categories = dataset_config["categories"]
    qa_list = sample_data.get("qa", [])

    for qa_item in tqdm(qa_list, desc=f"Processing sample {sample_id}"):
        question = qa_item.get("question")
        ground_truth = qa_item.get("answer")
        category = qa_item.get("category")
        response = qa_item.get("response", "")
        retrieved = qa_item.get("retrieved", [])

        if category not in categories or ground_truth is None:
            continue

        context = build_retrieved_context(retrieved, dataset_config["include_context_tokens"])
        nlp_metrics = calculate_nlp_metrics(
            ground_truth,
            response,
            context=context,
            include_context_tokens=dataset_config["include_context_tokens"],
            embedding_model_name=embedding_model_name,
        )

        graded_responses.append(
            {
                "question": question,
                "answer": response,
                "golden_answer": ground_truth,
                "category": category,
                "nlp_metrics": nlp_metrics,
                "response_duration_ms": 0.0,
                "search_duration_ms": 0.0,
                "total_duration_ms": 0.0,
                "retrieved": retrieved,
            }
        )

    return sample_id, graded_responses


def process_single_group(sample_id, sample_data, dataset_config, embedding_model_name):
    try:
        start_time = time.time()
        result = process_group_responses(
            sample_id,
            sample_data,
            dataset_config,
            embedding_model_name,
        )
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Sample {sample_id} processed in {elapsed_time} seconds")
        return result
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return sample_id, []


def build_results_dir(current_dir, dataset_config, method, version):
    results_dir = os.path.join(current_dir, dataset_config["result_root"], method)
    if version:
        results_dir = os.path.join(results_dir, version)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def build_output_paths(results_dir, dataset_config, method, version):
    response_path = os.path.join(results_dir, dataset_config["response_file"])
    judged_path = os.path.join(
        results_dir,
        dataset_config["judged_filename"].format(
            frame=method,
            file_prefix=dataset_config["file_prefix"],
        ),
    )
    statistics_txt_path = os.path.join(
        results_dir,
        dataset_config["statistics_txt_filename"].format(
            frame=method,
            file_prefix=dataset_config["file_prefix"],
        ),
    )
    statistics_json_path = os.path.join(
        results_dir,
        dataset_config["statistics_json_filename"].format(
            frame=method,
            file_prefix=dataset_config["file_prefix"],
        ),
    )
    return response_path, judged_path, statistics_txt_path, statistics_json_path


def main(dataset, method, version=None, embedding_model_name=DEFAULT_EMBEDDING_MODEL):
    dataset_config = DATASET_CONFIGS[dataset]
    dataset_label = dataset_config["display_name"]
    categories = dataset_config["categories"]

    print(f"\n=== Starting {dataset_label} evaluation for {method} ===")
    print(f"Using embedding model: {embedding_model_name}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = build_results_dir(current_dir, dataset_config, method, version)
    response_path, judged_path, statistics_path, statistics_json_path = build_output_paths(
        results_dir,
        dataset_config,
        method,
        version,
    )

    with open(response_path, "r", encoding="utf-8") as file:
        responses = json.load(file)

    all_grades = {}
    total_qa_count = 0
    for sample in responses:
        qa_list = sample.get("qa", [])
        valid_qa = [qa for qa in qa_list if qa.get("category") in categories]
        total_qa_count += len(valid_qa)

    print(
        f"Found {total_qa_count} total qa items with valid categories "
        f"{dataset_config['valid_category_hint']} to evaluate"
    )

    active_samples = 0
    for sample in responses:
        sample_id = sample.get("sample_id")
        if not sample_id:
            print("Warning: Found sample without sample_id, skipping")
            continue

        qa_list = sample.get("qa", [])
        if not qa_list:
            print(f"No qa items found for sample {sample_id}")
            continue

        valid_qa = [qa for qa in qa_list if qa.get("category") in categories]
        if not valid_qa:
            print(f"No valid category qa items found for sample {sample_id}")
            continue

        active_samples += 1
        sample_id, graded_responses = process_single_group(
            sample_id,
            sample,
            dataset_config,
            embedding_model_name,
        )
        all_grades[sample_id] = graded_responses

    print(f"Processed {active_samples} samples with valid qa items")
    print("\n=== Evaluation Complete ===")

    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w", encoding="utf-8") as f:
        json.dump(all_grades, f, indent=2)
        print(f"Saved detailed evaluation results to {judged_path}")

    statistics = calculate_statistics(all_grades, categories)
    save_statistics_table(statistics, statistics_path, dataset_label, categories)
    print(f"Saved detailed statistics to {statistics_path}")

    with open(statistics_json_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
        print(f"Saved statistics JSON to {statistics_json_path}")

    category_stats = {cat: 0 for cat in categories}
    for graded_responses in all_grades.values():
        for response in graded_responses:
            category = response.get("category")
            if category in category_stats:
                category_stats[category] += 1

    print("\nCategory statistics:")
    for cat, count in category_stats.items():
        print(f"Category {cat}: {count} items")
    print(f"Total evaluated items: {sum(category_stats.values())}")

    print("\nOverall Average Scores by Category:")
    for cat in categories:
        category_scores = statistics["overall_category_averages"].get(cat, {})
        if category_scores:
            print(f"Category {cat}:")
            for metric, score in category_scores.items():
                print(f"  {metric}: {score:.4f}")
        else:
            print(f"Category {cat}: No data available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset profile to evaluate: 'loco' or 'lme'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method name under the result directory.",
    )
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name or local path used for semantic similarity.",
    )
    args = parser.parse_args()

    valid_methods = DATASET_CONFIGS[args.dataset]["method_choices"]
    if args.method not in valid_methods:
        parser.error(
            f"Invalid --method '{args.method}' for dataset '{args.dataset}'. "
            f"Valid choices: {valid_methods}"
        )

    main(args.dataset, args.method, args.version, args.embedding_model)
