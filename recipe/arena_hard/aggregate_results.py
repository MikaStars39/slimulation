"""
Merge per-category judgments, compute overall score, and output a CSV summary.

Usage:
    python aggregate_results.py \
        --output-dir /path/to/arena_hard \
        --judge-model gpt-5 \
        --model-name MyModel \
        --baseline-model o3-mini-2025-01-31 \
        --categories hard_prompt creative_writing coding math
"""

import argparse
import csv
import json
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def merge_judgments(judge_dir: str, model_name: str, categories: list[str]) -> str:
    """Concatenate per-category judgment files into one overall file."""
    overall_path = os.path.join(judge_dir, f"{model_name}_overall.jsonl")
    with open(overall_path, "w") as out:
        for cat in categories:
            src = os.path.join(judge_dir, f"{model_name}_{cat}.jsonl")
            if not os.path.exists(src):
                print(f"Warning: {src} not found, skipping")
                continue
            with open(src) as f:
                out.write(f.read())
    return overall_path


def run_show_result(judgment_file: str, baseline_model: str, output: str,
                    answer_dir: str = None, control_features: list[str] = None):
    """Call show_result.py as a subprocess."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "show_result.py"),
        "--judgment-file", judgment_file,
        "--baseline-model", baseline_model,
        "--output", output,
    ]
    if control_features:
        cmd += ["--answer-dir", answer_dir, "--control-features"] + control_features
    subprocess.run(cmd, check=True)


def collect_scores(output_dir: str, judge_model: str, model_name: str,
                   categories: list[str]) -> list[dict]:
    """Read per-category + overall leaderboard JSONs and extract target model scores."""
    rows = []
    for cat in categories + ["overall"]:
        path = os.path.join(output_dir, f"leaderboard_{judge_model}_{cat}.json")
        if not os.path.exists(path):
            continue
        for entry in json.load(open(path)):
            if entry["model"] == model_name:
                rows.append({
                    "category": cat,
                    "model": model_name,
                    "score": round(entry["score"], 2),
                    "ci_lower": round(entry["ci_lower"], 2),
                    "ci_upper": round(entry["ci_upper"], 2),
                })
                break
    return rows


def write_csv(rows: list[dict], path: str):
    """Write score rows to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "model", "score", "ci_lower", "ci_upper"])
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict]):
    """Print a compact summary table to stdout."""
    print(f"\n{'Category':<20} {'Score':>8} {'CI':>18}")
    print("-" * 48)
    for r in rows:
        ci = f"(-{r['ci_lower']:.2f} / +{r['ci_upper']:.2f})"
        print(f"{r['category']:<20} {r['score']:>8.2f} {ci:>18}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate arena-hard results into CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--answer-dir", default=None)
    parser.add_argument("--control-features", nargs="+", default=None)
    args = parser.parse_args()

    judge_dir = os.path.join(args.output_dir, "judgments", args.judge_model)

    # Merge per-category judgments and compute overall score
    overall_file = merge_judgments(judge_dir, args.model_name, args.categories)
    overall_output = os.path.join(args.output_dir, f"leaderboard_{args.judge_model}_overall.json")
    run_show_result(overall_file, args.baseline_model, overall_output,
                    args.answer_dir, args.control_features)

    # Collect all scores and write CSV
    rows = collect_scores(args.output_dir, args.judge_model, args.model_name, args.categories)
    csv_path = os.path.join(args.output_dir, f"scores_{args.judge_model}_{args.model_name}.csv")
    write_csv(rows, csv_path)
    print_table(rows)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
