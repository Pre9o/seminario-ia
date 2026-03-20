import re
import numpy as np
import sys


def parse_runs(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    runs = content.strip().split("\n\n")
    parsed = []

    for block in runs:
        lines = block.strip().split("\n")
        run_data = {}
        for line in lines:
            if line.startswith("run:"):
                run_data["run"] = int(line.split(":")[1].strip())
            elif line.startswith("accuracy:"):
                run_data["accuracy"] = float(line.split(":")[1].strip())
            elif line.startswith("auc_roc:"):
                run_data["auc_roc"] = float(line.split(":")[1].strip())
            elif line.startswith("brier_score:"):
                run_data["brier_score"] = float(line.split(":")[1].strip())
            elif line.startswith("f1_macro:"):
                run_data["f1_macro"] = float(line.split(":")[1].strip())
            elif line.startswith("precision_macro:"):
                run_data["precision_macro"] = float(line.split(":")[1].strip())
            elif line.startswith("recall_macro:"):
                run_data["recall_macro"] = float(line.split(":")[1].strip())

        if run_data:
            parsed.append(run_data)

    return parsed


def get_five_best_runs(parsed_runs, sort_metric="f1_macro"):
    sorted_runs = sorted(parsed_runs, key=lambda x: x[sort_metric], reverse=True)
    return sorted_runs[:5]


def compute_stats(runs):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "brier_score", "auc_roc"]
    labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)", "Brier Score Loss", "AUC-ROC"]

    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 47)

    for metric, label in zip(metrics, labels):
        values = np.array([r[metric] for r in runs])
        print(f"{label:<25} {values.mean():>10.4f} ± {values.std():>10.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python five_best_runs.py <path_to_classification_metrics_runs.txt> [sort_metric]")
        print("  sort_metric: metric used to select top 5 (default: f1_macro)")
        print("  options: accuracy, precision_macro, recall_macro, f1_macro, brier_score, auc_roc")
        sys.exit(1)

    filepath = sys.argv[1]
    sort_metric = sys.argv[2] if len(sys.argv) > 2 else "f1_macro"

    parsed = parse_runs(filepath)
    best_runs = get_five_best_runs(parsed, sort_metric)

    print(f"\nTop 5 runs by {sort_metric}:")
    print(f"Runs: {[r['run'] for r in best_runs]}\n")
    compute_stats(best_runs)


if __name__ == "__main__":
    main()
