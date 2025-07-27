import json
import pandas as pd
import itertools
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# === Step 1: Load Test Data ===
INPUT_FILE = "test_set.jsonl"
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

# === Step 2: Find Most Frequent Label (ideally from TRAINING set) ===
most_frequent_label = Counter(
    int(item["gramm_score"]) for item in test_data
).most_common(1)[0][0]
print("Most Frequent Label:", most_frequent_label)

# === Step 3: Simulate Baseline Predictions ===
for item in test_data:
    item["pred_A"] = most_frequent_label  # only using one prediction column

# === Step 4: Build DataFrame and Group by Keyword ===
df = pd.DataFrame(test_data)
df = df.dropna(subset=["gramm_score"])
df["gramm_score"] = df["gramm_score"].astype(int)

# Aggregate to keyword level for evaluation
keyword_df = df.groupby("keyword").agg({
    "gramm_score": "first",
    "pred_A": "mean"
}).reset_index()

# === Step 5: Evaluation Function ===
def evaluate_predictions(df, metric):
    results = []
    truth = df["gramm_score"]
    pred = df[metric]

    # Spearman's ρ
    try:
        rho, _ = spearmanr(truth, pred)
        results.append({"Evaluation": "Spearman's ρ (rank)", metric: f"{rho:.2f}"})
    except:
        results.append({"Evaluation": "Spearman's ρ (rank)", metric: "NaN"})

    # Average Precision for pairwise degrees
    degrees = sorted(df["gramm_score"].unique())
    for d1, d2 in itertools.combinations(degrees, 2):
        subset = df[df["gramm_score"].isin([d1, d2])]
        y_true = (subset["gramm_score"] == d2).astype(int)
        if y_true.nunique() < 2:
            continue
        try:
            ap = average_precision_score(y_true, subset[metric])
            results.append({
                "Evaluation": f"AP (degrees {d1} vs. {d2})",
                metric: f"{ap:.2f}"
            })
        except:
            results.append({
                "Evaluation": f"AP (degrees {d1} vs. {d2})",
                metric: "NaN"
            })

    # Accuracy
    preds = pred.round().astype(int)
    acc = (preds == truth).mean()
    results.append({"Evaluation": "Accuracy (Exact Match)", metric: f"{acc:.2f}"})

    # Per-level accuracy
    for level in sorted(truth.unique()):
        mask = truth == level
        level_acc = (preds[mask] == level).mean()
        results.append({
            "Evaluation": f"Accuracy (Level {level})",
            metric: f"{level_acc:.2f}"
        })

    return pd.DataFrame(results).set_index("Evaluation")

# === Step 6: Run Evaluation and Save Results ===
metrics = ["pred_A"]
eval_df = evaluate_predictions(keyword_df, "pred_A")
eval_df.to_csv("evaluation_summary_most_frequent.csv")

# === Print Results ===
print("\n=== Evaluation Summary ===")
print(eval_df)
