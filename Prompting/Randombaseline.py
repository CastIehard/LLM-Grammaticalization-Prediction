import json
import pandas as pd
import random
import itertools
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

def run_random_baseline(input_file, output_csv, possible_scores=[1, 2, 3, 4]):
    with open(input_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    for item in test_data:
        item["pred_random"] = random.choice(possible_scores)

    df = pd.DataFrame(test_data)
    df = df.dropna(subset=["gramm_score"])
    df["gramm_score"] = df["gramm_score"].astype(int)

    keyword_df = df.groupby("keyword").agg({
        "gramm_score": "first",
        "pred_random": "mean"
    }).reset_index()

    results = []
    truth = keyword_df["gramm_score"]
    pred = keyword_df["pred_random"]

    try:
        rho, _ = spearmanr(truth, pred)
        results.append({"Evaluation": "Spearman's ρ (rank)", "Random": f"{rho:.2f}"})
    except:
        results.append({"Evaluation": "Spearman's ρ (rank)", "Random": "NaN"})

    degrees = sorted(truth.unique())
    for d1, d2 in itertools.combinations(degrees, 2):
        subset = keyword_df[keyword_df["gramm_score"].isin([d1, d2])]
        y_true = (subset["gramm_score"] == d2).astype(int)
        if y_true.nunique() < 2:
            continue
        try:
            ap = average_precision_score(y_true, subset["pred_random"])
            results.append({
                "Evaluation": f"AP (degrees {d1} vs. {d2})",
                "Random": f"{ap:.2f}"
            })
        except:
            results.append({
                "Evaluation": f"AP (degrees {d1} vs. {d2})",
                "Random": "NaN"
            })

    preds = pred.round().astype(int)
    acc = (preds == truth).mean()
    results.append({"Evaluation": "Accuracy (Exact Match)", "Random": f"{acc:.2f}"})

    for level in sorted(truth.unique()):
        mask = truth == level
        level_acc = (preds[mask] == level).mean()
        results.append({
            "Evaluation": f"Accuracy (Level {level})",
            "Random": f"{level_acc:.2f}"})

    eval_df = pd.DataFrame(results).set_index("Evaluation")
    eval_df.to_csv(output_csv)
    return eval_df

# === Run for TinyLlama
random_eval_df = run_random_baseline(
    input_file="test_set.jsonl",
    output_csv="evaluation_summary_random_baseline_tinyllama.csv"
)

print("\n=== Random Baseline Evaluation (TinyLlama) ===")
print(random_eval_df)