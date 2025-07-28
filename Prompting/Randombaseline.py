import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Input Files ---
METRICS_CSV = os.path.join(BASE_DIR, "keywords_metrics.csv")
DEV_OUTPUT = os.path.join(BASE_DIR, "data_dev.jsonl")

# --- Output File ---
EVALUATION_TABLE_CSV = os.path.join(BASE_DIR, "evaluation_summary_table.csv")

# --- Dummy column for random prediction
PRED_COLUMN = "random_prediction"
METRICS_TO_EVALUATE = [PRED_COLUMN]

# =============================================================================
# Evaluation Function
# =============================================================================

def generate_evaluation_table(df: pd.DataFrame, metrics_to_evaluate: list, output_path: str):
    eval_df = df.copy()
    eval_df.dropna(subset=['gramm_score'], inplace=True)

    if len(eval_df) < 2:
        print("Not enough data to evaluate.")
        return

    results_data = []
    ground_truth = eval_df['gramm_score']

    # --- Spearman’s ρ ---
    spearman_row = {'Evaluation': "Spearman's ρ (rank)"}
    for metric in metrics_to_evaluate:
        rho, _ = spearmanr(ground_truth, eval_df[metric])
        spearman_row[metric] = f"{rho:.2f}"
    results_data.append(spearman_row)

    # --- Average Precision (AP) ---
    degrees = sorted(eval_df['gramm_score'].unique())
    degree_pairs = list(itertools.combinations(degrees, 2))

    for d1, d2 in degree_pairs:
        ap_row = {'Evaluation': f'AP (degrees {int(d1)} vs. {int(d2)})'}
        pair_df = eval_df[eval_df['gramm_score'].isin([d1, d2])]
        y_true = (pair_df['gramm_score'] == d2).astype(int)

        if len(np.unique(y_true)) < 2:
            continue

        for metric in metrics_to_evaluate:
            y_score = (pair_df[metric] == d2).astype(int)
            ap = average_precision_score(y_true, y_score)
            ap_row[metric] = f"{ap:.2f}"
        results_data.append(ap_row)

    results_df = pd.DataFrame(results_data)
    results_df.set_index('Evaluation', inplace=True)
    results_df.to_csv(output_path)
    print("\n--- Final Evaluation Summary Table ---\n")
    print(results_df)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Load data
    df_metrics = pd.read_csv(METRICS_CSV)
    dev_df = pd.read_json(DEV_OUTPUT, lines=True)

    # Step 1: Get keywords used in dev set
    dev_keywords = dev_df['keyword'].unique()

    # Step 2: Filter keywords_metrics to only those in dev
    df_filtered = df_metrics[df_metrics['keyword'].isin(dev_keywords)].copy()

    # Step 3: Assign random predictions (1–4)
    np.random.seed(42)  # for reproducibility
    df_filtered[PRED_COLUMN] = np.random.choice([1, 2, 3, 4], size=len(df_filtered))

    print(f"Assigned random grammaticalization class to {len(df_filtered)} keywords")

    # Step 4: Evaluate using original gold gramm_score from keyword_metrics
    generate_evaluation_table(df_filtered, METRICS_TO_EVALUATE, EVALUATION_TABLE_CSV)

if __name__ == "__main__":
    main()
