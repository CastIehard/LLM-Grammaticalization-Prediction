# 3_evaluate_metrics.py

import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Input Files ---
METRICS_CSV = os.path.join(BASE_DIR, "data/keywords_metrics.csv")
DEV_OUTPUT = os.path.join(BASE_DIR, "data/data_dev.jsonl")
TEST_OUTPUT = os.path.join(BASE_DIR, "data/data_test.jsonl")

# --- Output Files ---
CORRELATION_PLOT_PATH = os.path.join(BASE_DIR, "data/metrics_correlation.png")
DISTRIBUTION_PLOT_PATH = os.path.join(BASE_DIR, "data/gramm_score_distribution.png")
# NEW: Path for the final evaluation table
EVALUATION_TABLE_CSV = os.path.join(BASE_DIR, "data/evaluation_summary_table.csv")

# --- Metrics to Evaluate ---
# These column names must match the ones in keywords_metrics.csv
METRICS_TO_EVALUATE = [
    'occurrences',
    'word_entropy',
    'amount_distinct_neighbors',
    'collocation_strength',
    'synthetic_context_adversity',
    'avg_character_count'
]

# =============================================================================
# Analysis & Evaluation Functions
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Plots and saves a correlation heatmap of all numeric metric columns."""
    print("\n--- 1. Correlation Heatmap (All Keywords) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation plot.")
        return
    
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", square=True, mask=mask,
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={"shrink": .8})
    plt.title("Keyword Metrics Correlation Matrix (All Keywords)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to {output_path}")
    plt.close()

def plot_distribution(dev_path: str, test_path: str, output_path: str):
    """Plots the distribution of 'gramm_score' for the dev and test sets."""
    print("\n--- 2. Ground Truth Score Distribution in Splits ---")
    dev_df = pd.read_json(dev_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Gramm Score Distribution (Sentence Count)", fontsize=16)
    
    dev_counts = dev_df['gramm_score'].value_counts().sort_index()
    ax1.bar(dev_counts.index, dev_counts.values, color='steelblue')
    ax1.set_title(f'Dev Set ({len(dev_df)} sentences)')
    ax1.set_xlabel('Gramm Score'); ax1.set_ylabel('Count of Sentences'); ax1.set_xticks(dev_counts.index)
    
    test_counts = test_df['gramm_score'].value_counts().sort_index()
    ax2.bar(test_counts.index, test_counts.values, color='sandybrown')
    ax2.set_title(f'test Set ({len(test_df)} sentences)')
    ax2.set_xlabel('Gramm Score'); ax2.set_xticks(test_counts.index)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Distribution plot saved to {output_path}")
    plt.close()

# *** NEW COMPREHENSIVE EVALUATION FUNCTION ***
def generate_evaluation_table(df: pd.DataFrame, metrics_to_evaluate: list, output_path: str):
    """
    Generates a table of evaluation results (AP scores, Spearman's Rho) for a
    list of metrics, similar to the format in Schlechtweg et al. (2020).
    """
    print("\n--- 3. Metric Performance Evaluation on Dev Set ---")
    
    # Use one entry per keyword for metric evaluation
    eval_df = df.drop_duplicates(subset=['keyword']).copy()
    eval_df.dropna(subset=['gramm_score'], inplace=True)
    print(f"Evaluating on {len(eval_df)} unique keywords from the dev set.")
    
    if len(eval_df) < 2:
        print("Not enough data to generate evaluation table.")
        return

    results_data = []
    ground_truth = eval_df['gramm_score']
    
    # --- Part 1: Spearman's Rho (Overall Ranking) ---
    spearman_row = {'Evaluation': "Spearman's ρ (rank)"}
    for metric in metrics_to_evaluate:
        if metric not in eval_df.columns: continue
        scores = eval_df[metric]
        rho, _ = spearmanr(ground_truth, scores)
        spearman_row[metric] = f"{rho:.2f}"
    results_data.append(spearman_row)

    # --- Part 2: Average Precision (Pairwise Degree Discrimination) ---
    degrees = sorted(eval_df['gramm_score'].unique())
    degree_pairs = list(itertools.combinations(degrees, 2))

    for deg1, deg2 in degree_pairs:
        ap_row = {'Evaluation': f'AP (degrees {int(deg1)} vs. {int(deg2)})'}
        
        # Filter for the pair of degrees
        pair_df = eval_df[eval_df['gramm_score'].isin([deg1, deg2])].copy()
        
        # Set the higher degree as the "positive" class (1)
        y_true = (pair_df['gramm_score'] == deg2).astype(int)
        
        # Skip if only one class is present in the pair (shouldn't happen with this logic)
        if len(np.unique(y_true)) < 2:
            continue

        for metric in metrics_to_evaluate:
            if metric not in pair_df.columns: continue
            y_scores = pair_df[metric]
            ap_score = average_precision_score(y_true, y_scores)
            ap_row[metric] = f"{ap_score:.2f}"
        
        results_data.append(ap_row)

    # --- Save the results to a CSV file ---
    results_df = pd.DataFrame(results_data)
    results_df.set_index('Evaluation', inplace=True)
    results_df.to_csv(output_path)
    
    print("\n--- Evaluation Summary Table ---")
    print(results_df)
    print(f"\nSummary table saved to {output_path}")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    required_files = [METRICS_CSV, DEV_OUTPUT, TEST_OUTPUT]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found at {f}")
            print("Please run '1_calculate_metrics.py' and '2_create_dataset.py' first.")
            return

    # 1. Analyze properties of ALL metrics across the entire keyword set
    df_metrics = pd.read_csv(METRICS_CSV)
    plot_correlation_heatmap(df_metrics, CORRELATION_PLOT_PATH)
    
    # 2. Analyze the properties of the data splits
    plot_distribution(DEV_OUTPUT, TEST_OUTPUT, DISTRIBUTION_PLOT_PATH)

    # 3. Generate the final evaluation summary table on the DEV set
    dev_df = pd.read_json(DEV_OUTPUT, lines=True)
    generate_evaluation_table(dev_df, METRICS_TO_EVALUATE, EVALUATION_TABLE_CSV)

if __name__ == "__main__":
    main()