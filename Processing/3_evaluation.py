import os
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
# Main metrics file (for overall correlation)
METRICS_CSV = os.path.join(BASE_DIR, "data/keywords_metrics.csv")
# Split dataset files (for evaluation and distribution analysis)
DEV_OUTPUT = os.path.join(BASE_DIR, "data/data_dev.jsonl")
TRAIN_OUTPUT = os.path.join(BASE_DIR, "data/data_train.jsonl")

# --- Output Files ---
CORRELATION_PLOT_PATH = os.path.join(BASE_DIR, "data/metrics_correlation.png")
DISTRIBUTION_PLOT_PATH = os.path.join(BASE_DIR, "data/gramm_score_distribution.png")

# --- Parameters ---
# Threshold for binarizing ground truth scores for Average Precision calculation.
AP_GROUND_TRUTH_THRESHOLD = 2.5 # e.g., scores > 2 are "positive"

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

def plot_distribution(dev_path: str, train_path: str, output_path: str):
    """
    Plots the distribution of 'gramm_score' for the dev and train sets.
    """
    print("\n--- 2. Ground Truth Score Distribution in Splits ---")
    dev_df = pd.read_json(dev_path, lines=True)
    train_df = pd.read_json(train_path, lines=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Gramm Score Distribution (Sentence Count)", fontsize=16)
    
    # Dev set plot
    dev_counts = dev_df['gramm_score'].value_counts().sort_index()
    ax1.bar([1, 2, 3, 4], [dev_counts.get(i, 0) for i in [1, 2, 3, 4]], color='steelblue')
    ax1.set_title(f'Dev Set ({len(dev_df)} sentences)')
    ax1.set_xlabel('Gramm Score')
    ax1.set_ylabel('Count of Sentences')
    ax1.set_xticks([1, 2, 3, 4])
    
    # Train set plot
    train_counts = train_df['gramm_score'].value_counts().sort_index()
    ax2.bar([1, 2, 3, 4], [train_counts.get(i, 0) for i in [1, 2, 3, 4]], color='sandybrown')
    ax2.set_title(f'Train Set ({len(train_df)} sentences)')
    ax2.set_xlabel('Gramm Score')
    ax2.set_xticks([1, 2, 3, 4])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)  # Show the plot without blocking the script
    plt.savefig(output_path, dpi=300)
    print(f"Distribution plot saved to {output_path}")
    plt.close()

def evaluate_ranking_metrics(df: pd.DataFrame):
    """
    Calculates Spearman's Rho and Average Precision for various metrics
    against the ground truth score on the provided dataframe.
    """
    print("\n--- 3. Metric Performance on the Dev Set ---")
    
    # The dev set might contain multiple sentences for one keyword.
    # For metric evaluation, we only need one entry per keyword.
    eval_df = df.drop_duplicates(subset=['keyword']).copy()
    print(f"Evaluating on {len(eval_df)} unique keywords from the dev set.")
    
    # Filter for keywords that have a ground truth score
    eval_df.dropna(subset=['gramm_score'], inplace=True)
    if len(eval_df) < 2:
        print("Not enough data with ground truth scores to evaluate.")
        return

    ground_truth = eval_df['gramm_score']
    metrics_to_evaluate = [
        'word_entropy', 
        'occurrences', 
        'collocation_strength', 
        'amount_distinct_neighbors'
    ]

    for metric in metrics_to_evaluate:
        if metric not in eval_df.columns:
            continue
        
        print(f"\n--- Evaluating metric: '{metric}' ---")
        scores = eval_df[metric]

        # 1. Spearman's Rank-Order Correlation (ρ)
        rho, p_value = spearmanr(ground_truth, scores)
        print(f"Spearman's Rank Correlation (ρ): {rho:.4f} (p-value: {p_value:.4f})")
        if p_value < 0.05:
            print("  -> Correlation is statistically significant.")
        else:
            print("  -> Correlation is not statistically significant.")

        # 2. Average Precision (AP)
        y_true = (ground_truth >= AP_GROUND_TRUTH_THRESHOLD).astype(int)
        if len(np.unique(y_true)) < 2:
            print(f"  -> Could not calculate AP: all ground truth scores are on one side of the threshold ({AP_GROUND_TRUTH_THRESHOLD}).")
            continue

        ap_score = average_precision_score(y_true, scores)
        print(f"Average Precision (AP): {ap_score:.4f} (GT threshold >= {AP_GROUND_TRUTH_THRESHOLD})")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    # --- Check for required files ---
    required_files = [METRICS_CSV, DEV_OUTPUT, TRAIN_OUTPUT]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found at {f}")
            print("Please run '1_calculate_metrics.py' and '2_create_dataset.py' first.")
            return

    # --- 1. Analyze properties of ALL metrics across the entire keyword set ---
    df_metrics = pd.read_csv(METRICS_CSV)
    plot_correlation_heatmap(df_metrics, CORRELATION_PLOT_PATH)
    
    # --- 2. Analyze the properties of the data splits ---
    plot_distribution(DEV_OUTPUT, TRAIN_OUTPUT, DISTRIBUTION_PLOT_PATH)

    # --- 3. Evaluate metric performance on the held-out DEV set ---
    dev_df = pd.read_json(DEV_OUTPUT, lines=True)
    evaluate_ranking_metrics(dev_df)


if __name__ == "__main__":
    main()