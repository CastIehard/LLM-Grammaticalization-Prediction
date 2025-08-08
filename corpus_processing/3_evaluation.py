import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#go one level up to the main project directory because this script is in corpus_processing
BASE_DIR = os.path.dirname(BASE_DIR)

# --- Input Files (Keyword Metrics CSV Files) ---
DEV_DATA_1_METRICS_CSV = os.path.join(BASE_DIR, "data/dev data 1 (for prompting)/dev_data_1_prompting_metrics.csv")
DEV_DATA_2_METRICS_CSV = os.path.join(BASE_DIR, "data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv")
TEST_DATA_METRICS_CSV = os.path.join(BASE_DIR, "data/test data (only use at the end)/test_data_metrics.csv")
FULL_DATA_METRICS_CSV = os.path.join(BASE_DIR, "data/full data (only for storing, do not use)/keywords_metrics_full.csv")

# --- Output Files (Distribution plots in each folder) ---
DEV_DATA_1_PLOT = os.path.join(BASE_DIR, "data/dev data 1 (for prompting)/gramm_score_distribution_dev1_prompting.png")
DEV_DATA_2_PLOT = os.path.join(BASE_DIR, "data/dev data 2 (for testing)/gramm_score_distribution_dev2_testing.png")
TEST_DATA_PLOT = os.path.join(BASE_DIR, "data/test data (only use at the end)/gramm_score_distribution_test.png")
FULL_DATA_PLOT = os.path.join(BASE_DIR, "data/full data (only for storing, do not use)/gramm_score_distribution_full.png")

# --- Output Files (Evaluation tables) ---
DEV_DATA_2_EVALUATION_CSV = os.path.join(BASE_DIR, "data/dev data 2 (for testing)/evaluation_summary_table_dev2_testing.csv")
TEST_DATA_EVALUATION_CSV = os.path.join(BASE_DIR, "data/test data (only use at the end)/evaluation_summary_table_test.csv")
FULL_DATA_EVALUATION_CSV = os.path.join(BASE_DIR, "data/full data (only for storing, do not use)/evaluation_summary_table_full.csv")

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

def plot_single_distribution(df: pd.DataFrame, dataset_name: str, output_path: str):
    """Plots the distribution of 'gramm_score' for a single dataset."""
    print(f"\n--- Distribution for {dataset_name} ---")
    
    if 'gramm_score' not in df.columns:
        print(f"Error: 'gramm_score' column not found in {dataset_name}")
        return
    
    # Count keywords by gramm_score
    score_counts = df['gramm_score'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(score_counts.index, score_counts.values, color='steelblue', alpha=0.7)
    plt.title(f'Grammaticalization Score Distribution - {dataset_name}\n({len(df)} keywords)', fontsize=14)
    plt.xlabel('Grammaticalization Score')
    plt.ylabel('Number of Keywords')
    plt.xticks(score_counts.index)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {output_path}")
    plt.close()

def generate_evaluation_table(df: pd.DataFrame, metrics_to_evaluate: list, output_path: str, dataset_name: str):
    """
    Generates a table of evaluation results (AP scores, Spearman's Rho) for a
    list of metrics, similar to the format in Schlechtweg et al. (2020).
    """
    print(f"\n--- Metric Performance Evaluation on {dataset_name} ---")
    
    eval_df = df.copy()
    eval_df.dropna(subset=['gramm_score'], inplace=True)
    print(f"Evaluating on {len(eval_df)} unique keywords from the {dataset_name}.")
    
    if len(eval_df) < 2:
        print(f"Not enough data to generate evaluation table for {dataset_name}.")
        return

    results_data = []
    ground_truth = eval_df['gramm_score']
    
    # --- Part 1: Spearman's Rho (Overall Ranking) ---
    spearman_row = {'Evaluation': "Spearman's ρ (rank)"}
    for metric in metrics_to_evaluate:
        if metric not in eval_df.columns: 
            spearman_row[metric] = "N/A"
            continue
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
        
        # Skip if only one class is present in the pair
        if len(np.unique(y_true)) < 2:
            for metric in metrics_to_evaluate:
                ap_row[metric] = "N/A"
            results_data.append(ap_row)
            continue

        for metric in metrics_to_evaluate:
            if metric not in pair_df.columns:
                ap_row[metric] = "N/A"
                continue
            y_scores = pair_df[metric]
            ap_score = average_precision_score(y_true, y_scores)
            ap_row[metric] = f"{ap_score:.2f}"
        
        results_data.append(ap_row)

    # --- Save the results to a CSV file ---
    results_df = pd.DataFrame(results_data)
    results_df.set_index('Evaluation', inplace=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results_df.to_csv(output_path)
    
    print(f"\n--- Evaluation Summary Table for {dataset_name} ---")
    print(results_df)
    print(f"\nSummary table saved to {output_path}")
# =============================================================================
# Main Execution
# =============================================================================
def main():
    # Define dataset configurations
    datasets = [
        {
            'name': 'Dev Data 1 (for prompting)',
            'metrics_csv': DEV_DATA_1_METRICS_CSV,
            'plot_output': DEV_DATA_1_PLOT,
            'eval_output': None  # No evaluation for dev data 1 (few-shot prompting)
        },
        {
            'name': 'Dev Data 2 (for testing)',
            'metrics_csv': DEV_DATA_2_METRICS_CSV,
            'plot_output': DEV_DATA_2_PLOT,
            'eval_output': DEV_DATA_2_EVALUATION_CSV
        },
        {
            'name': 'Test Data',
            'metrics_csv': TEST_DATA_METRICS_CSV,
            'plot_output': TEST_DATA_PLOT,
            'eval_output': TEST_DATA_EVALUATION_CSV
        },
        {
            'name': 'Full Data',
            'metrics_csv': FULL_DATA_METRICS_CSV,
            'plot_output': FULL_DATA_PLOT,
            'eval_output': FULL_DATA_EVALUATION_CSV
        }
    ]
    
    print("=== Keyword Metrics Analysis and Evaluation ===")
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset['name']}")
        print(f"{'='*60}")
        
        # Check if metrics file exists
        if not os.path.exists(dataset['metrics_csv']):
            print(f"Warning: Metrics file not found at {dataset['metrics_csv']}")
            print(f"Skipping {dataset['name']}")
            continue
        
        # Load the metrics CSV
        try:
            df = pd.read_csv(dataset['metrics_csv'])
            print(f"Loaded {len(df)} keywords from {dataset['metrics_csv']}")
        except Exception as e:
            print(f"Error loading {dataset['metrics_csv']}: {e}")
            continue
        
        # Generate distribution plot
        plot_single_distribution(df, dataset['name'], dataset['plot_output'])
        
        # Generate evaluation table (only for datasets that need evaluation)
        if dataset['eval_output'] is not None:
            generate_evaluation_table(df, METRICS_TO_EVALUATE, dataset['eval_output'], dataset['name'])
        else:
            print(f"Skipping evaluation for {dataset['name']} (used for few-shot prompting)")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()