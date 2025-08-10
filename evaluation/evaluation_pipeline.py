#!/usr/bin/env python3
"""
Evaluation Pipeline for Grammaticalization Degree Classification

This script processes all CSV files in the input_csv directory and calculates
essential evaluation metrics for each model's predictions against ground truth.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_DIR = os.path.join(BASE_DIR, "input_csv (only dev2 nothing else)")
OUTPUT_CSV = os.path.join(BASE_DIR, "evaluation_results.csv")
OUTPUT_PLOT = os.path.join(BASE_DIR, "evaluation_plot.png")

# =============================================================================
# Evaluation Functions
# =============================================================================

def calculate_essential_metrics(y_true, y_pred):
    """Calculate only the essential metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Correlation metrics
    spearman_corr, _ = spearmanr(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    metrics['spearman_correlation'] = spearman_corr
    metrics['pearson_correlation'] = pearson_corr
    
    # Error metric
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    return metrics

def evaluate_single_file(file_path):
    """Evaluate a single CSV file and return all metrics."""
    print(f"Evaluating: {os.path.basename(file_path)}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Validate required columns
    required_cols = ['keyword', 'gramm_score', 'predictions']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns in {file_path}: {missing_cols}")
        return None
    
    # Extract ground truth and predictions
    y_true = df['gramm_score'].values
    y_pred = df['predictions'].values
    
    # Handle missing values
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        print(f"No valid data in {file_path}")
        return None
    
    print(f"  Processing {len(y_true)} valid predictions")
    
    # Calculate essential metrics
    results = {'file_name': os.path.basename(file_path).replace('.csv', '')}
    results['n_samples'] = len(y_true)
    
    # Calculate metrics
    metrics = calculate_essential_metrics(y_true, y_pred)
    results.update(metrics)
    
    return results

def create_evaluation_plot(results_df):
    """Create a bar plot showing MSE for all models."""
    plt.figure(figsize=(12, 8))
    
    # Sort by MSE for better visualization
    sorted_df = results_df.sort_values('mse')
    
    # Create bar plot
    bars = plt.bar(range(len(sorted_df)), sorted_df['mse'], 
                   color=['red' if name == 'random' else 'steelblue' for name in sorted_df['file_name']])
    
    # Customize plot
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.title('Model Performance Comparison - Mean Squared Error', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_df)), sorted_df['file_name'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mse) in enumerate(zip(bars, sorted_df['mse'])):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mse:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight random baseline
    if 'random' in sorted_df['file_name'].values:
        random_idx = sorted_df[sorted_df['file_name'] == 'random'].index[0]
        plt.axhline(y=sorted_df.loc[random_idx, 'mse'], color='red', linestyle='--', alpha=0.7, 
                   label=f"Random Baseline (MSE={sorted_df.loc[random_idx, 'mse']:.3f})")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved to: {OUTPUT_PLOT}")
    plt.close()

def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("Grammaticalization Degree Classification Evaluation")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_CSV_DIR):
        print(f"Error: Input directory not found: {INPUT_CSV_DIR}")
        return
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(INPUT_CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {INPUT_CSV_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files to evaluate:")
    for f in csv_files:
        print(f"  - {f}")
    print()
    
    # Evaluate each file
    all_results = []
    for csv_file in sorted(csv_files):
        file_path = os.path.join(INPUT_CSV_DIR, csv_file)
        result = evaluate_single_file(file_path)
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("No valid results to save.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by file name, but put 'random' first if it exists
    if 'random' in results_df['file_name'].values:
        random_row = results_df[results_df['file_name'] == 'random']
        other_rows = results_df[results_df['file_name'] != 'random'].sort_values('file_name')
        results_df = pd.concat([random_row, other_rows], ignore_index=True)
    else:
        results_df = results_df.sort_values('file_name')
    
    # Round numeric columns for better readability
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    
    # Save results
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEvaluation complete! Results saved to: {OUTPUT_CSV}")
    
    # Create evaluation plot
    create_evaluation_plot(results_df)
    
    # Print summary
    print(f"\nSummary of results:")
    print(f"{'File':<20} {'Accuracy':<10} {'F1-Score':<10} {'Spearman ρ':<12} {'Pearson r':<10} {'MSE':<8}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['file_name']:<20} "
              f"{row['accuracy']:<10.4f} "
              f"{row['f1_score']:<10.4f} "
              f"{row['spearman_correlation']:<12.4f} "
              f"{row['pearson_correlation']:<10.4f} "
              f"{row['mse']:<8.4f}")
    
    # Highlight best performing model (excluding random baseline)
    non_random = results_df[results_df['file_name'] != 'random']
    if len(non_random) > 0:
        best_accuracy_idx = non_random['accuracy'].idxmax()
        best_spearman_idx = non_random['spearman_correlation'].idxmax()
        best_mse_idx = non_random['mse'].idxmin()  # Lower MSE is better
        
        print(f"\nBest performing models:")
        print(f"  Highest Accuracy: {results_df.loc[best_accuracy_idx, 'file_name']} "
              f"({results_df.loc[best_accuracy_idx, 'accuracy']:.4f})")
        print(f"  Highest Spearman ρ: {results_df.loc[best_spearman_idx, 'file_name']} "
              f"({results_df.loc[best_spearman_idx, 'spearman_correlation']:.4f})")
        print(f"  Lowest MSE: {results_df.loc[best_mse_idx, 'file_name']} "
              f"({results_df.loc[best_mse_idx, 'mse']:.4f})")

if __name__ == "__main__":
    main()