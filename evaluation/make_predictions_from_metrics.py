import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#go one level up to the main project directory because this script is in corpus_processing
BASE_DIR = os.path.dirname(BASE_DIR)
    # Define file paths
input_file = os.path.join(BASE_DIR, "data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv")
output_dir = os.path.join(BASE_DIR, "evaluation/output")

def calculate_correlation(metric_values, gramm_scores):
    """
    Calculate Spearman correlation between metric and grammaticalization scores.
    
    Args:
        metric_values: Series of metric values
        gramm_scores: Series of grammaticalization scores
    
    Returns:
        correlation coefficient (positive means higher metric = higher gramm score)
    """
    correlation, p_value = spearmanr(metric_values, gramm_scores)
    return correlation, p_value

def find_optimal_thresholds(metric_values, gramm_scores, correlation_sign):
    """
    Find optimal thresholds to separate grammaticalization levels 1-2, 2-3, 3-4.
    
    Args:
        metric_values: Series of metric values
        gramm_scores: Series of grammaticalization scores (1-4)
        correlation_sign: +1 for positive correlation, -1 for negative
    
    Returns:
        dict with thresholds and accuracy
    """
    # Get unique metric values and sort them
    unique_values = sorted(metric_values.unique())
    
    best_thresholds = {'threshold_1_to_2': None, 'threshold_2_to_3': None, 'threshold_3_to_4': None}
    best_accuracy = 0
    
    # Try different combinations of thresholds
    n_values = len(unique_values)
    
    best_t1, best_t2, best_t3 = None, None, None
    
    # Use percentile-based approach for efficiency
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    candidate_thresholds = [np.percentile(metric_values, p) for p in percentiles]
    
    for i, t1 in enumerate(candidate_thresholds):
        for j, t2 in enumerate(candidate_thresholds[i+1:], i+1):
            for k, t3 in enumerate(candidate_thresholds[j+1:], j+1):
                # Create predictions based on thresholds
                predictions = assign_scores_by_thresholds(metric_values, t1, t2, t3, correlation_sign)
                
                # Calculate accuracy
                accuracy = accuracy_score(gramm_scores, predictions)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_t1, best_t2, best_t3 = t1, t2, t3
    
    return {
        'threshold_1_to_2': best_t1,
        'threshold_2_to_3': best_t2, 
        'threshold_3_to_4': best_t3,
        'accuracy': best_accuracy
    }

def assign_scores_by_thresholds(metric_values, t1, t2, t3, correlation_sign):
    """
    Assign grammaticalization scores based on thresholds.
    
    Args:
        metric_values: Series of metric values
        t1, t2, t3: Thresholds for levels 1-2, 2-3, 3-4
        correlation_sign: +1 for positive correlation, -1 for negative
    
    Returns:
        Array of predicted scores (1-4)
    """
    predictions = np.ones(len(metric_values), dtype=int)
    
    if correlation_sign > 0:
        # Positive correlation: higher values = higher scores
        predictions[(metric_values > t1)] = 2
        predictions[(metric_values > t2)] = 3 
        predictions[(metric_values > t3)] = 4
    else:
        # Negative correlation: lower values = higher scores
        predictions[(metric_values < t3)] = 2
        predictions[(metric_values < t2)] = 3
        predictions[(metric_values < t1)] = 4
    
    return predictions

def process_all_metrics(input_file, output_dir):
    """
    Process all metrics: calculate correlations, find optimal thresholds, generate predictions.
    
    Args:
        input_file: Path to dev2 metrics CSV
        output_dir: Directory to save prediction files and threshold summary
    """
    # Read the dev2 metrics
    print(f"Loading dev2 metrics from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Found {len(df)} keywords with {len(df.columns)} columns")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to process (excluding keyword and gramm_score)
    metrics_to_process = [
        'occurrences',
        'avg_character_count', 
        'amount_distinct_neighbors',
        'word_entropy',
        'collocation_strength',
        'synthetic_context_adversity'
    ]
    
    # Store results for threshold summary
    threshold_results = []
    metric_summaries = []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(metrics_to_process)} metrics...")
    print(f"{'='*60}")
    
    # Main loop: iterate over each metric
    for metric in metrics_to_process:
        print(f"\n--- Processing metric: {metric} ---")
        
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data, skipping...")
            continue
        
        # Prepare data for this metric
        metric_df = df[['keyword', 'gramm_score', metric]].copy()
        metric_df = metric_df.dropna(subset=[metric, 'gramm_score'])
        
        if len(metric_df) == 0:
            print(f"No valid data for metric {metric}, skipping...")
            continue
        
        print(f"Keywords with valid data: {len(metric_df)}")
        print(f"Metric range: {metric_df[metric].min():.3f} - {metric_df[metric].max():.3f}")
        
        # Step 1: Calculate correlation with grammaticalization scores
        correlation, p_value = calculate_correlation(metric_df[metric], metric_df['gramm_score'])
        correlation_sign = 1 if correlation > 0 else -1
        
        print(f"Spearman correlation with gramm_score: {correlation:.3f} (p={p_value:.3f})")
        print(f"Correlation direction: {'Positive' if correlation_sign > 0 else 'Negative'}")
        
        # Step 2: Find optimal thresholds
        print("Finding optimal thresholds...")
        threshold_info = find_optimal_thresholds(
            metric_df[metric], 
            metric_df['gramm_score'], 
            correlation_sign
        )
        
        print(f"Optimal thresholds found:")
        print(f"  Level 1→2: {threshold_info['threshold_1_to_2']:.3f}")
        print(f"  Level 2→3: {threshold_info['threshold_2_to_3']:.3f}")
        print(f"  Level 3→4: {threshold_info['threshold_3_to_4']:.3f}")
        print(f"  Threshold-based accuracy: {threshold_info['accuracy']:.3f}")
        
        # Step 3: Generate predictions using optimal thresholds
        predictions = assign_scores_by_thresholds(
            metric_df[metric],
            threshold_info['threshold_1_to_2'],
            threshold_info['threshold_2_to_3'], 
            threshold_info['threshold_3_to_4'],
            correlation_sign
        )
        
        metric_df['predictions'] = predictions
        
        # Step 4: Save individual metric predictions CSV
        output_file = os.path.join(output_dir, f"{metric}_dev2.csv")
        metric_df.to_csv(output_file, index=False)
        
        # Calculate statistics
        true_counts = metric_df['gramm_score'].value_counts().sort_index()
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        
        print(f"True distribution: {dict(true_counts)}")
        print(f"Pred distribution: {dict(pred_counts)}")
        print(f"Saved predictions to: {output_file}")
        
        # Store threshold information for summary CSV
        threshold_results.append({
            'metric': metric,
            'correlation': correlation,
            'correlation_direction': 'positive' if correlation_sign > 0 else 'negative',
            'threshold_1_to_2': threshold_info['threshold_1_to_2'],
            'threshold_2_to_3': threshold_info['threshold_2_to_3'],
            'threshold_3_to_4': threshold_info['threshold_3_to_4'],
            'threshold_accuracy': threshold_info['accuracy']
        })
        
        # Store summary statistics
        metric_summaries.append({
            'metric': metric,
            'keywords_processed': len(metric_df),
            'correlation': correlation,
            'threshold_accuracy': threshold_info['accuracy'],
            'metric_min': metric_df[metric].min(),
            'metric_max': metric_df[metric].max(),
            'output_file': output_file
        })
    
    # Step 5: Save threshold summary CSV
    threshold_df = pd.DataFrame(threshold_results)
    threshold_summary_file = os.path.join(output_dir, "metric_thresholds_summary.csv")
    threshold_df.to_csv(threshold_summary_file, index=False)
    
    # Step 6: Save general summary CSV  
    summary_df = pd.DataFrame(metric_summaries)
    summary_file = os.path.join(output_dir, "metric_predictions_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Processed {len(metric_summaries)} metrics")
    print(f"Threshold summary saved to: {threshold_summary_file}")
    print(f"General summary saved to: {summary_file}")
    
    print(f"\nThreshold-based accuracies:")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']}: {row['threshold_accuracy']:.3f}")
    
    print(f"\nCorrelations with grammaticalization:")
    for _, row in summary_df.iterrows():
        direction = "↑" if row['correlation'] > 0 else "↓"
        print(f"  {row['metric']}: {row['correlation']:.3f} {direction}")
    
    return summary_df, threshold_df


def main():
    """Main function to generate metric-based predictions with optimal thresholds."""


    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return
    
    try:
        summary_df, threshold_df = process_all_metrics(input_file, output_dir)
        print(f"\n✅ Successfully created metric-based predictions with optimal thresholds!")
        
    except Exception as e:
        print(f"Error processing metrics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
