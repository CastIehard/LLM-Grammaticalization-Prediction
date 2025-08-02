#!/usr/bin/env python3
"""
Script to generate random baseline predictions for dev2 dataset.
Extracts keywords and gramm_score from dev_data_2_testing_metrics.csv,
adds random predictions (1-4), and saves to evaluation/input_csv (only dev2 nothing else)/
"""

import pandas as pd
import numpy as np
import os


def generate_random_baseline_dev2(random_seed=42):
    """
    Generate random baseline for dev2 dataset.
    
    Args:
        random_seed (int): Seed for reproducible random numbers
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Define file paths
    base_dir = "/Users/luca/Desktop/UTN/LLMs-for-Classification-of-Grammaticalization-Degrees"
    input_file = os.path.join(base_dir, "data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv")
    output_dir = os.path.join(base_dir, "evaluation/input_csv (only dev2 nothing else)")
    output_file = os.path.join(output_dir, "random_baseline_dev2.csv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the dev2 metrics file
    print(f"Reading dev2 metrics from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Found {len(df)} keywords in dev2 dataset")
    
    # Extract only keyword and gramm_score columns
    baseline_df = df[['keyword', 'gramm_score']].copy()
    
    # Generate random predictions (1-4) for each keyword
    predictions = np.random.randint(1, 5, size=len(baseline_df))
    baseline_df['predictions'] = predictions
    
    # Save the random baseline
    baseline_df.to_csv(output_file, index=False)
    print(f"Random baseline saved to: {output_file}")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total keywords: {len(baseline_df)}")
    
    print(f"\nTrue label distribution:")
    true_counts = baseline_df['gramm_score'].value_counts().sort_index()
    for score, count in true_counts.items():
        percentage = (count / len(baseline_df)) * 100
        print(f"  Level {score}: {count} keywords ({percentage:.1f}%)")
    
    print(f"\nRandom prediction distribution:")
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    for score, count in pred_counts.items():
        percentage = (count / len(baseline_df)) * 100
        print(f"  Level {score}: {count} keywords ({percentage:.1f}%)")
    
    # Show first few rows as example
    print(f"\nFirst 10 rows:")
    print(baseline_df.head(10).to_string(index=False))
    
    return baseline_df


def main():
    """Main function to generate random baseline for dev2 dataset."""
    try:
        df = generate_random_baseline_dev2()
        print(f"\nSuccessfully generated random baseline for dev2 dataset!")
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
