#!/usr/bin/env python3
"""
Script to extract distinct keywords from dev and test datasets and create
separate keyword metrics files for each dataset.
"""

import json
import pandas as pd
import os

def extract_keywords_from_jsonl(file_path):
    """
    Extract distinct keywords from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        set: Set of distinct keywords
    """
    keywords = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            keyword = data.get('keyword')
            if keyword:
                keywords.add(keyword)
    
    return keywords

def create_keyword_metrics_subset(keywords_metrics_path, keywords_set, output_path):
    """
    Create a subset of keyword metrics for the given set of keywords.
    
    Args:
        keywords_metrics_path (str): Path to the full keywords metrics CSV
        keywords_set (set): Set of keywords to extract
        output_path (str): Path to save the subset CSV
    """
    # Read the full keywords metrics
    df = pd.read_csv(keywords_metrics_path)
    
    # Filter for keywords in the set
    filtered_df = df[df['keyword'].isin(keywords_set)].copy()
    
    # Sort by occurrences (descending) for consistency
    filtered_df = filtered_df.sort_values('occurrences', ascending=False)
    
    # Save to output file
    filtered_df.to_csv(output_path, index=False)
    
    print(f"Created {output_path} with {len(filtered_df)} keywords")
    return len(filtered_df)

def main():
    # Define file paths
    base_dir = "/Users/luca/Desktop/UTN/LLMs-for-Classification-of-Grammaticalization-Degrees"
    data_dir = os.path.join(base_dir, "data")
    
    dev_file = os.path.join(data_dir, "data_dev.jsonl")
    test_file = os.path.join(data_dir, "data_test.jsonl")
    keywords_metrics_file = os.path.join(data_dir, "keywords_metrics.csv")
    
    dev_metrics_output = os.path.join(data_dir, "keywords_metrics_dev.csv")
    test_metrics_output = os.path.join(data_dir, "keywords_metrics_test.csv")
    
    # Check if input files exist
    for file_path in [dev_file, test_file, keywords_metrics_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    print("Extracting keywords from datasets...")
    
    # Extract keywords from dev and test files
    dev_keywords = extract_keywords_from_jsonl(dev_file)
    test_keywords = extract_keywords_from_jsonl(test_file)
    
    print(f"Dev dataset: {len(dev_keywords)} distinct keywords")
    print(f"Test dataset: {len(test_keywords)} distinct keywords")
    
    # Find common keywords
    common_keywords = dev_keywords.intersection(test_keywords)
    print(f"Common keywords: {len(common_keywords)}")
    
    # Create subset metrics files
    print("\nCreating keyword metrics subsets...")
    
    dev_count = create_keyword_metrics_subset(
        keywords_metrics_file, dev_keywords, dev_metrics_output
    )
    
    test_count = create_keyword_metrics_subset(
        keywords_metrics_file, test_keywords, test_metrics_output
    )
    
    print(f"\nSummary:")
    print(f"- Dev keywords metrics file: {dev_metrics_output} ({dev_count} keywords)")
    print(f"- Test keywords metrics file: {test_metrics_output} ({test_count} keywords)")
    
    # Show sample of keywords from each set
    print(f"\nSample dev keywords: {list(dev_keywords)[:10]}")
    print(f"Sample test keywords: {list(test_keywords)[:10]}")
    print(f"Sample common keywords: {list(common_keywords)[:10]}")

if __name__ == "__main__":
    main()
