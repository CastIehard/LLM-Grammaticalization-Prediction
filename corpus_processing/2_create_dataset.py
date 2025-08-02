import os
import csv
import json
import math
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#go one level up to the main project directory because this script is in corpus_processing
BASE_DIR = os.path.dirname(BASE_DIR)



# --- Input Files ---
CORPUS_PATH = os.path.join(BASE_DIR, "data/sdewac-v3.txt") 
FULL_DATA_DIR = os.path.join(BASE_DIR, "data/full data (only for storing, do not use)")
METRICS_CSV = os.path.join(FULL_DATA_DIR, "keywords_metrics_full.csv")
KEYWORDS_CSV = os.path.join(BASE_DIR, "data/general data (some additional stuff)/keyword_groundtruth.csv")

# --- Output Files ---
FULL_OUTPUT = os.path.join(FULL_DATA_DIR, "data_full.jsonl")
# Helper file with sentence counts per keyword
SENTENCE_COUNTS_CSV = os.path.join(BASE_DIR, "data/general data (some additional stuff)/sentence_counts.csv")

# Few-shot prompting data (2 keywords per level)
DEV_DATA_1_OUTPUT = os.path.join(BASE_DIR, "data/dev data 1 (for prompting)/dev_data_1_prompting.jsonl")
# Development data for testing
DEV_DATA_2_OUTPUT = os.path.join(BASE_DIR, "data/dev data 2 (for testing)/dev_data_2_testing.jsonl")
# Final test data
TEST_DATA_OUTPUT = os.path.join(BASE_DIR, "data/test data (only use at the end)/test_data.jsonl")

# --- Parameters ---
JSONL_READ_BATCH_SIZE = 100_000
MAX_EXAMPLES_PER_KEYWORD = 10

# =============================================================================
# Helper Functions for Dataset Creation
# =============================================================================

def load_metrics_map(path: str) -> dict:
    df = pd.read_csv(path)
    return {row["keyword"].lower(): row.to_dict() for _, row in df.iterrows()}

def load_variant_map(metrics_map: dict) -> tuple[dict, int]:
    lookup, max_len = {}, 1
    for key in metrics_map:
        for v in key.split("/"):
            tok_tuple = tuple(v.split())
            lookup[tok_tuple] = key
            max_len = max(max_len, len(tok_tuple))
    return lookup, max_len

def parse_sentences(lines: list):
    sentence = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("<"):
            if sentence: yield sentence
            sentence = []
            continue
        parts = ln.split("\t")
        sentence.append(parts[0])
    if sentence: yield sentence

def _process_jsonl_batch(lines, used, counts, lookup, max_len,
                         metrics_map, freq_map, out_f, max_examples):
    for sent in parse_sentences(lines):
        tokens_lc = [t.lower() for t in sent]
        sent_str = " ".join(tokens_lc)
        if sent_str in used: continue
        
        candidates = []
        for i in range(len(tokens_lc)):
            for n in range(1, max_len + 1):
                if i + n > len(tokens_lc): break
                key = lookup.get(tuple(tokens_lc[i:i+n]))
                if key and counts[key] < max_examples:
                    candidates.append((key, i, n))
        
        if not candidates: continue

        min_freq = min(freq_map.get(k, 0) for k,_,_ in candidates)
        best = [c for c in candidates if freq_map.get(c[0], 0) == min_freq]
        key, i, n = random.choice(best)
        
        counts[key] += 1
        used.add(sent_str)
        
        sent_marked = sent.copy()
        sent_marked[i] = "[" + sent_marked[i]
        sent_marked[i+n-1] += "]"
        
        entry = {"sentence_raw": " ".join(sent_marked), **metrics_map[key]}
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def create_annotated_jsonl(corpus_path: str, metrics_map: dict,
                           lookup: dict, max_len: int, freq_map: dict,
                           out_path: str, max_examples: int):
    
    # Get the set of all keywords we are looking for
    all_target_keywords = set(metrics_map.keys())
    num_total_keywords = len(all_target_keywords)
    
    used, counts = set(), defaultdict(int)

    with open(corpus_path, encoding="utf-8") as src, \
         open(out_path, "w", encoding="utf-8") as out_f:
        batch = []
        # Use tqdm's dynamic total if you have it, otherwise just iterate
        with tqdm(desc="Creating full annotated dataset", unit=" lines") as pbar:
            for i, ln in enumerate(src):
                batch.append(ln)
                pbar.update(1) # Manually update progress bar

                if i > 0 and i % JSONL_READ_BATCH_SIZE == 0:
                    _process_jsonl_batch(batch, used, counts, lookup, max_len, metrics_map, freq_map, out_f, max_examples)
                    batch = []

                    # Check if we have found enough examples for ALL keywords
                    keywords_completed = sum(1 for kw in all_target_keywords if counts[kw] >= max_examples)
                    if keywords_completed == num_total_keywords:
                        print(f"\nFound {max_examples} examples for all {num_total_keywords} keywords. Stopping early.")
                        break # Exit the main loop
        
        # Process the final batch if the loop didn't break early
        if batch:
            _process_jsonl_batch(batch, used, counts, lookup, max_len, metrics_map, freq_map, out_f, max_examples)

    print(f"\nFull annotated dataset created at {out_path}")
def save_sentence_counts(jsonl_path: str, output_csv: str):
    df = pd.read_json(jsonl_path, lines=True)
    counts = df['keyword'].value_counts().sort_values(ascending=False)
    counts_df = counts.rename_axis('keyword').reset_index(name='sentence_count')
    counts_df.to_csv(output_csv, index=False)
    print(f"Sentence counts saved to {output_csv}")

def extract_keyword_metrics_for_dataset(keywords_set: set, full_metrics_csv: str, output_csv: str):
    """
    Extract keyword metrics for a specific set of keywords and save to CSV.
    
    Args:
        keywords_set (set): Set of keywords to extract
        full_metrics_csv (str): Path to the full keyword metrics CSV
        output_csv (str): Path to save the subset CSV
    """
    if not keywords_set:
        print(f"No keywords to extract for {output_csv}")
        return 0
        
    # Read the full keywords metrics
    df = pd.read_csv(full_metrics_csv)
    
    # Filter for keywords in the set
    filtered_df = df[df['keyword'].isin(keywords_set)].copy()
    
    # Sort by occurrences (descending) for consistency
    filtered_df = filtered_df.sort_values('occurrences', ascending=False)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save to output file
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"Created keyword metrics CSV: {output_csv} with {len(filtered_df)} keywords")
    return len(filtered_df)

def split_dataset_into_three_sets(keywords_csv: str, annotated_jsonl: str,
                                  dev_data_1_output: str, dev_data_2_output: str, 
                                  test_data_output: str):
    """
    Splits the main annotated JSONL into three sets:
    1. Dev Data 1 (for prompting): 2 keywords per grammaticalization level (1-4)
    2. Dev Data 2 (for testing): 50% of remaining keywords
    3. Test Data (only use at the end): 50% of remaining keywords
    """
    print("\n--- Splitting dataset into three sets ---")
    
    # --- 1. Load keywords with their grammaticalization scores ---
    kw_df = pd.read_csv(keywords_csv)
    if "keyword" not in kw_df.columns or "gramm_score" not in kw_df.columns:
        raise KeyError(f"Required columns 'keyword' and 'gramm_score' not found in {keywords_csv}")

    # Clean and prepare keywords
    kw_df["keyword"] = kw_df["keyword"].str.strip().str.lower()
    kw_df = kw_df.dropna(subset=["keyword", "gramm_score"])
    
    # Group keywords by grammaticalization level
    grouped = kw_df.groupby("gramm_score")
    
    # --- 2. Select 2 keywords per level for few-shot prompting ---
    few_shot_keywords = set()
    random.seed(42)  # For reproducible results
    
    for level in [1, 2, 3, 4]:
        if level in grouped.groups:
            level_keywords = grouped.get_group(level)["keyword"].tolist()
            if len(level_keywords) >= 2:
                selected = random.sample(level_keywords, 2)
                few_shot_keywords.update(selected)
                print(f"Level {level}: Selected {selected}")
            else:
                print(f"Warning: Only {len(level_keywords)} keywords available for level {level}")
                few_shot_keywords.update(level_keywords)
    
    print(f"Total few-shot keywords: {len(few_shot_keywords)}")
    
    # --- 3. Get remaining keywords and split 50/50 ---
    all_keywords = set(kw_df["keyword"].tolist())
    remaining_keywords = all_keywords - few_shot_keywords
    remaining_list = list(remaining_keywords)
    
    # Shuffle remaining keywords
    random.shuffle(remaining_list)
    
    # Split 50/50
    mid_point = len(remaining_list) // 2
    dev_data_2_keywords = set(remaining_list[:mid_point])
    test_data_keywords = set(remaining_list[mid_point:])
    
    print(f"Remaining keywords: {len(remaining_keywords)}")
    print(f"Dev Data 2 keywords: {len(dev_data_2_keywords)}")
    print(f"Test Data keywords: {len(test_data_keywords)}")
    
    # --- 4. Load all annotated sentences ---
    print(f"Loading sentences from {annotated_jsonl}...")
    df = pd.read_json(annotated_jsonl, lines=True)
    if "keyword" not in df.columns:
        raise KeyError(f"No 'keyword' column found in {annotated_jsonl}")
    
    # Lowercase keywords to match consistently
    df["keyword"] = df["keyword"].str.lower()
    
    # --- 5. Split the DataFrame by keyword sets ---
    dev_data_1_df = df[df["keyword"].isin(few_shot_keywords)]
    dev_data_2_df = df[df["keyword"].isin(dev_data_2_keywords)]
    test_data_df = df[df["keyword"].isin(test_data_keywords)]
    
    # --- 6. Save to separate JSONL files ---
    # Create directories if they don't exist
    for output_path in [dev_data_1_output, dev_data_2_output, test_data_output]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    #sort dev_data_1_df by keyword and gramm_score to ensure consistent order for few-shot prompting
    dev_data_1_df = dev_data_1_df.sort_values(by=["keyword", "gramm_score"]).reset_index(drop=True)
    dev_data_1_df.to_json(dev_data_1_output, orient="records", lines=True, force_ascii=False)
    dev_data_2_df.to_json(dev_data_2_output, orient="records", lines=True, force_ascii=False)
    test_data_df.to_json(test_data_output, orient="records", lines=True, force_ascii=False)
    
    print(f"\nSaved {len(dev_data_1_df)} sentences to Dev Data 1 (prompting): {dev_data_1_output}")
    print(f"Saved {len(dev_data_2_df)} sentences to Dev Data 2 (testing): {dev_data_2_output}")
    print(f"Saved {len(test_data_df)} sentences to Test Data (final): {test_data_output}")
    
    # --- 7. Extract and save keyword metrics for each dataset ---
    print(f"\nExtracting keyword metrics for each dataset...")
    
    # Generate CSV file paths by replacing .jsonl with .csv
    dev_data_1_csv = dev_data_1_output.replace('.jsonl', '_metrics.csv')
    dev_data_2_csv = dev_data_2_output.replace('.jsonl', '_metrics.csv')
    test_data_csv = test_data_output.replace('.jsonl', '_metrics.csv')

    # Extract keyword metrics for each dataset
    extract_keyword_metrics_for_dataset(few_shot_keywords, METRICS_CSV, dev_data_1_csv)
    extract_keyword_metrics_for_dataset(dev_data_2_keywords, METRICS_CSV, dev_data_2_csv)
    extract_keyword_metrics_for_dataset(test_data_keywords, METRICS_CSV, test_data_csv)
    
    # --- 8. Print summary by level for Dev Data 1 ---
    print(f"\nFew-shot prompting data summary:")
    for level in [1, 2, 3, 4]:
        level_data = dev_data_1_df[dev_data_1_df["gramm_score"] == level]
        if len(level_data) > 0:
            keywords_in_level = level_data["keyword"].unique()
            print(f"Level {level}: {len(level_data)} sentences from {len(keywords_in_level)} keywords: {list(keywords_in_level)}")
    
    # --- 9. Print file summary ---
    print(f"\nDataset files created:")
    print(f"- Dev Data 1 JSONL: {dev_data_1_output}")
    print(f"- Dev Data 1 CSV: {dev_data_1_csv}")
    print(f"- Dev Data 2 JSONL: {dev_data_2_output}")
    print(f"- Dev Data 2 CSV: {dev_data_2_csv}")
    print(f"- Test Data JSONL: {test_data_output}")
    print(f"- Test Data CSV: {test_data_csv}")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    # Create output directories if they don't exist
    os.makedirs(FULL_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SENTENCE_COUNTS_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(DEV_DATA_1_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(DEV_DATA_2_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_DATA_OUTPUT), exist_ok=True)
    
    # --- Step 1: Check if full dataset already exists ---
    if os.path.exists(FULL_OUTPUT):
        print(f"Full dataset already exists at {FULL_OUTPUT}")
        print("Skipping dataset creation and proceeding to split...")
    else:
        print(f"Full dataset not found at {FULL_OUTPUT}")
        
        # --- Check for metrics file ---
        if not os.path.exists(METRICS_CSV):
            print(f"Error: Metrics file not found at {METRICS_CSV}")
            print("Please ensure the full metrics file exists.")
            return

        # --- Create the main annotated dataset ---
        print("Loading pre-calculated metrics...")
        metrics_map = load_metrics_map(METRICS_CSV)
        lookup, max_len = load_variant_map(metrics_map)
        freq_map = {kw: data['occurrences'] for kw, data in metrics_map.items()}
        
        create_annotated_jsonl(CORPUS_PATH, metrics_map, lookup, max_len,
                               freq_map, FULL_OUTPUT, MAX_EXAMPLES_PER_KEYWORD)
    
    # --- Step 2: Save sentence counts ---
    if os.path.exists(FULL_OUTPUT):
        save_sentence_counts(FULL_OUTPUT, SENTENCE_COUNTS_CSV)
    else:
        print(f"Warning: {FULL_OUTPUT} was not created. Skipping sentence count.")
        
    # --- Step 3: Split the dataset into three sets ---
    if os.path.exists(FULL_OUTPUT) and os.path.exists(KEYWORDS_CSV):
        split_dataset_into_three_sets(KEYWORDS_CSV, FULL_OUTPUT, 
                                     DEV_DATA_1_OUTPUT, DEV_DATA_2_OUTPUT, 
                                     TEST_DATA_OUTPUT)
    else:
        missing_files = []
        if not os.path.exists(FULL_OUTPUT):
            missing_files.append(FULL_OUTPUT)
        if not os.path.exists(KEYWORDS_CSV):
            missing_files.append(KEYWORDS_CSV)
        print(f"\nSkipping dataset split because required files are missing: {missing_files}")

if __name__ == "__main__":
    main()