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

# --- Input Files ---
# Note: This script can run on the full or sampled corpus
CORPUS_PATH = os.path.join(BASE_DIR, "data/sdewac-v3.txt_sample.txt") 
METRICS_CSV = os.path.join(BASE_DIR, "data/keywords_metrics.csv")
KEYWORDS_CSV = os.path.join(BASE_DIR, "data/keyword_groundtruth.csv")

# --- Output Files ---
# Primary annotated dataset
FULL_OUTPUT = os.path.join(BASE_DIR, "data/data_full.jsonl")
# Helper file with sentence counts per keyword
SENTENCE_COUNTS_CSV = os.path.join(BASE_DIR, "data/sentence_counts.csv")
# Split datasets
DEV_OUTPUT = os.path.join(BASE_DIR, "data/data_dev.jsonl")
TEST_OUTPUT = os.path.join(BASE_DIR, "data/data_test.jsonl") # Renamed from TEST_OUTPUT for clarity

# --- Parameters ---
JSONL_READ_BATCH_SIZE = 100_000
MAX_EXAMPLES_PER_KEYWORD = 10
DEV_RATIO = 0.2  # 20% of keywords for the development set

# =============================================================================
# Helper Functions for Dataset Creation
# =============================================================================

def load_metrics_map(path: str) -> dict:
    df = pd.read_csv(path)
    return {row["keyword"].lower(): row.to_dict() for _, row in df.iterrows()}

def load_variant_map(metrics_map: dict) -> (dict, int):
    lookup, max_len = {}, 1
    for key in metrics_map:
        for v in key.split("/"):
            tok_tuple = tuple(v.split())
            lookup[tok_tuple] = key
            max_len = max(max_len, len(tok_tuple))
    return lookup, max_len

def parse_sentences(lines: list) -> list:
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
    used, counts = set(), defaultdict(int)
    with open(corpus_path, encoding="utf-8") as src, \
         open(out_path, "w", encoding="utf-8") as out_f:
        batch = []
        for i, ln in enumerate(tqdm(src, desc="Creating full annotated dataset")):
            batch.append(ln)
            if i > 0 and i % JSONL_READ_BATCH_SIZE == 0:
                _process_jsonl_batch(batch, used, counts, lookup, max_len, metrics_map, freq_map, out_f, max_examples)
                batch = []
        if batch:
            _process_jsonl_batch(batch, used, counts, lookup, max_len, metrics_map, freq_map, out_f, max_examples)
    print(f"\nFull annotated dataset created at {out_path}")

def save_sentence_counts(jsonl_path: str, output_csv: str):
    df = pd.read_json(jsonl_path, lines=True)
    counts = df['keyword'].value_counts().sort_values(ascending=False)
    counts_df = counts.rename_axis('keyword').reset_index(name='sentence_count')
    counts_df.to_csv(output_csv, index=False)
    print(f"Sentence counts saved to {output_csv}")

def split_dataset_by_keywords(keywords_csv: str, annotated_jsonl: str,
                              dev_output: str, test_output: str,
                              ratio: float = 0.2):
    """
    Splits the main annotated JSONL into dev and test sets based on keywords.
    Ensures that all sentences for a given keyword go into the same split.
    """
    print("\n--- Splitting dataset into dev and test sets ---")
    
    # --- 1. Load and shuffle keywords from the ground truth CSV ---
    kw_df = pd.read_csv(keywords_csv)
    if "keyword" not in kw_df.columns:
        raise KeyError(f"No 'keyword' column found in {keywords_csv}. Available: {kw_df.columns}")

    # Randomly shuffle the keywords for an unbiased split
    kw_df = kw_df.sample(frac=1, random_state=42).reset_index(drop=True)

    all_keywords = [kw.strip().lower() for kw in kw_df["keyword"].dropna()]
    total_keywords = len(all_keywords)
    dev_size = math.ceil(total_keywords * ratio)

    dev_keywords = set(all_keywords[:dev_size])
    test_keywords = set(all_keywords[dev_size:])

    print(f"Total keywords from ground truth: {total_keywords}")
    print(f"Dev keywords: {len(dev_keywords)}, test keywords: {len(test_keywords)}")

    # --- 2. Load all annotated sentences ---
    print(f"Loading sentences from {annotated_jsonl}...")
    df = pd.read_json(annotated_jsonl, lines=True)
    if "keyword" not in df.columns:
        raise KeyError(f"No 'keyword' column found in {annotated_jsonl}. Available: {df.columns}")

    # Lowercase keywords to match consistently
    df["keyword"] = df["keyword"].str.lower()

    # --- 3. Split the DataFrame by keyword sets ---
    dev_df = df[df["keyword"].isin(dev_keywords)]
    test_df = df[df["keyword"].isin(test_keywords)]

    # --- 4. Save to separate JSONL files ---
    dev_df.to_json(dev_output, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_output, orient="records", lines=True, force_ascii=False)

    print(f"\nSaved {len(dev_df)} sentences to dev set: {dev_output}")
    print(f"Saved {len(test_df)} sentences to test set: {test_output}")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    # --- Step 1: Check for metrics file ---
    if not os.path.exists(METRICS_CSV):
        print(f"Error: Metrics file not found at {METRICS_CSV}")
        print("Please run '1_calculate_metrics.py' first.")
        return

    # --- Step 2: Create the main annotated dataset ---
    print("Loading pre-calculated metrics...")
    metrics_map = load_metrics_map(METRICS_CSV)
    lookup, max_len = load_variant_map(metrics_map)
    freq_map = {kw: data['occurrences'] for kw, data in metrics_map.items()}

    create_annotated_jsonl(CORPUS_PATH, metrics_map, lookup, max_len,
                           freq_map, FULL_OUTPUT, MAX_EXAMPLES_PER_KEYWORD)
    
    # --- Step 3: Save sentence counts ---
    if os.path.exists(FULL_OUTPUT):
        save_sentence_counts(FULL_OUTPUT, SENTENCE_COUNTS_CSV)
    else:
        print(f"Warning: {FULL_OUTPUT} was not created. Skipping sentence count.")
        
    # --- Step 4: Split the newly created dataset ---
    if os.path.exists(FULL_OUTPUT) and os.path.exists(KEYWORDS_CSV):
        split_dataset_by_keywords(KEYWORDS_CSV, FULL_OUTPUT, DEV_OUTPUT, test_OUTPUT, DEV_RATIO)
    else:
        print("\nSkipping dataset split because required input files are missing.")

if __name__ == "__main__":
    main()