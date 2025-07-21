import os
import math
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_CSV = os.path.join(BASE_DIR, "data/keyword_groundtruth.csv")
ANNOTATED_JSONL = os.path.join(BASE_DIR, "data/sentences_annotated.jsonl")

DEV_OUTPUT = os.path.join(BASE_DIR, "data/dataset_dev.jsonl")
TEST_OUTPUT = os.path.join(BASE_DIR, "data/data_test.jsonl")

DEV_RATIO = 0.2  # 20% of keywords for dev

# =============================================================================
# Split Function
# =============================================================================
def split_dataset_by_keywords(keywords_csv: str, annotated_jsonl: str,
                              dev_output: str, train_output: str,
                              ratio: float = 0.2):
    # --- 1. Load keywords from CSV ---
    kw_df = pd.read_csv(keywords_csv)
    if "keyword" not in kw_df.columns:
        raise KeyError(f"No 'keyword' column found in {keywords_csv}. Available: {kw_df.columns}")

    all_keywords = [kw.strip().lower() for kw in kw_df["keyword"].dropna()]
    total_keywords = len(all_keywords)
    dev_size = math.ceil(total_keywords * ratio)

    dev_keywords = set(all_keywords[:dev_size])
    train_keywords = set(all_keywords[dev_size:])

    print(f"Total keywords: {total_keywords}")
    print(f"Dev/Test keywords: {len(dev_keywords)}, Train keywords: {len(train_keywords)}")

    # --- 2. Load annotated sentences ---
    df = pd.read_json(annotated_jsonl, lines=True)
    if "keyword" not in df.columns:
        raise KeyError(f"No 'keyword' column found in {annotated_jsonl}. Available: {df.columns}")

    # Lowercase keywords to match consistently
    df["keyword"] = df["keyword"].str.lower()

    # --- 3. Split by keyword sets ---
    dev_df = df[df["keyword"].isin(dev_keywords)]
    train_df = df[df["keyword"].isin(train_keywords)]

    # --- 4. Save to JSONL ---
    dev_df.to_json(dev_output, orient="records", lines=True, force_ascii=False)
    train_df.to_json(train_output, orient="records", lines=True, force_ascii=False)

    print(f"Saved {len(dev_df)} sentences to {dev_output}")
    print(f"Saved {len(train_df)} sentences to {train_output}")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    split_dataset_by_keywords(KEYWORDS_CSV, ANNOTATED_JSONL, DEV_OUTPUT, TEST_OUTPUT, DEV_RATIO)