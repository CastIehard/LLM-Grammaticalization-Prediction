import os
import csv
import json
import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/sdewac-v3.txt")
KEYWORDS_CSV = os.path.join(BASE_DIR, "data/keyword_groundtruth.csv")
METRICS_CSV = os.path.join(BASE_DIR, "data/keywords_metrics.csv")
ANNOTATED_JSONL = os.path.join(BASE_DIR, "data/sentences_annotated.jsonl")

# Subsampling for testing (set to None to process full corpus)
SENTENCES_COUNT = 1  # e.g., 10000 or None for full corpus

# JSONL creation parameters
BATCH_SIZE = 100_000
MAX_EXAMPLES_PER_KEYWORD = 10

# =============================================================================
# Data Preparation Functions
# =============================================================================

def sample_corpus(input_path: str, output_path: str, max_sentences: int) -> str:
    """
    Extract up to max_sentences sentences from the input corpus and save to output_path.
    Returns the path to the sampled file.
    """
    if max_sentences is None:
        return input_path
    sentence_count = 0
    with open(input_path, encoding="utf-8") as src, \
         open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            if "<sentence>" in line:
                sentence_count += 1
                if sentence_count >= max_sentences:
                    break
    print(f"Saved first {sentence_count} sentences to {output_path}")
    return output_path


def read_keywords(csv_path: str) -> (list, dict, int):
    """
    Read keyword variants and their ground-truth scores from CSV.
    Returns a list of (variant_tokens_list, key), a score_map, and max token length.
    """
    variants = []
    score_map = {}
    max_len = 1
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for kw, score in reader:
            kw = kw.strip()
            if not kw:
                continue
            splits = [v.lower() for v in kw.split('/')]
            tokens_list = [tuple(v.split()) for v in splits]
            max_len = max(max_len, max(len(t) for t in tokens_list))
            key = '/'.join(splits)
            variants.append((tokens_list, key))
            score_map[key] = float(score.strip())
    return variants, score_map, max_len


def build_variant_lookup(variants: list) -> dict:
    """
    Build mapping from token tuple to keyword key.
    """
    lookup = {}
    for tokens_list, key in variants:
        for tok_tuple in tokens_list:
            lookup[tok_tuple] = key
    return lookup


def read_tokens_tags(corpus_path: str) -> list:
    """
    Read token and tag pairs from corpus, skipping only markup lines.
    Returns a list of (token, tag). If tag missing, use empty string.
    """
    results = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading corpus"):
            line = line.strip()
            if not line or line.startswith('<'):
                continue
            parts = line.split('\t')
            tok = parts[0].lower()
            tag = parts[1] if len(parts) > 1 else ''
            results.append((tok, tag))
    return results

# =============================================================================
# Metric Computation Functions
# =============================================================================

def count_tokens(tokens_tags: list) -> dict:
    """
    Count occurrences of each token in the corpus.
    Returns a token -> count map.
    """
    counts = defaultdict(int)
    for tok, _ in tqdm(tokens_tags, desc="Counting tokens"):
        counts[tok] += 1
    return counts


def sliding_window_match(tokens_tags: list, lookup: dict, max_len: int) -> (dict, dict, dict, dict, dict):
    """
    Perform sliding window over tokens to count keyword occurrences and contexts.
    Returns freq, contexts, pre_tags, post_tags, bigram_counts.
    """
    freq = defaultdict(int)
    contexts = defaultdict(list)
    pre_tags = defaultdict(set)
    post_tags = defaultdict(set)
    bigram_counts = defaultdict(int)
    total = len(tokens_tags)
    for i in tqdm(range(total), desc="Sliding window"):
        for n in range(1, max_len + 1):
            if i + n > total:
                break
            window = tuple(tok for tok, _ in tokens_tags[i:i+n])
            if window in lookup:
                key = lookup[window]
                freq[key] += 1
                if i > 0:
                    prev_tok, prev_tag = tokens_tags[i-1]
                    contexts[key].append(prev_tok)
                    pre_tags[key].add(prev_tag)
                    bigram_counts[(key, prev_tok)] += 1
                if i + n < total:
                    next_tok, next_tag = tokens_tags[i+n]
                    contexts[key].append(next_tok)
                    post_tags[key].add(next_tag)
                    bigram_counts[(key, next_tok)] += 1
                break
    return freq, contexts, pre_tags, post_tags, bigram_counts


def compute_metrics(variants: list, freq: dict, contexts: dict,
                    pre_tags: dict, post_tags: dict,
                    bigram_counts: dict, token_counts: dict,
                    score_map: dict, total_tokens: int) -> pd.DataFrame:
    """
    Calculate normalized occurrences, context entropy, collocation strength, etc.
    Returns a sorted DataFrame of metrics per keyword.
    """
    rows = []
    if freq:
        max_occ = max(freq.values())
        min_occ = min(freq.values())
        occ_range = max_occ - min_occ or 1
    else:
        max_occ = min_occ = occ_range = 1
    for token_lists, key in variants:
        k_freq = freq.get(key, 0)
        norm = (k_freq - min_occ) / occ_range if k_freq not in {max_occ, min_occ} else (1.0 if k_freq == max_occ else 0.0)
        norm = round(max(norm, 0), 3)
        avg_len = sum(len(t) for t in token_lists[0]) / len(token_lists[0])
        avg_len = round(avg_len, 1)
        entropy = len(set(contexts.get(key, [])))
        sca = len(pre_tags.get(key, [])) + len(post_tags.get(key, []))
        weighted_pmi = 0.0
        for (w, c), bc in bigram_counts.items():
            if w != key:
                continue
            p_wc = bc / max(total_tokens - 1, 1)
            p_w = k_freq / total_tokens if k_freq else 1 / total_tokens
            p_c = token_counts.get(c, 1) / total_tokens
            weighted_pmi += (math.log2(p_wc / (p_w * p_c) + 1e-12) * bc)
        col_str = weighted_pmi / k_freq if k_freq else 0.0
        col_str = round(col_str, 3)
        rows.append({
            "keyword": key,
            "occurrences": k_freq,
            "normalized_occurrences": norm,
            "avg_length": avg_len,
            "context_entropy": entropy,
            "collocation_strength": col_str,
            "synthetic_context_adversity": sca,
            "gramm_score": score_map.get(key, None)
        })
    df = pd.DataFrame(rows).sort_values("occurrences", ascending=False)
    return df

# =============================================================================
# Output Functions
# =============================================================================

def save_metrics(df: pd.DataFrame, output_path: str):
    """
    Save metrics DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def plot_correlation(df: pd.DataFrame, output_path: str = None):
    """
    Plot and optionally save a correlation heatmap of metric columns.
    """
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation plot")
        return
    
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", square=True, mask=mask,
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={"shrink": .8})
    plt.title("Keyword Metrics Correlation Matrix")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to {output_path}")

# =============================================================================
# JSONL Creation Functions
# =============================================================================

def load_metrics_map(path: str) -> dict:
    """
    Load keyword metrics CSV into a mapping from keyword to metrics dict.
    """
    df = pd.read_csv(path)
    return {row["keyword"].lower(): row.to_dict() for _, row in df.iterrows()}


def load_variant_map(metrics_map: dict) -> (dict, int):
    """
    Build variant lookup map and determine max keyword token length.
    """
    lookup, max_len = {}, 1
    for key in metrics_map:
        for v in key.split("/"):
            tok_tuple = tuple(v.split())
            lookup[tok_tuple] = key
            max_len = max(max_len, len(tok_tuple))
    return lookup, max_len


def parse_sentences(lines: list) -> list:
    """
    Yield lists of tokens representing each sentence in a batch of lines.
    Do not skip any tokens (except markup for boundaries).
    """
    sentence = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("<"):
            if sentence:
                yield sentence
            sentence = []
            continue
        parts = ln.split("\t")
        tok = parts[0]
        sentence.append(tok)
    if sentence:
        yield sentence


def create_annotated_jsonl(corpus_path: str, metrics_map: dict,
                           lookup: dict, max_len: int,
                           freq_map: dict,
                           out_path: str, max_examples: int):
    """
    Create a JSONL file with annotated sentences and associated metrics.
    If multiple keywords in one sentence, pick the least frequent based on freq_map.
    """
    used, counts = set(), defaultdict(int)
    with open(corpus_path, encoding="utf-8") as src, \
         open(out_path, "w", encoding="utf-8") as out_f:
        batch = []
        for i, ln in enumerate(src, 1):
            batch.append(ln)
            if i % BATCH_SIZE == 0:
                _process_batch(batch, used, counts, lookup, max_len, metrics_map, freq_map, out_f, max_examples)
                batch = []
        if batch:
            _process_batch(batch, used, counts, lookup, max_len, metrics_map, freq_map, out_f, max_examples)
    print(f"Annotated JSONL created at {out_path}")


def _process_batch(lines, used, counts, lookup, max_len,
                   metrics_map, freq_map,
                   out_f, max_examples):
    """
    Internal helper to process a batch of lines for JSONL creation.
    """
    for sent in parse_sentences(lines):
        tokens_lc = [t.lower() for t in sent]
        sent_str = " ".join(tokens_lc)
        if sent_str in used:
            continue
        candidates = []
        for i in range(len(tokens_lc)):
            for n in range(1, max_len + 1):
                if i + n > len(tokens_lc):
                    break
                key = lookup.get(tuple(tokens_lc[i:i+n]))
                if key and counts[key] < max_examples:
                    candidates.append((key, i, n))
        if not candidates:
            continue
        # select candidate with minimal overall freq; if tie, pick random
        min_freq = min(freq_map[k] for k,_,_ in candidates)
        best = [c for c in candidates if freq_map[c[0]] == min_freq]
        key, i, n = random.choice(best)
        counts[key] += 1
        used.add(sent_str)
        sent_marked = sent.copy()
        sent_marked[i] = "[" + sent_marked[i]
        sent_marked[i+n-1] += "]"
        entry = {"sentence_raw": " ".join(sent_marked), **metrics_map[key]}
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# =============================================================================
# Main Execution
# =============================================================================

# =============================================================================
# Sentence Count Output Function
# =============================================================================
def save_sentence_counts(jsonl_path: str, output_csv: str):
    """
    Load the annotated JSONL and save the number of sentences per keyword to a CSV.
    """
    df = pd.read_json(jsonl_path, lines=True)
    counts = df['keyword'].value_counts().sort_values(ascending=False)
    counts_df = counts.rename_axis('keyword').reset_index(name='sentence_count')
    counts_df.to_csv(output_csv, index=False)
    print(f"Sentence counts saved to {output_csv}")

def main():
    sample_path = sample_corpus(DATA_PATH, f"{DATA_PATH}_sample.txt", SENTENCES_COUNT)
    variants, score_map, max_kw_len = read_keywords(KEYWORDS_CSV)
    lookup = build_variant_lookup(variants)
    tokens_tags = read_tokens_tags(sample_path)
    token_counts = count_tokens(tokens_tags)
    freq, contexts, pre_tags, post_tags, bigram_counts = \
        sliding_window_match(tokens_tags, lookup, max_kw_len)

    df_metrics = compute_metrics(variants, freq, contexts, pre_tags, post_tags,
                                 bigram_counts, token_counts, score_map, len(tokens_tags))
    save_metrics(df_metrics, METRICS_CSV)
    plot_correlation(df_metrics, output_path=os.path.join(BASE_DIR, "data/metrics_correlation.png"))

    metrics_map = load_metrics_map(METRICS_CSV)
    lookup2, max_len2 = load_variant_map(metrics_map)
    create_annotated_jsonl(sample_path, metrics_map, lookup2, max_len2,
                           freq, ANNOTATED_JSONL, MAX_EXAMPLES_PER_KEYWORD)
    save_sentence_counts(ANNOTATED_JSONL, os.path.join(BASE_DIR, "data/sentence_counts.csv"))

if __name__ == "__main__":
    main()
