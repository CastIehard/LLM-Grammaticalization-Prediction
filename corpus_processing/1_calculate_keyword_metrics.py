import os
import csv
import json
import math
import gc
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#go one level up to the main project directory because this script is in corpus_processing
BASE_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "sdewac-v3.txt") #needs to be downloaded and extracted from the zip file
KEYWORDS_CSV = os.path.join(BASE_DIR, "data/general data (some additional stuff)/keyword_groundtruth.csv")
METRICS_CSV = os.path.join(BASE_DIR, "data/full data (only for storing, do not use)/keywords_metrics_full.csv")

# Subsampling for testing (set to None to process full corpus)
SENTENCES_COUNT = None

# Batch size for processing the corpus to calculate metrics
PROCESSING_BATCH_SIZE = 100_000

# =============================================================================
# Helper Functions
# =============================================================================

def sample_corpus(input_path: str, output_path: str, max_sentences: int) -> str:
    if max_sentences is None:
        print("Processing the full corpus file.")
        return input_path
    if os.path.exists(output_path):
         print(f"Using existing sampled corpus: {output_path}")
         return output_path
    sentence_count = 0
    with open(input_path, encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in tqdm(src, desc=f"Creating sample of {max_sentences} sentences"):
            dst.write(line)
            if "<sentence>" in line:
                sentence_count += 1
                if sentence_count >= max_sentences:
                    break
    print(f"Saved first {sentence_count} sentences to {output_path}")
    return output_path

def read_keywords(csv_path: str) -> (list, dict, int):
    variants, score_map, max_len = [], {}, 1
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for kw, score in reader:
            kw = kw.strip()
            if not kw: continue
            splits = [v.lower() for v in kw.split('/')]
            tokens_list = [tuple(v.split()) for v in splits]
            max_len = max(max_len, max(len(t) for t in tokens_list))
            key = '/'.join(splits)
            variants.append((tokens_list, key))
            score_map[key] = float(score.strip())
    return variants, score_map, max_len

def build_variant_lookup(variants: list) -> dict:
    lookup = {}
    for tokens_list, key in variants:
        for tok_tuple in tokens_list:
            lookup[tok_tuple] = key
    return lookup

def count_total_sentences_and_tokens(corpus_path: str) -> (int, int):
    """
    Efficiently counts the total number of sentences and tokens in the corpus
    by iterating through the file just once.
    """
    print("Pre-calculating total sentences and tokens...")
    sentence_count = 0
    token_count = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if the line is a markup tag
            if line.startswith('<'):
                if "<sentence>" in line:
                    sentence_count += 1
            # Otherwise, it's a token line
            else:
                token_count += 1
    print(f"Found {sentence_count} sentences and {token_count} tokens.")
    #save to corpus statistics file
    path = os.path.join(os.path.dirname(corpus_path), "corpus_statistics.json")
    with open(path, 'w', encoding='utf-8') as stats_file:
        json.dump({
            "total_sentences": sentence_count,
            "total_tokens": token_count
        }, stats_file, indent=4)
    print(f"Corpus statistics saved to {path}")
    return sentence_count, token_count

def read_corpus_batches(corpus_path: str, batch_size_sentences: int):
    with open(corpus_path, encoding="utf-8") as f:
        batch_tokens_tags = []
        sentence_count_in_batch = 0
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('<'):
                if "<sentence>" in line:
                    sentence_count_in_batch += 1
                    if sentence_count_in_batch >= batch_size_sentences:
                        yield batch_tokens_tags
                        batch_tokens_tags = []
                        sentence_count_in_batch = 0
                        gc.collect()
                continue
            parts = line.split('\t')
            tok, tag = parts[0].lower(), parts[1] if len(parts) > 1 else ''
            batch_tokens_tags.append((tok, tag))
        if batch_tokens_tags:
            yield batch_tokens_tags
            gc.collect()

def sliding_window_match(tokens_tags: list, lookup: dict, max_len: int) -> (dict, dict, dict, dict, dict):
    freq, contexts, pre_tags, post_tags, bigram_counts = defaultdict(int), defaultdict(list), defaultdict(set), defaultdict(set), defaultdict(int)
    total = len(tokens_tags)
    for i in range(total):
        for n in range(1, max_len + 1):
            if i + n > total: break
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
    return freq, contexts, pre_tags, post_tags, bigram_counts

# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_metrics(variants: list, freq: dict, contexts: dict,
                    pre_tags: dict, post_tags: dict,
                    bigram_counts: dict, token_counts: dict,
                    score_map: dict, total_tokens: int) -> pd.DataFrame:
    """
    Calculate all metrics, including Shannon entropy of the context.
    """
    rows = []
    if not freq:
        print("Warning: No keyword occurrences found.")
        return pd.DataFrame()

    max_occ = max(freq.values()) if freq else 0
    min_occ = min(freq.values()) if freq else 0
    occ_range = max_occ - min_occ or 1
    
    for token_lists, key in tqdm(variants, desc="Calculating final metrics"):
        k_freq = freq.get(key, 0)
        
        # Average Character Count
        all_tokens_in_keyword = [tok for t_list in token_lists for tok in t_list[0].split()]
        avg_char_count = sum(len(tok) for tok in all_tokens_in_keyword) / len(all_tokens_in_keyword) if all_tokens_in_keyword else 0
        
        context_list = contexts.get(key, [])
        distinct_neighbors = len(set(context_list))
        
        word_entropy = 0.0
        if context_list:
            total_neighbors = len(context_list)
            neighbor_counts = Counter(context_list)
            for count in neighbor_counts.values():
                p = count / total_neighbors
                word_entropy -= p * math.log2(p)

        weighted_pmi = 0.0
        if k_freq > 0 and total_tokens > 0:
            for (w, c), bc in bigram_counts.items():
                if w != key: continue
                p_wc = bc / total_tokens
                p_w = k_freq / total_tokens
                p_c = token_counts.get(c, 1) / total_tokens
                pmi = math.log2((p_wc / (p_w * p_c)) + 1e-12)
                weighted_pmi += pmi * bc
        col_str = weighted_pmi / k_freq if k_freq else 0.0
        
        # Syntactic Context Adversity
        sca = len(pre_tags.get(key, set())) + len(post_tags.get(key, set()))
        
        rows.append({
            "keyword": key,
            "occurrences": k_freq,
            "avg_character_count": round(avg_char_count, 2),
            "amount_distinct_neighbors": distinct_neighbors,
            "word_entropy": round(word_entropy, 3),
            "collocation_strength": round(col_str, 3),
            "synthetic_context_adversity": sca,
            "gramm_score": score_map.get(key, None)
        })
    return pd.DataFrame(rows).sort_values("occurrences", ascending=False)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    corpus_to_process = sample_corpus(DATA_PATH, f"{DATA_PATH}_sample.txt", SENTENCES_COUNT)
    variants, score_map, max_kw_len = read_keywords(KEYWORDS_CSV)
    lookup = build_variant_lookup(variants)

    # *** UPDATED: Get both counts at once ***
    total_sentences, total_tokens = count_total_sentences_and_tokens(corpus_to_process)
    if total_sentences == 0:
        print(f"No sentences found in '{corpus_to_process}'. Exiting.")
        return

    # Initialize aggregators
    agg_freq, agg_contexts, agg_pre_tags, agg_post_tags, agg_bigram_counts, agg_token_counts = \
        defaultdict(int), defaultdict(list), defaultdict(set), defaultdict(set), defaultdict(int), defaultdict(int)
    
    num_batches = math.ceil(total_sentences / PROCESSING_BATCH_SIZE)
    batch_generator = read_corpus_batches(corpus_to_process, PROCESSING_BATCH_SIZE)
    
    print(f"\nStarting metric calculation on {total_sentences} sentences in {num_batches} batches...")
    
    for batch_tokens_tags in tqdm(batch_generator, total=num_batches, desc="Processing Corpus Batches"):
        # Count token occurrences for the entire corpus (for PMI)
        for tok, _ in batch_tokens_tags:
            agg_token_counts[tok] += 1
        
        # Run sliding window on the current batch
        b_freq, b_contexts, b_pre, b_post, b_bigram = \
            sliding_window_match(batch_tokens_tags, lookup, max_kw_len)
        
        # Aggregate results
        for k, v in b_freq.items(): agg_freq[k] += v
        for k, v in b_contexts.items(): agg_contexts[k].extend(v)
        for k, v in b_pre.items(): agg_pre_tags[k].update(v)
        for k, v in b_post.items(): agg_post_tags[k].update(v)
        for k, v in b_bigram.items(): agg_bigram_counts[k] += v

    print("\nBatch processing finished. Aggregating results...")
    
    df_metrics = compute_metrics(
        variants, agg_freq, agg_contexts, agg_pre_tags, agg_post_tags,
        agg_bigram_counts, agg_token_counts, score_map, total_tokens
    )
    
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics successfully calculated and saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()