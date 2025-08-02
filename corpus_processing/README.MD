# Corpus Processing Pipeline

This directory contains a comprehensive pipeline for processing German grammaticalization corpus data and calculating linguistic metrics for preposition classification. The pipeline implements a sliding window approach to extract contextual features from the SDEWAC corpus.

## Pipeline Overview

The processing consists of three main stages:

1. **Keyword Metrics Calculation** (`1_calculate_keyword_metrics.py`)
2. **Dataset Creation and Splitting** (`2_create_dataset.py`) 
3. **Statistical Evaluation** (`3_evaluation.py`)

## Stage 1: Keyword Metrics Calculation

### Input Data
- **Corpus**: SDEWAC v3 corpus (`sdewac-v3.txt`) - A large German web corpus with POS tagging
- **Keywords**: Ground truth CSV (`keyword_groundtruth.csv`) containing 206 prepositions/preposition-like expressions with grammaticalization scores (1-4)

### Sliding Window Analysis

The script implements a sophisticated sliding window approach to analyze keyword occurrences:

1. **Variable Window Size**: Uses windows of 1 to N tokens (where N is the maximum keyword length)
2. **Keyword Variants**: Handles multi-word expressions and variants (e.g., "auf Grund/aufgrund")
3. **Context Extraction**: For each keyword occurrence, extracts:
   - Preceding and following tokens
   - POS tags of neighboring words
   - Bigram co-occurrence patterns

### Calculated Metrics

For each keyword, the following metrics are computed:

#### Core Frequency Metrics
- **Occurrences**: Total frequency in the corpus
- **Average Character Count**: Mean character length of keyword tokens

#### Contextual Diversity Metrics
- **Amount of Distinct Neighbors**: Number of unique words appearing adjacent to the keyword
- **Word Entropy (Shannon)**: Entropy of the distribution of neighboring words
  ```
  H = -Σ p(w) × log₂(p(w))
  ```
  where p(w) is the probability of word w appearing in the context

#### Syntactic Metrics
- **Synthetic Context Adversity (SCA)**: Number of distinct POS tags appearing before and after the keyword
- **Collocation Strength**: Weighted Pointwise Mutual Information (PMI) of keyword-context pairs
  ```
  PMI = log₂(P(keyword,context) / (P(keyword) × P(context)))
  ```

### Processing Methodology

- **Batch Processing**: Corpus is processed in batches (100K sentences) to manage memory
- **Efficient Counting**: Single-pass counting of total tokens for PMI calculations
- **Aggregation**: Results from all batches are aggregated to compute final metrics

## Stage 2: Dataset Creation and Splitting

### Sentence Extraction

The script extracts example sentences containing keywords using:

1. **Sliding Window Matching**: Same approach as metrics calculation
2. **Greedy Selection**: For sentences with multiple keywords, selects the least frequent one
3. **Sentence Marking**: Keywords are marked with brackets `[keyword]` in output
4. **Deduplication**: Ensures no duplicate sentences across the dataset

### Three-Way Data Split

The dataset is split into three distinct sets ensuring **no keyword overlap**:

#### 1. Dev Data 1 (Few-shot Prompting)
- **Purpose**: Provides examples for few-shot prompting
- **Selection**: 2 keywords randomly selected from each grammaticalization level (1-4)
- **Total**: 8 keywords (2 × 4 levels)
- **Output**: `data/dev data 1 (for prompting)/`

#### 2. Dev Data 2 (Development Testing)
- **Purpose**: Development and hyperparameter tuning
- **Selection**: 50% of remaining keywords after few-shot selection
- **Output**: `data/dev data 2 (for testing)/`

#### 3. Test Data (Final Evaluation)
- **Purpose**: Final model evaluation (use only once)
- **Selection**: Remaining 50% of keywords
- **Output**: `data/test data (only use at the end)/`

### Output Files

For each dataset split, the following files are generated:
- **JSONL file**: Contains sentences with keyword metrics and grammaticalization scores
- **CSV file**: Keyword metrics subset for that specific split
- **Distribution plot**: Visual representation of grammaticalization score distribution

## Stage 3: Statistical Evaluation

### Distribution Analysis

Generates distribution plots showing the number of keywords per grammaticalization level for each dataset split, enabling verification of balanced distributions.

### Metric Performance Evaluation

Evaluates how well each calculated metric correlates with human grammaticalization annotations:

#### Spearman's Rank Correlation (ρ)
- Measures overall ranking correlation between metric values and grammaticalization scores
- Higher values indicate better ranking performance

#### Average Precision (AP) Scores
- Evaluates pairwise discrimination between grammaticalization levels
- Computed for all pairs: (1 vs 2), (1 vs 3), (1 vs 4), (2 vs 3), (2 vs 4), (3 vs 4)
- Higher scores indicate better binary classification performance

### Evaluation Scope

- **Full Data**: Complete evaluation on all 206 keywords
- **Dev Data 2**: Development set evaluation for model tuning
- **Test Data**: Final evaluation metrics
- **Dev Data 1**: Not evaluated (reserved for prompting examples)