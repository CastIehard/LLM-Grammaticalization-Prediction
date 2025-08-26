# Data‑Leakage Checks for LLMs

Tools to (1) build a likely **leaked subset** of your test data by scanning large corpora and (2) run **leakage-detection experiments** on local HF models (TinyLlama, FLAN‑T5) and OpenAI models (gpt‑4o, gpt‑4o‑mini).

---

## What this repo does

- **Build leaked subsets** by streaming public corpora (Wikipedia, C4/mC4, The Pile) and finding overlaps with your evaluation keywords.
- **Probe models for leakage** with a suite of methods:
  - **M1 – Partial Keyword Recall**
  - **ICL Recall – In‑Context Learning variant of M1**
  - **M2 – Real vs. Fake Keyword Completion**
  - **Half‑Set Context Recall**
  - **M3 – Score‑Based “Suggest Another”**
  - **M4 – Predict Score from Keyword**
- **Log metrics + plots** per model × dataset to the `results/` folder.

---

## Repo layout

```
.
├── results/                         # auto‑generated metrics & plots
├── README.md                        # this file
├── build_leaked_subset.py           # build leaked subsets from big corpora
├── leakage_checks.py                # run M1..M4, ICL, Half‑Set on chosen models
├── leaked_from_source_c4.csv        # example leaked subset (C4 EN)
├── leaked_from_source_c4_de.csv     # example leaked subset (mC4 DE)
├── leaked_from_source_pile.csv      # example leaked subset (Pile)
├── leaked_from_source_wiki.csv      # example leaked subset (Wikipedia)
└── testset_original.csv             # your original evaluation set (TSV/CSV, no header OK)
```

---

## Requirements

- Python 3.9+
- `pip install -U datasets pandas tqdm unidecode torch transformers matplotlib`
- Optional (only if you use OpenAI models in `leakage_checks.py`):
  - `pip install -U openai python-dotenv`
  - Set `OPENAI_API_KEY` in your environment (see below).

---

## Input format

`testset_original.csv` can be TSV/CSV with or without headers.  
The loader auto‑detects:
- **Keyword** column (the German preposition/term),
- **Score** column (must be in {1,2,3,4}; last digit is parsed if needed).

> Tip: Your file currently parses as 206 rows with balanced scores, which the loader prints on start.

---

## Quick start

### 1) (Optional) Build a leaked subset
This scans a big source (Wikipedia/C4/mC4/Pile) and saves matching keywords with a snippet.

```bash
python build_leaked_subset.py
```

**Edit inside the script**:
- `SOURCE = "wikipedia" | "c4" | "pile"`
- For Wikipedia: `WIKI_CONFIG = "20231101.de"` (or `.en`)
- For C4/mC4: set `C4_LANG = "en"` or `"de"`
- Paths: `TESTSET_ORIGINAL`, `OUT_CSV`
- Limits: `MAX_ROWS_TO_SCAN`, `MAX_UNIQUE_KEYWORDS`

Output CSVs (examples already included):
- `leaked_from_source_c4.csv`
- `leaked_from_source_c4_de.csv`
- `leaked_from_source_pile.csv`
- `leaked_from_source_wiki.csv`

### 2) Run leakage checks (M1..M4 + ICL + Half‑Set)

```bash
python leakage_checks.py
```

**Edit inside the script**:

- **Pick datasets** (original + any leaked subsets you want):

```python
DATASETS = [
    ("./testset_original.csv", "original"),
    # ("./leaked_from_source_c4_de.csv", "c4de"),
    # ("./leaked_from_source_c4.csv", "c4en"),
    # ("./leaked_from_source_wiki.csv", "wiki"),
    # ("./leaked_from_source_pile.csv", "pile"),
]
```

- **Pick models**:

```python
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-base",
    # "openai:gpt-4o",
    # "openai:gpt-4o-mini",
]
```

Outputs go to:

```
results/<tag>_<model>/
  ├── <tag>_m1.csv
  ├── <tag>_icl.csv
  ├── <tag>_m2_real.csv
  ├── <tag>_m2_fake.csv
  ├── <tag>_halfset.csv
  ├── <tag>_m3.csv
  ├── <tag>_m4.csv
  ├── <tag>_metrics_summary.csv
  └── <tag>_summary_plot.png
results/all_runs_summary.csv
```

---

## Using OpenAI models (optional)

1) Create a file named `.env` in the project root:

```
OPENAI_API_KEY=sk-************************
```

2) In `leakage_checks.py`, include one or both:

```python
MODELS = [
    # ...
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
]
```

The script auto‑routes prompts via the OpenAI Chat Completions API with `temperature=0.0`.

> If you see `AuthenticationError 401`, check that your key is valid and loaded, or export it in the shell:
> `export OPENAI_API_KEY=sk-...`

---

## Methods & metrics (why these help)

- **M1 – Partial Recall**: Can the model restore a keyword from `[x___]`?  
  High recall hints at prior exposure to the exact tokens.
- **ICL Recall**: Same as M1 but after giving a small list of known examples.  
  If recall spikes here, the model may be pattern‑matching memorized neighbors.
- **M2 – Real vs Fake Gap**: Compare completions for true prepositions vs German‑ish non‑words.  
  A large Real−Fake gap is a classic leakage signal.
- **Half‑Set Context**: Give half the test set as context, test the other half.  
  Helps separate *on‑the‑fly priming* from *pretraining memory*.
- **M3 – Score‑Matching “Suggest Another”**: With a few examples for a score bucket, can the model name another true item?  
  Elevated match rate can indicate exposure to the labeled inventory.
- **M4 – Predict Score from Keyword**: Ask for the grammaticalization level (1–4).  
  Alignment with ground truth can be due to generalization or label memorization; interpret together with other methods.

All metrics aggregate into `results/all_runs_summary.csv` and per‑run plots.

---
 
## Troubleshooting

- **File parsing issues**: The loader tries TSV/CSV with/without headers. If it still fails, open your file and ensure there’s a column of strings (keywords) and a column containing digits 1–4.
- **OpenAI errors**: Ensure `openai` & `python‑dotenv` are installed, `.env` exists, and the key is active.

