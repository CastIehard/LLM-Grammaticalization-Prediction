#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a leaked subset from a parquet/streamed Hugging Face dataset.

Sources supported:
  - wikipedia  -> wikimedia/wikipedia (parquet; pick a dump + lang)
  - c4         -> allenai/c4          (parquet stream; specify lang)
  - pile       -> monology/pile-uncopyrighted

Output:
  CSV file containing matched Keywords, their Snippets, and Scores.
"""

import re
import unicodedata
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


# -------- CONFIG --------
SOURCE = "c4"     # "wikipedia" | "c4" | "pile"
WIKI_CONFIG = "20231101.de"  # For wikipedia: dump + lang code
C4_LANG  = "de"              # For c4: language code
MAX_ROWS_TO_SCAN     = 50_000
MAX_UNIQUE_KEYWORDS  = 80

TESTSET_ORIGINAL = "/content/testset_original.csv"
OUT_CSV          = f"/content/leaked_from_source_{SOURCE}_de.csv"


# ---------- Helper functions ----------
def _coerce_score_series(s):
    """
    Convert a score column to integer values 1–4 if possible.
    Tries numeric conversion first, else extracts last digit 1–4 from string.
    """
    import numpy as np

    def extract_1to4(x):
        if pd.isna(x):
            return np.nan
        digs = [ch for ch in str(x) if ch in "1234"]
        return int(digs[-1]) if digs else np.nan

    s_num = pd.to_numeric(s, errors="coerce")
    cand = s_num if s_num.notna().any() else s.apply(extract_1to4)
    cand = cand.where(cand.isin([1, 2, 3, 4]), np.nan)
    return cand.astype("Int64")


def _looks_like_keyword_col(s):
    """
    Heuristic check if a column contains keyword-like strings.
    - At least 60% entries contain letters
    - Average length >= 2
    """
    sample = s.dropna().astype(str).head(100)
    if sample.empty:
        return False
    alpha = sample.str.contains(r"[A-Za-zÄÖÜäöüß]", regex=True).mean()
    return alpha >= 0.6 and sample.str.len().mean() >= 2


def load_df_flexible_tsv_first(csv_path: str):
    """
    Load a TSV/CSV test set and detect Keyword and Score columns.
    Tries multiple delimiter/header combinations.
    Returns a DataFrame with 'Keyword' and 'Score' columns.
    """
    try:
        df = pd.read_csv(csv_path, sep="\t", header=None, dtype=str, on_bad_lines="skip")
    except Exception:
        df = None

    if df is None or df.shape[1] == 1:
        for opts in [
            dict(sep=None, engine="python", header=None),
            dict(sep=None, engine="python", header=0),
            dict(sep=",", header=None), dict(sep=",", header=0),
            dict(sep=";", header=None), dict(sep=";", header=0),
            dict(sep="\t", header=0),
        ]:
            try:
                tmp = pd.read_csv(csv_path, **opts, dtype=str, on_bad_lines="skip")
                if tmp.shape[1] >= 1:
                    df = tmp.copy()
                    break
            except:
                pass
    if df is None:
        raise RuntimeError(f"Could not read {csv_path}")

    kw_col = None
    for c in df.columns:
        if _looks_like_keyword_col(df[c]):
            kw_col = c
            break
    if kw_col is None:
        kw_col = df.columns[0]

    best_c, best_ratio = None, -1
    for c in df.columns:
        cand = _coerce_score_series(df[c])
        ratio = cand.notna().mean()
        if ratio > best_ratio and ratio >= 0.3:
            best_c, best_ratio = c, ratio
    sc_col = best_c if best_c is not None else df.columns[-1]

    out = pd.DataFrame({
        "Keyword": df[kw_col].astype(str).str.strip(),
        "Score": _coerce_score_series(df[sc_col])
    }).dropna(subset=["Keyword", "Score"]).reset_index(drop=True)
    print(f"[Loader] Keyword col: {kw_col}, Score col: {sc_col}, Rows: {len(out)}")
    return out


def build_patterns(words):
    """
    Build compiled regex patterns for keyword matching.
    Uses Unicode normalization (NFKC) and case-insensitive match.
    """
    pats = {}
    for w in words:
        w_norm = unicodedata.normalize("NFKC", w)
        pats[w] = re.compile(rf"\b{re.escape(w_norm)}\b", flags=re.IGNORECASE)
    return pats


def open_stream(source: str):
    """
    Open a streaming Hugging Face dataset.
    Returns (dataset_iterable, text_field_name).
    """
    if source == "wikipedia":
        ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG, split="train", streaming=True)
        return ds, "text"
    elif source == "c4":
        ds = load_dataset("allenai/c4", C4_LANG, split="train", streaming=True)
        return ds, "text"
    elif source == "pile":
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        return ds, "text"
    else:
        raise ValueError("SOURCE must be one of: 'wikipedia', 'c4', 'pile'")


def main():
    """
    Main script logic:
    1. Load test set.
    2. Stream chosen dataset.
    3. Search for keyword matches.
    4. Save matched leaked subset to CSV.
    """
    df_orig = load_df_flexible_tsv_first(TESTSET_ORIGINAL)
    keywords = sorted(set(df_orig["Keyword"].astype(str).str.lower()))
    patterns = build_patterns(keywords)

    print(f"\nOpening stream for SOURCE='{SOURCE}' …")
    ds, text_field = open_stream(SOURCE)

    found = {}
    rows_scanned = 0

    for ex in tqdm(ds, total=MAX_ROWS_TO_SCAN, desc=f"Scanning {SOURCE}"):
        txt = (ex.get(text_field) or "").strip()
        if not txt:
            continue
        txt_norm_lc = unicodedata.normalize("NFKC", txt).lower()

        for kw, pat in patterns.items():
            if kw not in found and pat.search(txt_norm_lc):
                found[kw] = txt[:800]
                if len(found) >= MAX_UNIQUE_KEYWORDS:
                    break

        rows_scanned += 1
        if rows_scanned >= MAX_ROWS_TO_SCAN or len(found) >= MAX_UNIQUE_KEYWORDS:
            break

    df_leaked = pd.DataFrame({
        "Keyword": list(found.keys()),
        "Snippet": [found[k] for k in found.keys()]
    })
    df_leaked = df_leaked.merge(df_orig[["Keyword", "Score"]], on="Keyword", how="left")
    df_leaked = df_leaked.drop_duplicates("Keyword").reset_index(drop=True)
    df_leaked.to_csv(OUT_CSV, index=False)

    print(f"\nBuilt leaked subset: {len(df_leaked)} unique keywords")
    print(f"Rows scanned: {rows_scanned}")
    print("Saved to:", OUT_CSV)
    print(df_leaked.head(10))


if __name__ == "__main__":
    main()
