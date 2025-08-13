#!/usr/bin/env python3
# ============================================
# Leakage Checks — TinyLlama & FLAN‑T5
# - Robust TSV/CSV loader (no headers OK)
# - Methods: M1..M4 + ICL + Half-Set context
# - Per-method CSVs, summary CSVs, plots
# - Manual config for datasets & models (no argparse)
# - OpenAI backend support for gpt-4o / gpt-4o-mini  ✅
# ============================================

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# ---- NEW: OpenAI + .env (optional; only used if you list an "openai:*" model) --
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads OPENAI_API_KEY if you have a .env file
except Exception:
    pass
# ---------------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============== MANUAL CONFIG (EDIT THESE) ==============
# List of (path, tag) pairs. Add your leaked CSVs here.
DATASETS = [
    ("/content/testset_original.csv", "original"),
    # Uncomment any of your leaked sets:
    # ("/content/leaked_from_source_c4_de.csv", "c4de"),
    # ("/content/leaked_from_source_c4.csv", "c4en"),
    # ("/content/leaked_from_source_wiki.csv", "wiki"),
    # ("/content/leaked_from_source_pile.csv", "pile"),
]

# Which models to test:
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-base",
    # Add either of these to use OpenAI:
    # "openai:gpt-4o",
    # "openai:gpt-4o-mini",
]

# Root folder for outputs:
OUT_ROOT = "./results"
# ========================================================


# ---------- 1) Robust TSV-first loader ----------

def _coerce_score_series(s: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to integer scores in {1,2,3,4}.
    Strategy:
      1) Try numeric conversion.
      2) If that fails, extract the last digit in {1,2,3,4} from string cells.
    Returns a nullable Int64 dtype series.
    """
    def extract_1to4(x):
        if pd.isna(x):
            return np.nan
        digs = [ch for ch in str(x) if ch in "1234"]
        return int(digs[-1]) if digs else np.nan
    s_num = pd.to_numeric(s, errors="coerce")
    cand = s_num if s_num.notna().any() else s.apply(extract_1to4)
    cand = cand.where(cand.isin([1, 2, 3, 4]), np.nan)
    return cand.astype("Int64")


def _looks_like_keyword_col(s: pd.Series) -> bool:
    """
    Heuristic to detect a keyword-like column:
      - ≥60% of entries contain letters (including German diacritics)
      - average string length ≥ 2
    """
    sample = s.dropna().astype(str).head(100)
    if sample.empty:
        return False
    alpha = sample.str.contains(r"[A-Za-zÄÖÜäöüß]", regex=True).mean()
    return alpha >= 0.6 and sample.str.len().mean() >= 2


def load_df_flexible_tsv_first(csv_path: str) -> pd.DataFrame:
    """
    Load a TSV/CSV (unknown format) and return a DataFrame with columns:
      - Keyword (str)
      - Score   (Int64 in {1,2,3,4})
    Tries TSV-without-header first, then multiple delimiter/header combos.
    Guesses Keyword & Score columns by content.
    """
    # Prefer TSV without header
    try:
        df = pd.read_csv(csv_path, sep="\t", header=None, dtype=str, on_bad_lines="skip")
    except Exception:
        df = None
    # Fallbacks
    if df is None or df.shape[1] == 1:
        try_orders = [
            dict(sep=None, engine="python", header=None),
            dict(sep=None, engine="python", header=0),
            dict(sep=",", header=None), dict(sep=",", header=0),
            dict(sep=";", header=None), dict(sep=";", header=0),
            dict(sep="\t", header=0)
        ]
        for opts in try_orders:
            try:
                tmp = pd.read_csv(csv_path, **opts, dtype=str, on_bad_lines="skip")
                if tmp.shape[1] >= 1:
                    df = tmp.copy(); break
            except:
                pass
    if df is None:
        raise RuntimeError(f"Could not read {csv_path}")

    # Guess Keyword & Score by content
    kw_col, sc_col = None, None
    for c in df.columns:
        if _looks_like_keyword_col(df[c]): kw_col = c; break
    best_c, best_ratio = None, -1
    for c in df.columns:
        coerced = _coerce_score_series(df[c])
        ratio = coerced.notna().mean()
        if ratio > best_ratio and ratio >= 0.3:
            best_c, best_ratio = c, ratio
    sc_col = best_c

    if kw_col is None: kw_col = df.columns[0]
    if sc_col is None: sc_col = df.columns[-1]

    out = pd.DataFrame({
        "Keyword": df[kw_col].astype(str).str.strip(),
        "Score": _coerce_score_series(df[sc_col])
    }).dropna(subset=["Keyword","Score"]).reset_index(drop=True)

    print(f"[Loader] Keyword col: {kw_col}, Score col: {sc_col}, Rows: {len(out)}")
    print(out.head(5))
    print("Score distribution:\n", out["Score"].value_counts().sort_index())
    return out


# ---------- 2) Model loading + query wrappers ----------

def load_model(model_name: str):
    """
    Load a Hugging Face model (TinyLlama, FLAN‑T5, etc.) on DEVICE.
    Returns: (tokenizer, model, mode)
      - mode='seq2seq' for encoder-decoder (e.g., FLAN‑T5)
      - mode='causal'  for decoder-only   (e.g., TinyLlama)
    """
    print(f"\nLoading model (HF): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "flan-t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE); mode = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE); mode = "causal"
    return tokenizer, model, mode


def make_query_fn(tokenizer, model, mode):
    """
    Create a callable `query_fn(prompt, max_tokens=32) -> str` that runs
    generation for the selected architecture, decodes and trims the response.
    """
    if mode == "seq2seq":  # FLAN‑T5
        def query(prompt, max_tokens=32):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            return tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return query
    else:  # TinyLlama (decoder-only)
        def query(prompt, max_tokens=32):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(inputs["input_ids"], max_new_tokens=max_tokens, do_sample=False)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            return decoded[len(prompt):].strip()
        return query

# ---- NEW: OpenAI backend (gpt-4o / gpt-4o-mini) ----------------

_OAI_CLIENT = None

def _get_openai_client():
    """
    Lazily create an OpenAI client using OPENAI_API_KEY from env/.env.
    """
    global _OAI_CLIENT
    if _OAI_CLIENT is None:
        if OpenAI is None:
            raise RuntimeError("Please `pip install openai python-dotenv` to use GPT‑4o / GPT‑4o‑mini.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or a .env file.")
        _OAI_CLIENT = OpenAI(api_key=api_key)
    return _OAI_CLIENT


def make_openai_query_fn(oai_model: str):
    """
    Return a `query(prompt, max_tokens)` function backed by OpenAI Chat Completions.
    Works with 'gpt-4o' and 'gpt-4o-mini'.
    """
    client = _get_openai_client()

    def _query(prompt: str, max_tokens: int = 64) -> str:
        resp = client.chat.completions.create(
            model=oai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    return _query


def load_or_route_model(model_name: str):
    """
    Unified loader/router:
      - If `model_name` starts with 'openai:', return (None, None, 'openai', query_fn)
      - Else, load a Hugging Face model and return (tokenizer, model, mode, query_fn)
    """
    if model_name.startswith("openai:"):
        oai_model = model_name.split("openai:", 1)[1]  # e.g., 'gpt-4o' or 'gpt-4o-mini'
        print(f"\nRouting to OpenAI Chat Completions: {oai_model}")
        query_fn = make_openai_query_fn(oai_model)
        return None, None, "openai", query_fn

    tok, mdl, mode = load_model(model_name)
    query = make_query_fn(tok, mdl, mode)
    return tok, mdl, mode, query
# ---------------------------------------------------------


# ---------- 3) Utilities (masking, fake gen, inventory) ----------

def mask_from_keyword(kw: str) -> str:
    """Return a masked form '[x___]' using the first character of the keyword."""
    return f"[{kw[0]}___]"


def extract_last_digit_1_4(text: str):
    """Extract the last digit in {1,2,3,4} from a generated text (or return None)."""
    if not isinstance(text, str): return None
    digs = [ch for ch in text if ch in "1234"]
    return int(digs[-1]) if digs else None


GERMANISH_SYLL = [
    "ab","an","auf","aus","bei","be","durch","ein","ent","er","ge","hinter",
    "hin","her","in","im","mit","nach","neben","ober","unter","über","um","ver",
    "von","vor","zu","zum","zur","zwischen"
]

def _mutate_like(word: str) -> str:
    """
    Create a fake German-looking token by simple mutations:
      - random insertion/duplication/swap
      - add German-ish prefix/suffix
    """
    ops=[]
    i=random.randint(0,len(word))
    ops.append(word[:i]+random.choice("bcdfghjklmnpqrstvwxyz")+word[i:])
    if len(word)>1:
        j=random.randint(0,len(word)-1)
        ops.append(word[:j]+word[j]+word[j:])
    if len(word)>2:
        k=random.randint(0,len(word)-2)
        ops.append(word[:k]+word[k+1]+word[k]+word[k+2:])
    ops.append(random.choice(GERMANISH_SYLL)+word)
    ops.append(word+random.choice(["lich","bar","los","chen","lein","zu","hin"]))
    return re.sub(r"[^a-zäöüß]", "", random.choice(ops).lower())


def generate_fake_keywords(n: int, forbid: set) -> list:
    """
    Generate `n` fake keywords not present in `forbid` (lowercased).
    Uses lightweight morphological mutations to look German-ish.
    """
    fakes=set(); seeds=list(forbid)+GERMANISH_SYLL; tries=0
    while len(fakes)<n and tries<10000:
        base=random.choice(seeds)
        if len(base)>6: base=base[:random.randint(3,6)]
        cand=_mutate_like(base)
        if len(cand)>=3 and cand not in forbid:
            fakes.add(cand)
        tries+=1
    return sorted(fakes)


def expected_inventory(df: pd.DataFrame):
    """
    Compute inventory stats:
      - unique keywords
      - per-score buckets
      - per-score counts
    Useful for sanity checks and saving small summaries.
    """
    uniq=sorted(df["Keyword"].str.lower().unique())
    per_score=(df.assign(kw=df["Keyword"].str.lower())
                 .drop_duplicates("kw")
                 .groupby("Score")["kw"]
                 .apply(lambda s: sorted(list(s))).to_dict())
    stats={
        "n_items":len(df),
        "n_unique":len(uniq),
        "by_score_counts": df.groupby("Score")["Keyword"].nunique().to_dict()
    }
    return uniq, per_score, stats


# ---------- 4) Methods ----------

def method1_partial_recall(df: pd.DataFrame, query_fn):
    """
    Method 1: Partial keyword recall from first letter mask.
    Prompt: "Fill in the missing German preposition: '[x___]'."
    Returns DataFrame with (keyword, pred, recovered: bool).
    """
    rows=[]
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Method1"):
        kw=r["Keyword"]
        out=query_fn(f"Fill in the missing German preposition: '{mask_from_keyword(kw)}'.")
        rows.append({"keyword":kw, "pred":out, "recovered": kw.lower() in out.lower()})
    return pd.DataFrame(rows)


def method_icl_recall(df: pd.DataFrame, query_fn, k=10, trials=3, seed=0):
    """
    ICL variant of partial recall:
      - Sample k seed keywords as examples (context list).
      - For each remaining keyword, ask to complete the masked form.
    Returns DataFrame with trial index and recovered flag.
    """
    rng=random.Random(seed); all_rows=[]
    for t in range(trials):
        seeds=df.sample(n=min(k,len(df)), random_state=rng.randint(0,10**9))
        ctx=", ".join(seeds["Keyword"].tolist())
        hold=df.drop(seeds.index)
        for _, r in tqdm(hold.iterrows(), total=len(hold), desc=f"ICL trial {t+1}"):
            kw=r["Keyword"]
            prompt=("Here are German prepositions:\n"
                    f"{ctx}\n\nNow complete: '{mask_from_keyword(kw)}'.")
            out=query_fn(prompt)
            all_rows.append({"trial":t,"keyword":kw,"pred":out,"recovered": kw.lower() in out.lower(),"k_shots":len(seeds)})
    return pd.DataFrame(all_rows)


def method2_real_vs_fake(df: pd.DataFrame, query_fn, forbid_set: set, extra_fake_count=250):
    """
    Method 2: Real vs. Fake recovery gap.
      - Query real keywords with masked form.
      - Query many fake, German-ish non-words.
      - Compare recovery rates (real vs fake).
    """
    seed_fakes=["abvon","jeztlich","bezeug","zumaln","entunter","durchhin","alben","überzu"]
    seed_fakes=[w for w in seed_fakes if w not in forbid_set]
    more_fakes=generate_fake_keywords(extra_fake_count, forbid_set)
    fake_keywords=seed_fakes+more_fakes

    real_rows=[]; fake_rows=[]
    for kw in tqdm(df["Keyword"], desc="M2-real"):
        out=query_fn(f"Complete the German preposition: '{mask_from_keyword(kw)}'")
        real_rows.append({"keyword":kw,"pred":out,"recovered": kw.lower() in out.lower()})
    for fk in tqdm(fake_keywords, desc="M2-fake"):
        out=query_fn(f"Complete the German preposition: '{mask_from_keyword(fk)}'")
        fake_rows.append({"keyword":fk,"pred":out,"recovered": fk.lower() in out.lower()})
    return pd.DataFrame(real_rows), pd.DataFrame(fake_rows)


def method_halfset_context(df: pd.DataFrame, query_fn, frac=0.5, seed=0):
    """
    Half-set context control:
      - Provide half the keywords as a comma-separated context list.
      - Test masked completion on the remaining half.
    Returns (results_df, n_context_items).
    """
    rng=random.Random(seed)
    context=df.sample(frac=frac, random_state=rng.randint(0,10**9))
    rest=df.drop(context.index)
    ctx=", ".join(context["Keyword"].tolist())
    rows=[]
    for _, r in tqdm(rest.iterrows(), total=len(rest), desc="Half-set"):
        kw=r["Keyword"]
        out=query_fn("Below is a list of German prepositions (context):\n"
                     f"{ctx}\n\nNow complete: '{mask_from_keyword(kw)}'.")
        rows.append({"keyword":kw,"pred":out,"recovered": kw.lower() in out.lower()})
    return pd.DataFrame(rows), len(context)


def method3_next_in_score(df: pd.DataFrame, query_fn, sample_size=5):
    """
    Method 3: Score-based "suggest another".
      - For each score bucket, show a small list of examples.
      - Ask the model to suggest another from the same score.
    Returns DataFrame with one row per score bucket.
    """
    preds=[]
    for score, grp in df.groupby("Score"):
        ks=grp["Keyword"].tolist()
        ctx=", ".join(random.sample(ks, min(len(ks), sample_size)))
        out=query_fn(f"Here are German prepositions with grammaticalization score {score}: {ctx}. "
                     f"Suggest another from the same score:")
        preds.append({"Score":score,"Prompt":f"...{ctx}...", "ModelPrediction":out})
    return pd.DataFrame(preds)


def method4_predict_score(df: pd.DataFrame, query_fn):
    """
    Method 4: Predict score from keyword.
      - Ask: 'What is the grammaticalization score (1..4) of <keyword>?'
      - Parse last digit in {1,2,3,4} from model output.
    Returns DataFrame (keyword, true_score, predicted_score, raw).
    """
    rows=[]
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Method4"):
        kw, true = r["Keyword"], int(r["Score"])
        out=query_fn(f"What is the grammaticalization score (1=very low, 4=very high) "
                     f"of the German preposition '{kw}'? Respond with one digit 1-4.")
        rows.append({"keyword":kw,"true_score":true,"predicted_score": extract_last_digit_1_4(out),"raw":out})
    return pd.DataFrame(rows)


# ---------- 5) Metrics + plots ----------

def m1_summary(df: pd.DataFrame) -> dict:
    """Return recovery rate for Method 1."""
    return {"m1_recovery": float(df["recovered"].mean())}

def icl_summary(df: pd.DataFrame) -> dict:
    """Return recovery rate for ICL partial recall."""
    return {"icl_recall": float(df["recovered"].mean())}

def m2_summary(real_df: pd.DataFrame, fake_df: pd.DataFrame) -> dict:
    """Return real/fake recovery rates and their gap (real - fake)."""
    rr=float(real_df["recovered"].mean()) if len(real_df) else np.nan
    rf=float(fake_df["recovered"].mean()) if len(fake_df) else np.nan
    return {"m2_real": rr, "m2_fake": rf, "m2_gap": (rr-rf) if (not np.isnan(rr) and not np.isnan(rf)) else np.nan}

def halfset_summary(df: pd.DataFrame, n_ctx: int) -> dict:
    """Return half-set recall and the number of context items used."""
    return {"halfset_recall": float(df["recovered"].mean()), "halfset_context_n": int(n_ctx)}

def m3_summary(preds: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    """Approximate match rate: how many predictions match any gold keyword (lowercased)."""
    gold=set(gold_df["Keyword"].str.lower().unique())
    hits=sum([str(p).strip().lower() in gold for p in preds["ModelPrediction"]])
    return {"m3_match_rate": hits / max(len(preds),1)}

def m4_summary(df: pd.DataFrame) -> dict:
    """Overall accuracy for score prediction where a valid digit was parsed."""
    d=df.dropna(subset=["predicted_score"])
    return {"m4_overall_acc": float((d["true_score"]==d["predicted_score"]).mean()) if len(d) else np.nan}

def save_bar(scores: dict, title: str, out_png: str):
    """Save a compact barplot for the provided metric dictionary."""
    labels=list(scores.keys()); vals=[scores[k] for k in labels]
    plt.figure(figsize=(9,4))
    bars=plt.bar(labels, vals)
    plt.ylim(0,1); plt.ylabel("Score"); plt.title(title)
    for b,v in zip(bars, vals): plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2%}", ha='center')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show()

def plot_all_runs(summary_csv="all_runs_summary.csv"):
    """
    Plot per-metric bars across all (model,dataset) rows aggregated into summary CSV.
    Expects columns:
      model, dataset, m1_recovery, icl_recall, m2_real, m2_fake, halfset_recall, m3_match_rate, m4_overall_acc
    """
    df = pd.read_csv(summary_csv)
    keep = ["m1_recovery", "icl_recall", "m2_real", "m2_fake",
            "halfset_recall", "m3_match_rate", "m4_overall_acc"]
    melted = df.melt(id_vars=["model","dataset"], value_vars=keep,
                     var_name="metric", value_name="score")
    for metric, g in melted.groupby("metric"):
        plt.figure(figsize=(10,4))
        labels = [f"{m.split('/')[-1]}-{d}" for m,d in zip(g["model"], g["dataset"])]
        plt.bar(labels, g["score"])
        plt.ylim(0,1); plt.ylabel("Score"); plt.title(metric)
        for i,y in enumerate(g["score"]): plt.text(i, y+0.02, f"{y:.2%}", ha='center')
        plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.show()


# ---------- 6) Runner per model×dataset ----------

def run_all_on_dataset(model_name: str, tag: str, df: pd.DataFrame, outdir: str, forbid_set: set):
    """
    Run all leakage methods and metrics for (model, dataset):
      - Saves per-method CSVs in outdir
      - Saves a per-run summary plot
      - Returns a flat dict of metric scores for aggregation
    """
    os.makedirs(outdir, exist_ok=True)

    # CHANGED (minimal): route to HF or OpenAI backend
    tok, mdl, mode, query = load_or_route_model(model_name)

    # optional sanity inventory
    uniq, per_score, stats = expected_inventory(df)
    pd.Series(uniq).to_csv(os.path.join(outdir, f"{tag}_inventory_unique.csv"), index=False)
    pd.DataFrame({"Score": list(stats["by_score_counts"].keys()),
                  "Count": list(stats["by_score_counts"].values())}).to_csv(
        os.path.join(outdir, f"{tag}_inventory_by_score.csv"), index=False)

    # Methods
    m1 = method1_partial_recall(df, query); m1.to_csv(os.path.join(outdir, f"{tag}_m1.csv"), index=False)
    icl = method_icl_recall(df, query, k=10, trials=3); icl.to_csv(os.path.join(outdir, f"{tag}_icl.csv"), index=False)
    m2_real, m2_fake = method2_real_vs_fake(df, query, forbid_set, extra_fake_count=250)
    m2_real.to_csv(os.path.join(outdir, f"{tag}_m2_real.csv"), index=False)
    m2_fake.to_csv(os.path.join(outdir, f"{tag}_m2_fake.csv"), index=False)
    half_df, n_ctx = method_halfset_context(df, query, frac=0.5)
    half_df.to_csv(os.path.join(outdir, f"{tag}_halfset.csv"), index=False)
    m3 = method3_next_in_score(df, query); m3.to_csv(os.path.join(outdir, f"{tag}_m3.csv"), index=False)
    m4 = method4_predict_score(df, query); m4.to_csv(os.path.join(outdir, f"{tag}_m4.csv"), index=False)

    # Metrics
    m1s = m1_summary(m1)
    icls = icl_summary(icl)
    m2s = m2_summary(m2_real, m2_fake)
    halfs = halfset_summary(half_df, n_ctx)
    m3s = m3_summary(m3, df)
    m4s = m4_summary(m4)

    flat = {"model": model_name, "dataset": tag, **m1s, **icls, **m2s, **halfs, **m3s, **m4s}
    pd.DataFrame([flat]).to_csv(os.path.join(outdir, f"{tag}_metrics_summary.csv"), index=False)

    save_bar(
        {"M1":m1s["m1_recovery"], "ICL":icls["icl_recall"], "M2 Real":m2s["m2_real"],
         "M2 Fake":m2s["m2_fake"], "Half":halfs["halfset_recall"], "M3":m3s["m3_match_rate"],
         "M4":m4s["m4_overall_acc"]},
        title=f"{model_name} on {tag}",
        out_png=os.path.join(outdir, f"{tag}_summary_plot.png")
    )
    return flat


# ---------- 7) Orchestrator (manual config) ----------

def main():
    """
    Entry point:
      - Loads each dataset in DATASETS (original or leaked subset)
      - Runs all methods for each model in MODELS
      - Saves per-run metrics & an aggregated all_runs_summary.csv
      - Plots per-metric bars across runs
    """
    os.makedirs(OUT_ROOT, exist_ok=True)

    all_rows=[]
    for path, tag in DATASETS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        print(f"\n[Dataset] Loading: {path}  (tag={tag})")
        df = load_df_flexible_tsv_first(path)
        forbid = set(df["Keyword"].str.lower().unique())

        for m in MODELS:
            safe_m = m.replace('/','_').replace(':','_')
            outdir = os.path.join(OUT_ROOT, f"{tag}_{safe_m}")
            flat = run_all_on_dataset(m, tag, df, outdir, forbid)
            all_rows.append(flat)

    # Save and plot aggregate
    summary_csv = os.path.join(OUT_ROOT, "all_runs_summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved {summary_csv}")
    plot_all_runs(summary_csv)


if __name__ == "__main__":
    main()
