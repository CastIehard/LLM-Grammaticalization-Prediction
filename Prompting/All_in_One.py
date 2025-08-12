# main_pipeline.py
import os
import re
import itertools
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


from prompts import PromptBuilder

# --- DSPy config (unchanged model; flip later if you switch models) ---
import dspy
dspy.configure(lm=dspy.LM('ollama/tinyllama', api_base='http://localhost:11434'))

# >>> DSPy zero-shot frameworks
# MINIMAL CHANGE: import from dspy_frameworks (and alias to keep your original names)
from dspy_models import (
    make_zs_nodefs,                          # Zero-shot, no definitions, no tuning
    make_zs_withdefs_const as make_zs_withdefs,  # Zero-shot, with definitions (fixed inside the module)
    compile_zero_shot_instruction_optimized, # Zero-shot, instruction-optimized (0-Shot MIPRO)
    LABEL_DEFS,                              # Canonical label definitions (used for WITH-defs paths)
)

# === OPTIONAL: Compile DSPy teleprompter model ===
# teleprompt_model = compile_with_teleprompter(train_examples, accuracy_metric, mode="bootstrap")

# === Load environment variables ===
load_dotenv()

# === Config ===
MODEL_NAME = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

INPUT_FILE = "./data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv"
OUTPUT_DIR = "./evaluation/input_csv (only dev2 nothing else)/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_LABELS = ["A", "B", "C", "D", "E", "F"]

# Descriptive slugs for filenames (no change to prompt logic)
PROMPT_FILENAME_SLUG = {
    "A": "basic_plain",
    "B": "explicit_io",
    "C": "expert_plain",
    "D": "labeldesc_io",
    "E": "fewshot_plain",
    "F": "fewshot_labeldesc",
}
# Make MODEL_NAME filesystem-safe, e.g. "llama2:7b" -> "llama2_7b"
def _model_slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)

MODEL_SLUG = _model_slug(MODEL_NAME)

# === Few-shot examples (dynamic, INTERLEAVED: 1a,2a,3a,4a, 1b,2b,3b,4b) ===
EXAMPLES_FILE = "./data/dev data 1 (for prompting)/dev_data_1_prompting_metrics.csv"  # adjust if needed

try:
    examples_block = PromptBuilder.make_examples_block_from_csv_interleave(
        EXAMPLES_FILE,
        per_level=2,
        seed=42
    )
    print("[Few-shot] Built interleaved examples block from:", EXAMPLES_FILE)
except Exception as e:
    print(f"[Few-shot] Failed to build examples from {EXAMPLES_FILE}: {e}")
    examples_block = ""  # E/F will still run, but without examples if this fails

# === LLM Call Helper ===
def call_ollama(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return response.json().get("response", "").strip() if response.status_code == 200 else ""
    except Exception as e:
        print(f" Error calling Ollama: {e}")
        return ""

# === DSPy Call Helper (unchanged signature; WITH-defs model is wrapped to inject defs) ===
def call_dspy(model, keyword):
    """Call a DSPy model with a keyword (preposition)."""
    try:
        pred_obj = model(preposition=keyword)
        return str(pred_obj.label)
    except Exception as e:
        print(f" Error calling DSPy: {e}")
        return ""

# === Label Extraction ===
def extract_label(text):
    if not text:
        return None
    match = re.search(r"Label:\s*\[?([1-4])\]?", text)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([1-4])\b", text)
    return int(fallback[-1]) if fallback else 0

# === Load Input ===
df_input = pd.read_csv(INPUT_FILE)
df_input = df_input[["keyword", "gramm_score"]].dropna()
data = df_input.to_dict(orient="records")

# === Inference for 6 BASIC PROMPTS (outputs: predictions-only CSV per run) ===
for label in PROMPT_LABELS:
    slug = PROMPT_FILENAME_SLUG.get(label, label.lower())
    is_fewshot = label in ("E", "F")
    prefix = "FS" if is_fewshot else "ZS"  # Few-shot vs Zero-shot
    description = f"{MODEL_SLUG}_{prefix}_{slug}"
       # descriptive filename

    print(f"\n Running inference with Prompt {label} ({slug}) on {len(data)} keywords using model: {MODEL_NAME}\n")
    results = []

    output_path = os.path.join(OUTPUT_DIR, f"{description}.csv")

    for item in tqdm(data):
        keyword = item["keyword"]
        true_score = float(item["gramm_score"])

        # Build the prompt (E/F need examples_block; others don't)
        try:
            prompt_func = getattr(PromptBuilder, f"build_prompt_{label.lower()}")
            if label in ("E", "F"):
                prompt = prompt_func(keyword, examples_block)
            else:
                prompt = prompt_func(keyword)
        except AttributeError:
            print(f" Missing: PromptBuilder.build_prompt_{label.lower()}()")
            continue

        # === Option 1: Ollama (default)
        response = call_ollama(prompt)
        pred = extract_label(response)

        results.append({
            "keyword": keyword,
            "gramm_score": true_score,
            f"pred_{label}": pred
        })

    # === Save predictions-only CSV
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f" Predictions saved to: {output_path}")

# =====================================================================
#                          DSPy ZERO-SHOT RUNS
#             (outputs: predictions-only CSV per run)
# =====================================================================

# One-click ON: produce all DSPy files in the same run
RUN_DSPY = True          # set False to disable DSPy runs

if RUN_DSPY:
    # 1) Build a tiny labeled set for instruction optimization (0-Shot MIPRO).
    #    Uses your few-shot CSV ONLY for tuning the instruction; no demos are added at inference.
    trainset_for_mipro = []
    try:
        _df_fs = pd.read_csv(EXAMPLES_FILE)
        _kw_col = next((c for c in ["keyword", "Keyword", "preposition", "Preposition"] if c in _df_fs.columns), None)
        _lb_col = next((c for c in ["gramm_score", "Gramm_Score", "label", "Label", "level", "Level"] if c in _df_fs.columns), None)
        if _kw_col is None or _lb_col is None:
            raise ValueError(f"Expected keyword and label columns in {EXAMPLES_FILE}")
        _tmp = _df_fs[[_kw_col, _lb_col]].dropna().copy()
        _tmp[_lb_col] = _tmp[_lb_col].astype(float).round().astype(int)
        _tmp = _tmp[_tmp[_lb_col].between(1, 4)]
        _tmp = _tmp.drop_duplicates(subset=[_kw_col])

        trainset_for_mipro = [
            {"preposition": str(r[_kw_col]), "label": int(r[_lb_col])}
            for _, r in _tmp.iterrows()
        ]
        print(f"[DSPy] Built instruction-optimization trainset from {EXAMPLES_FILE}: {len(trainset_for_mipro)} rows")
    except Exception as e:
        print(f"[DSPy] Failed to build trainset from {EXAMPLES_FILE}: {e}")
        trainset_for_mipro = []

    # 2) Instantiate two untuned zero-shot DSPy models
    dspy_zs_no_defs  = make_zs_nodefs()     # zero-shot, no definitions
    dspy_zs_withdefs = make_zs_withdefs()   # zero-shot, with definitions (const LABEL_DEFS)

    # 3) Compile BOTH optimized variants (if trainset available)
    opt_nodefs = None
    opt_withdefs = None
    if len(trainset_for_mipro) > 0:
        opt_nodefs = compile_zero_shot_instruction_optimized(
            trainset=trainset_for_mipro, use_defs=False, defs=None, auto="medium"
        )
        opt_withdefs = compile_zero_shot_instruction_optimized(
            trainset=trainset_for_mipro, use_defs=True, defs=LABEL_DEFS, auto="medium"
        )
        print("[DSPy] Compiled instruction-optimized zero-shot models (no-defs & with-defs).")
    else:
        print("[DSPy] Skipping instruction optimization (no trainset).")

    # 4) Prepare ALL DSPy runs (4 in total; optimized ones are conditional)
    dspy_runs = [
        ("DSPy_1_nodefs", dspy_zs_no_defs),
        ("DSPy_2_withdefs", dspy_zs_withdefs),
    ]
    if opt_nodefs is not None:
        dspy_runs.append(("DSPy_3_opt_nodefs", opt_nodefs))
    if opt_withdefs is not None:
        dspy_runs.append(("DSPy_4_opt_withdefs", opt_withdefs))

    # 5) Inference & outputs (predictions-only CSV per run)
    for dsp_name, dsp_model in dspy_runs:
        print(f"\n Running inference with {dsp_name} on {len(data)} keywords using DSPy\n")
        results = []

        description = f"{MODEL_NAME}_{dsp_name}"
        output_path = os.path.join(OUTPUT_DIR, f"{description}.csv")

        for item in tqdm(data):
            keyword = item["keyword"]
            true_score = float(item["gramm_score"])

            response = call_dspy(dsp_model, keyword)
            pred = extract_label(response)

            results.append({
                "keyword": keyword,
                "gramm_score": true_score,
                f"pred_{dsp_name}": pred
            })

        # Save predictions-only CSV
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f" Predictions saved to: {output_path}")
