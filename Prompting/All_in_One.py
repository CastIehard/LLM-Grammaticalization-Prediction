import os
import re
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import openai
import dspy

from prompts import PromptBuilder
from dspy_models import (
    make_zs_nodefs,
    make_zs_withdefs_const as make_zs_withdefs,
    compile_zero_shot_instruction_optimized,
    LABEL_DEFS,
)

# === CONFIGURATION ===
# Set to False to use OpenAI GPT-4o, True to use local Ollama
LOCAL_LLM = False
# DSPy configuration
RUN_DSPY = False


# Model configurations
LOCAL_MODEL_NAME = "tinyllama"
OPENAI_MODEL_NAME = "gpt-4o"
OLLAMA_URL = "http://localhost:11434"

# File paths
INPUT_FILE = "./data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv"
OUTPUT_DIR = "./evaluation/input_csv (only dev2 nothing else)/"
EXAMPLES_FILE = "./data/dev data 1 (for prompting)/dev_data_1_prompting_metrics.csv"

# Prompt configuration
PROMPT_LABELS = ["A", "B", "C", "D", "E", "F"]
PROMPT_FILENAME_SLUG = {
    "A": "basic_plain",
    "B": "explicit_io",
    "C": "expert_plain",
    "D": "labeldesc_io",
    "E": "fewshot_plain",
    "F": "fewshot_labeldesc",
}



# === ENVIRONMENT SETUP ===
load_dotenv()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure based on LOCAL_LLM flag
if LOCAL_LLM:
    MODEL_NAME = LOCAL_MODEL_NAME
    dspy.configure(lm=dspy.LM(f'ollama/{LOCAL_MODEL_NAME}', api_base=OLLAMA_URL))
    print(f"Using local LLM: {LOCAL_MODEL_NAME}")
else:
    MODEL_NAME = OPENAI_MODEL_NAME
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    openai.api_key = OPENAI_API_KEY
    # For DSPy with OpenAI, configure accordingly
    dspy.configure(lm=dspy.LM(model=f"openai/{OPENAI_MODEL_NAME}", api_key=OPENAI_API_KEY))
    print(f"Using OpenAI: {OPENAI_MODEL_NAME}")

# === UTILITY FUNCTIONS ===
def _model_slug(name: str) -> str:
    """Make MODEL_NAME filesystem-safe, e.g. 'llama2:7b' -> 'llama2_7b'"""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)

def extract_label(text):
    """Extract label from model response"""
    if not text:
        return None
    match = re.search(r"Label:\s*\[?([1-4])\]?", text)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([1-4])\b", text)
    return int(fallback[-1]) if fallback else 0

# === MODEL CALLING FUNCTIONS ===
def call_openai_gpt4o(prompt):
    """Call OpenAI GPT-4o"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI GPT-4o: {e}")
        return ""


def call_ollama(prompt):
    """Call local Ollama model"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LOCAL_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""

def call_llm(prompt):
    """Call the appropriate LLM based on LOCAL_LLM flag"""
    if LOCAL_LLM:
        return call_ollama(prompt)
    else:
        return call_openai_gpt4o(prompt)

def call_dspy(model, keyword):
    """Call a DSPy model with a keyword (preposition)"""
    try:
        pred_obj = model(preposition=keyword)
        return str(pred_obj.label)
    except Exception as e:
        print(f"Error calling DSPy: {e}")
        return ""

# === INITIALIZATION ===
MODEL_SLUG = _model_slug(MODEL_NAME)

# Build few-shot examples
try:
    examples_block = PromptBuilder.make_examples_block_from_csv_interleave(
        EXAMPLES_FILE,
        per_level=2,
        seed=42
    )
    print(f"[Few-shot] Built interleaved examples block from: {EXAMPLES_FILE}")
except Exception as e:
    print(f"[Few-shot] Failed to build examples from {EXAMPLES_FILE}: {e}")
    examples_block = ""

# Load input data
df_input = pd.read_csv(INPUT_FILE)
df_input = df_input[["keyword", "gramm_score"]].dropna()
data = df_input.to_dict(orient="records")

# === MAIN INFERENCE LOOP FOR BASIC PROMPTS ===
print(f"\n=== Running Basic Prompts with {'Local LLM' if LOCAL_LLM else 'OpenAI GPT-4o'} ===")

for label in PROMPT_LABELS:
    slug = PROMPT_FILENAME_SLUG.get(label, label.lower())
    is_fewshot = label in ("E", "F")
    prefix = "FS" if is_fewshot else "ZS"
    description = f"{MODEL_SLUG}_{prefix}_{slug}"

    print(f"\nRunning inference with Prompt {label} ({slug}) on {len(data)} keywords")
    results = []
    output_path = os.path.join(OUTPUT_DIR, f"{description}.csv")

    for item in tqdm(data):
        keyword = item["keyword"]
        true_score = float(item["gramm_score"])

        # Build the prompt
        try:
            prompt_func = getattr(PromptBuilder, f"build_prompt_{label.lower()}")
            if label in ("E", "F"):
                prompt = prompt_func(keyword, examples_block)
            else:
                prompt = prompt_func(keyword)
        except AttributeError:
            print(f"Missing: PromptBuilder.build_prompt_{label.lower()}()")
            continue

        # Call the appropriate LLM
        response = call_llm(prompt)
        pred = extract_label(response)

        results.append({
            "keyword": keyword,
            "gramm_score": true_score,
            f"pred_{label}": pred
        })

    # Save predictions
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

# === DSPY ZERO-SHOT RUNS ===
if RUN_DSPY:
    print(f"\n=== Running DSPy Models ===")
    
    # Build training set for instruction optimization
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
        print(f"[DSPy] Built instruction-optimization trainset: {len(trainset_for_mipro)} rows")
    except Exception as e:
        print(f"[DSPy] Failed to build trainset from {EXAMPLES_FILE}: {e}")

    # Instantiate DSPy models
    dspy_zs_no_defs = make_zs_nodefs()
    dspy_zs_withdefs = make_zs_withdefs()

    # Compile optimized variants
    opt_nodefs = None
    opt_withdefs = None
    if len(trainset_for_mipro) > 0:
        opt_nodefs = compile_zero_shot_instruction_optimized(
            trainset=trainset_for_mipro, use_defs=False, defs=None, auto="medium"
        )
        opt_withdefs = compile_zero_shot_instruction_optimized(
            trainset=trainset_for_mipro, use_defs=True, defs=LABEL_DEFS, auto="medium"
        )
        print("[DSPy] Compiled instruction-optimized zero-shot models")
    else:
        print("[DSPy] Skipping instruction optimization (no trainset)")

    # Prepare DSPy runs
    dspy_runs = [
        ("DSPy_1_nodefs", dspy_zs_no_defs),
        ("DSPy_2_withdefs", dspy_zs_withdefs),
    ]
    if opt_nodefs is not None:
        dspy_runs.append(("DSPy_3_opt_nodefs", opt_nodefs))
    if opt_withdefs is not None:
        dspy_runs.append(("DSPy_4_opt_withdefs", opt_withdefs))

    # Run DSPy inference
    for dsp_name, dsp_model in dspy_runs:
        print(f"\nRunning inference with {dsp_name} on {len(data)} keywords")
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

        # Save predictions
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

print("\n=== Pipeline Complete ===")
