import os
import json
import re
import itertools
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# === Config ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tinyllama"
INPUT_FILE = "test_set.jsonl"

TAG = f"{MODEL_NAME}_ZS_ABC_basic"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"predictions_{TAG}.csv")
KEYWORD_AGG_FILE = os.path.join(OUTPUT_DIR, f"keyword_predictions_summary_{TAG}.csv")
EVAL_FILE = os.path.join(OUTPUT_DIR, f"evaluation_summary_{TAG}.csv")
SAMPLED_IO_FILE = os.path.join(OUTPUT_DIR, "sampled_io_examples.csv")

# === Prompt Definitions ===
DEFINITIONS = """Level 1: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

# === Prompt Builders ===
def build_prompt_a(sentence):
    return f"""You will be given a sentence containing a bracketed German preposition.  
Your task is to classify how grammaticalized that preposition is, based on how fixed or reduced its structure appears.  

Input sentence:  
{sentence}

Choose one of four levels to indicate the degree of grammaticalization:  
Label: [1] or Label: [2] or Label: [3] or Label: [4]"""

def build_prompt_b(sentence):
    return f"""You are a highly trained text data annotation tool capable of providing subjective responses.
Your task is to rate the degree of grammaticalization of a bracketed German preposition in the given sentence.

Sentence:
{sentence}

Focus only on the structure and functional reduction of the bracketed preposition as it appears in this sentence.
Please provide a judgment as a single grammaticalization level using the below definition:

{DEFINITIONS}

Respond strictly in the format to indicate the level of grammaticalization:

Label: [1] or Label: [2] or Label: [3] or Label: [4]

Expected format:  
Label: [<level>]"""

def build_prompt_c(sentence):
    return f"""You are a linguist specializing in German syntax and grammaticalization.
Your task is to classify the degree of grammaticalization of the bracketed preposition in the given sentence, using only the sentence as input. Choose one of the following four levels. Use the definitions below exactly.

{DEFINITIONS}

Respond with only the label in the exact output format below:

Label: [1] or Label: [2] or Label: [3] or Label: [4]

Respond in this exact format exactly:
Label: [<level>]

Input sentence:
{sentence}"""

# === Ollama Helpers ===
def call_ollama(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return response.json().get("response", "").strip() if response.status_code == 200 else ""
    except:
        return ""

def extract_label(text):
    if not text:
        return None
    match = re.search(r"Label:\s*\[?([1-4])\]?", text)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([1-4])\b", text)
    return int(fallback[-1]) if fallback else None

# === Evaluation Function ===
def evaluate_predictions(df, metrics, output_path):
    df.dropna(subset=["gramm_score"], inplace=True)
    df = df.drop_duplicates(subset=["keyword"]).copy()
    if len(df) < 2:
        print("Insufficient keywords for evaluation.")
        return

    results = []
    truth = df["gramm_score"]

    # Spearman's ρ
    rho_row = {"Evaluation": "Spearman's ρ (rank)"}
    for m in metrics:
        rho, _ = spearmanr(truth, df[m])
        rho_row[m] = f"{rho:.2f}"
    results.append(rho_row)

    # Average Precision (AP)
    degrees = sorted(df["gramm_score"].unique())
    for d1, d2 in itertools.combinations(degrees, 2):
        subset = df[df["gramm_score"].isin([d1, d2])]
        y_true = (subset["gramm_score"] == d2).astype(int)
        if y_true.nunique() < 2:
            continue
        row = {"Evaluation": f"AP (degrees {d1} vs. {d2})"}
        for m in metrics:
            ap = average_precision_score(y_true, subset[m])
            row[m] = f"{ap:.2f}"
        results.append(row)

    # Accuracy and per-level accuracy
    for m in metrics:
        preds = df[m].round().astype(int)
        acc = (preds == truth).mean()
        results.append({"Evaluation": "Accuracy (Exact Match)", m: f"{acc:.2f}"})

        for level in sorted(truth.unique()):
            mask = truth == level
            level_acc = (preds[mask] == level).mean()
            results.append({
                "Evaluation": f"Accuracy (Level {level})",
                m: f"{level_acc:.2f}"
            })

    pd.DataFrame(results).set_index("Evaluation").to_csv(output_path)
    print(f"\n✓ Evaluation with exact and level-distributed accuracy saved to {output_path}")

# === Run Inference ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"\n🔍 Running inference on {len(data)} examples...\n")
results = []

for item in tqdm(data):
    sentence = item["sentence_raw"]
    true = int(item["gramm_score"])
    keyword = item["keyword"]

    prompts = {
        "A": build_prompt_a(sentence),
        "B": build_prompt_b(sentence),
        "C": build_prompt_c(sentence)
    }

    row = {"sentence": sentence, "keyword": keyword, "gramm_score": true}
    for k in "ABC":
        prompt = prompts[k]
        response = call_ollama(prompt)
        label = extract_label(response)

        row[f"prompt_{k}"] = prompt
        row[f"raw_response_{k}"] = response
        row[f"pred_{k}"] = label

    results.append(row)

# === Save all sentence-level results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Raw predictions saved to {OUTPUT_FILE}")

# === Save 10 random input/output examples
sample = df.sample(10, random_state=42)[[
    "sentence", "keyword",
    "prompt_A", "raw_response_A",
    "prompt_B", "raw_response_B",
    "prompt_C", "raw_response_C"
]]
sample.to_csv(SAMPLED_IO_FILE, index=False)
print(f"✓ Saved 10 input/output examples to '{SAMPLED_IO_FILE}'")

# === Keyword-level aggregation
keyword_df = df.groupby("keyword").agg({
    "gramm_score": "first",
    "pred_A": "mean",
    "pred_B": "mean",
    "pred_C": "mean"
}).reset_index()
keyword_df.to_csv(KEYWORD_AGG_FILE, index=False)
print(f"✓ Keyword-level summary saved to {KEYWORD_AGG_FILE}")

# === Final Evaluation
evaluate_predictions(keyword_df, ["pred_A", "pred_B", "pred_C"], EVAL_FILE)
