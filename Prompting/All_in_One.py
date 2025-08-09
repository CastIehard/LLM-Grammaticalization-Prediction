import os
import json
import re
import itertools
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from prompts import PromptBuilder

# === Load environment variables (if any)
load_dotenv()

# === Config ===
MODEL_NAME = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

# === Input / Output ===
INPUT_FILE = "./data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv"
OUTPUT_DIR = "./evaluation/input_csv (only dev2 nothing else)/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Prompt Variants to Run ===
PROMPT_LABELS = ["A", "B", "C", "D", "E", "F"]  # Update as needed

# === LLM Call Helper
def call_ollama(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return response.json().get("response", "").strip() if response.status_code == 200 else ""
    except Exception as e:
        print(f" Error calling Ollama: {e}")
        return ""

# === Label Extraction
def extract_label(text):
    if not text:
        return None
    match = re.search(r"Label:\s*\[?([1-4])\]?", text)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([1-4])\b", text)
    return int(fallback[-1]) if fallback else 0

# === Evaluation Function
def evaluate_predictions(df, metrics, output_path):
    df.dropna(subset=["gramm_score"], inplace=True)
    df = df.drop_duplicates(subset=["keyword"]).copy()
    if len(df) < 2:
        print(" Not enough keywords for evaluation.")
        return

    results = []
    truth = df["gramm_score"]

    # Spearman's rho
    rho_row = {"Evaluation": "Spearman's ρ (rank)"}
    for m in metrics:
        rho, _ = spearmanr(truth, df[m])
        rho_row[m] = f"{rho:.2f}"
    results.append(rho_row)

    # Average Precision
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

    # Accuracy
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
    print(f" Evaluation summary saved to: {output_path}")

# === Load Input
df_input = pd.read_csv(INPUT_FILE)
df_input = df_input[["keyword", "gramm_score"]].dropna()

data = df_input.to_dict(orient="records")

# === Run Inference Prompt by Prompt
for label in PROMPT_LABELS:
    print(f"\n Running inference with Prompt {label} on {len(data)} keywords using model: {MODEL_NAME}\n")
    results = []

    # === Output file paths for this prompt
    description = f"{MODEL_NAME}_ZS_prompt{label.lower()}_basic"
    output_path = os.path.join(OUTPUT_DIR, f"{description}.csv")
    sampled_io_path = os.path.join(OUTPUT_DIR, f"{description}_samples.csv")
    keyword_agg_path = os.path.join(OUTPUT_DIR, f"{description}_summary.csv")
    eval_path = os.path.join(OUTPUT_DIR, f"{description}_eval.csv")

    # === Inference Loop
    for item in tqdm(data):
        keyword = item["keyword"]
        true_score = float(item["gramm_score"])

        try:
            prompt_func = getattr(PromptBuilder, f"build_prompt_{label.lower()}")
            prompt = prompt_func(keyword)  # note: pass keyword instead of sentence
        except AttributeError:
            print(f" Missing: PromptBuilder.build_prompt_{label.lower()}()")
            continue

        response = call_ollama(prompt)
        pred = extract_label(response)

        results.append({
            "keyword": keyword,
            "gramm_score": true_score,
            f"prompt_{label}": prompt,
            f"raw_response_{label}": response,
            f"pred_{label}": pred
        })

    # === Save Raw Predictions
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f" Full predictions saved to: {output_path}")

    # === Save Sample I/O
    sample_cols = ["keyword", f"prompt_{label}", f"raw_response_{label}"]
    df.sample(min(10, len(df)), random_state=42)[sample_cols].to_csv(sampled_io_path, index=False)
    print(f" Sample I/O saved to: {sampled_io_path}")

    # === Keyword-Level Aggregation
    keyword_df = df.groupby("keyword").agg({
        "gramm_score": "first",
        f"pred_{label}": "mean"
    }).reset_index()
    keyword_df.to_csv(keyword_agg_path, index=False)
    print(f" Keyword-level summary saved to: {keyword_agg_path}")

    # === Evaluation
    evaluate_predictions(keyword_df, [f"pred_{label}"], eval_path)
