import json
import requests
import re
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Configuration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama2"
INPUT_FILE = "test_set_150.jsonl"

# === Dynamically name output files ===
TAG = f"{MODEL_NAME}_ZS_COT"
ZS_FILE = f"predictions_ZS_{TAG}.csv"
COT_FILE = f"predictions_COT_{TAG}.csv"
INCORRECT_FILE = f"incorrect_predictions_{TAG}.csv"
SUMMARY_FILE = f"accuracy_summary_{TAG}.csv"
PLOT_FILE = f"accuracy_comparison_{TAG}.png"

# === Prompt Definitions ===
DEFINITIONS = """Level 1: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NPs», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

# === Prompt Constructor ===
def build_prompt(sentence, cot=False):
    prompt = f"""You are a linguist specializing in German syntax and grammaticalization.
Your task is to classify the degree of grammaticalization of the bracketed preposition in the given sentence, using only the sentence as input. Choose one of the following four levels. Use the definitions below exactly.

{DEFINITIONS}

Input sentence:  
{sentence}

Expected format:  
Label: [1–4]  
Reason: [brief explanation based on definition]
"""
    if cot:
        prompt += "\nThink step by step before making a decision."
    return prompt

# === API Call ===
def call_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return ""

# === Label Extractor ===
def extract_label(text):
    if not text:
        return None
    match = re.search(r"Label:\s*\[?([1-4])\]?", text)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([1-4])\b", text)
    if fallback:
        return int(fallback[-1])
    return None

# === Load Data ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

results = []
zs_correct = 0
cot_correct = 0
zs_per_level = {i: [0, 0] for i in range(1, 5)}  # correct, total
cot_per_level = {i: [0, 0] for i in range(1, 5)}  # correct, total

print(f"\nRunning inference on all {len(data)} examples...\n")

for i, item in enumerate(tqdm(data, desc="Processing examples"), 1):
    sentence = item["sentence_raw"]
    true = int(item["gramm_score"])

    # Zero-shot
    zs_prompt = build_prompt(sentence, cot=False)
    zs_response = call_ollama(zs_prompt)
    zs_label = extract_label(zs_response)
    zs_correct += int(zs_label == true)
    zs_per_level[true][1] += 1
    zs_per_level[true][0] += int(zs_label == true)

    # Chain-of-Thought
    cot_prompt = build_prompt(sentence, cot=True)
    cot_response = call_ollama(cot_prompt)
    cot_label = extract_label(cot_response)
    cot_correct += int(cot_label == true)
    cot_per_level[true][1] += 1
    cot_per_level[true][0] += int(cot_label == true)

    results.append({
        "sentence": sentence,
        "keyword": item["keyword"],
        "true_level": true,
        "zero_shot_pred": zs_label,
        "cot_pred": cot_label,
        "zs_response": zs_response,
        "cot_response": cot_response
    })

# === Save Predictions ===
def save_predictions(filename, pred_key):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "keyword", "true_level", pred_key])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "sentence": row["sentence"],
                "keyword": row["keyword"],
                "true_level": row["true_level"],
                pred_key: row[pred_key]
            })

save_predictions(ZS_FILE, "zero_shot_pred")
save_predictions(COT_FILE, "cot_pred")

# === Save Incorrect Predictions ===
with open(INCORRECT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "keyword", "true_level", "zero_shot_pred", "cot_pred"])
    writer.writeheader()
    for row in results:
        if row["true_level"] != row["zero_shot_pred"] or row["true_level"] != row["cot_pred"]:
            writer.writerow({
                "sentence": row["sentence"],
                "keyword": row["keyword"],
                "true_level": row["true_level"],
                "zero_shot_pred": row["zero_shot_pred"],
                "cot_pred": row["cot_pred"]
            })

# === Print Overall Accuracy ===
total = len(results)
zs_acc = zs_correct / total
cot_acc = cot_correct / total
print(f"\n Zero-shot Accuracy: {zs_acc:.2%} ({zs_correct}/{total})")
print(f" CoT Accuracy:       {cot_acc:.2%} ({cot_correct}/{total})")

# === Per-Level Accuracy Summary ===
print("\n Per-level Accuracy:")
with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Level", "ZS Accuracy", "COT Accuracy", "ZS Correct", "COT Correct", "Total"])
    for lvl in range(1, 5):
        zc, zt = zs_per_level[lvl]
        cc, ct = cot_per_level[lvl]
        zacc = zc / zt if zt else 0
        cacc = cc / ct if ct else 0
        print(f"Level {lvl}: ZS={zacc:.2%}, COT={cacc:.2%}")
        writer.writerow([lvl, round(zacc * 100, 2), round(cacc * 100, 2), zc, cc, zt])
    writer.writerow(["Overall", round(zs_acc * 100, 2), round(cot_acc * 100, 2), zs_correct, cot_correct, total])

# === Plot Model Comparison ===
plt.figure(figsize=(6, 4))
bars = plt.bar(["Zero-shot", "Chain-of-Thought"], [zs_acc, cot_acc], color=["#1f77b4", "#ff7f0e"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Grammaticalization Accuracy Comparison")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2%}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.show()
