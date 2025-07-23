import json
import requests
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# === Configuration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama2"
INPUT_FILE = "test_set_40.jsonl"  # Ensure this is in the same folder

# === Output file tagging ===
TAG = f"{MODEL_NAME}_ZS_A_vs_B"
OUTPUT_FILE = f"predictions_{TAG}.csv"
INCORRECT_FILE = f"incorrect_predictions_{TAG}.csv"
PLOT_FILE = f"accuracy_comparison_{TAG}.png"
DIST_PLOT_FILE = f"level_distribution_comparison_{TAG}.png"
ACC_SUMMARY_FILE = f"accuracy_summary_{TAG}.csv"

# === Prompt A: Original Minimal ===
def build_prompt_a(sentence):
    return f"""You will be given a sentence containing a bracketed German preposition.  
Your task is to classify how grammaticalized that preposition is, based on how fixed or reduced its structure appears.  
Choose one of four levels: from very low (structured and article-based) to fully grammaticalized (function-like and eroded). .  
Respond with: Label: [1–4].

Input sentence:  
{sentence}
"""

# === Prompt B: Customized Prompt 1 Style ===
def build_prompt_b(sentence):
    return f"""You are a highly trained text data annotation tool capable of providing subjective responses.
Your task is to rate the degree of grammaticalization of a bracketed German preposition in the given sentence.

Sentence:
{sentence}

Focus only on the structure and functional reduction of the bracketed preposition as it appears in this sentence.
Please provide a judgment as a single integer, where:

1 = Very Low (structured and article-based)  
2 = Low (partially fixed or still compositional)  
3 = High (reduced, little compositionality)  
4 = Very High (fully grammaticalized, function-like and eroded)

Respond with:  
Label: [1-4]
"""

# === LLM call ===
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

# === Extract Label ===
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

# === Evaluation Loop ===
results = []
a_correct = 0
b_correct = 0
level_correct_a = {i: 0 for i in range(1, 5)}
level_correct_b = {i: 0 for i in range(1, 5)}
level_total = {i: 0 for i in range(1, 5)}

print(f"\n Running inference on {len(data)} examples using both prompts...\n")

for i, item in enumerate(tqdm(data, desc="Processing"), 1):
    sentence = item["sentence_raw"]
    true = int(item["gramm_score"])

    # Prompt A
    prompt_a = build_prompt_a(sentence)
    response_a = call_ollama(prompt_a)
    label_a = extract_label(response_a)

    # Prompt B
    prompt_b = build_prompt_b(sentence)
    response_b = call_ollama(prompt_b)
    label_b = extract_label(response_b)

    a_correct += int(label_a == true)
    b_correct += int(label_b == true)
    level_total[true] += 1
    if label_a == true:
        level_correct_a[true] += 1
    if label_b == true:
        level_correct_b[true] += 1

    results.append({
        "sentence": sentence,
        "keyword": item["keyword"],
        "true_level": true,
        "label_a": label_a,
        "label_b": label_b,
        "response_a": response_a,
        "response_b": response_b
    })

# === Save predictions ===
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "keyword", "true_level", "label_a", "label_b"])
    writer.writeheader()
    for row in results:
        writer.writerow({k: row[k] for k in writer.fieldnames})

with open(INCORRECT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "keyword", "true_level", "label_a", "label_b"])
    writer.writeheader()
    for row in results:
        if row["label_a"] != row["true_level"] or row["label_b"] != row["true_level"]:
            writer.writerow({k: row[k] for k in writer.fieldnames})

# === Accuracy ===
acc_a = a_correct / len(results)
acc_b = b_correct / len(results)

print(f"\n Prompt A Accuracy: {acc_a:.2%}")
print(f" Prompt B Accuracy: {acc_b:.2%}")

# === Per-level Accuracy ===
level_acc_a = {lvl: (level_correct_a[lvl] / level_total[lvl]) if level_total[lvl] else 0 for lvl in range(1, 5)}
level_acc_b = {lvl: (level_correct_b[lvl] / level_total[lvl]) if level_total[lvl] else 0 for lvl in range(1, 5)}

with open(ACC_SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Level", "Accuracy A (%)", "Accuracy B (%)", "Total"])
    for lvl in range(1, 5):
        writer.writerow([
            lvl,
            round(level_acc_a[lvl] * 100, 2),
            round(level_acc_b[lvl] * 100, 2),
            level_total[lvl]
        ])
    writer.writerow(["Overall", round(acc_a * 100, 2), round(acc_b * 100, 2), len(results)])

# === Plot Overall Accuracy ===
plt.figure(figsize=(6, 4))
bars = plt.bar(["Prompt A", "Prompt B"], [acc_a, acc_b], color=["#1f77b4", "#ff7f0e"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Grammaticalization Accuracy: Prompt A vs Prompt B")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2%}", ha='center')
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.close()

# === Plot Label Distribution ===
true_counts = [level_total[i] for i in range(1, 5)]
pred_a = [sum(1 for r in results if r["label_a"] == i) for i in range(1, 5)]
pred_b = [sum(1 for r in results if r["label_b"] == i) for i in range(1, 5)]

x = np.arange(4)
width = 0.25
plt.figure(figsize=(8, 5))
plt.bar(x - width, true_counts, width, label='True', color='gray')
plt.bar(x, pred_a, width, label='Prompt A', color='#1f77b4')
plt.bar(x + width, pred_b, width, label='Prompt B', color='#ff7f0e')
plt.xticks(x, ['Level 1', 'Level 2', 'Level 3', 'Level 4'])
plt.ylabel('Count')
plt.title('Label Distribution: True vs Prompt A vs Prompt B')
plt.legend()
plt.tight_layout()
plt.savefig(DIST_PLOT_FILE)
plt.close()
