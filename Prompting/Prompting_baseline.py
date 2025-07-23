import json
import requests
import re
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Configuration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama2"
INPUT_FILE = "test_set_40.jsonl"

# === Dynamically name output files ===
TAG = f"{MODEL_NAME}_ZS_ALL"
OUTPUT_FILE = f"predictions_{TAG}.csv"
INCORRECT_FILE = f"incorrect_predictions_{TAG}.csv"
PLOT_FILE = f"accuracy_comparison_{TAG}.png"

# === Fixed prompt definitions ===
DEFINITIONS = """Level 1: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NPs», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

def build_prompt(sentence, cot=False):
    base = f"""You are a linguist specializing in German syntax and grammaticalization.
Your task is to classify the degree of grammaticalization of the bracketed preposition in the given sentence, using only the sentence as input. Choose one of the following four levels. Use the definitions below exactly.

{DEFINITIONS}

Input sentence:  
{sentence}

Expected format:  
Label: [1–4]  
Reason: [brief explanation based on definition]
"""
    if cot:
        base += "\nThink step by step before making a decision."
    return base

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

# === Load entire dataset ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

results = []
zs_correct = 0

print(f"\nRunning inference on all {len(data)} examples...\n")

for i, item in enumerate(tqdm(data, desc="Processing examples"), 1):
    sentence = item["sentence_raw"]
    true = int(item["gramm_score"])

    zs_prompt = build_prompt(sentence, cot=False)
    zs_response = call_ollama(zs_prompt)
    zs_label = extract_label(zs_response)

    zs_correct += int(zs_label == true)

    results.append({
        "sentence": sentence,
        "keyword": item["keyword"],
        "true_level": true,
        "zero_shot_pred": zs_label,
        "zs_response": zs_response,
    })

    if i <= 5:
        print(f"Example {i}:")
        print(f"Sentence: {sentence}")
        print(f"True: {true}")
        print(f"Zero-shot: {zs_label}")
        print()

# === Save predictions ===
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "keyword", "true_level", "zero_shot_pred"])
    writer.writeheader()
    for row in results:
        writer.writerow({
            "sentence": row["sentence"],
            "keyword": row["keyword"],
            "true_level": row["true_level"],
            "zero_shot_pred": row["zero_shot_pred"]
        })

# === Save incorrect predictions ===
with open(INCORRECT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence", "keyword", "true_level", "zero_shot_pred"])
    writer.writeheader()
    for row in results:
        if row["zero_shot_pred"] != row["true_level"]:
            writer.writerow({
                "sentence": row["sentence"],
                "keyword": row["keyword"],
                "true_level": row["true_level"],
                "zero_shot_pred": row["zero_shot_pred"]
            })

# === Evaluation ===
zs_acc = zs_correct / len(results)

print(f"\n Zero-shot accuracy: {zs_acc*100:.2f}% ({zs_correct}/{len(results)})")

# === Plot accuracy ===
strategies = ["Zero-shot"]
accuracies = [zs_acc]

plt.figure(figsize=(6, 4))
bars = plt.bar(strategies, accuracies, color=["#1f77b4"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Grammaticalization Accuracy (All Examples)")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2%}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.show()


##### per level accuracy ######

# === Accuracy per grammaticalization level ===
level_correct = {1: 0, 2: 0, 3: 0, 4: 0}
level_total = {1: 0, 2: 0, 3: 0, 4: 0}

for row in results:
    true = row["true_level"]
    pred = row["zero_shot_pred"]
    level_total[true] += 1
    if pred == true:
        level_correct[true] += 1

# Compute and print level-wise accuracy
print("\n Per-level Accuracy:")
level_accuracies = {}
for level in range(1, 5):
    correct = level_correct[level]
    total = level_total[level]
    acc = correct / total if total > 0 else 0
    level_accuracies[level] = acc
    print(f"Level {level}: {acc:.2%} ({correct}/{total})")

# Save accuracy summary to CSV
with open(f"accuracy_summary_{TAG}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Grammaticalization Level", "Accuracy (%)", "Correct", "Total"])
    for level in range(1, 5):
        writer.writerow([
            level,
            round(level_accuracies[level] * 100, 2),
            level_correct[level],
            level_total[level]
        ])
    writer.writerow(["Overall", round(zs_acc * 100, 2), zs_correct, len(results)])

# === Plot distribution of predicted levels vs true levels ===
import numpy as np

true_counts = [level_total[i] for i in range(1, 5)]
pred_counts = [sum(1 for r in results if r["zero_shot_pred"] == i) for i in range(1, 5)]

levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4']
x = np.arange(len(levels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, true_counts, width, label='True', color='gray')
plt.bar(x + width/2, pred_counts, width, label='Predicted (ZS)', color='#1f77b4')

plt.ylabel('Count')
plt.title('Distribution of Grammaticalization Levels (True vs Predicted)')
plt.xticks(x, levels)
plt.legend()
plt.tight_layout()

dist_plot_file = f"level_distribution_comparison_{TAG}.png"
plt.savefig(dist_plot_file)
plt.show()

print(f"\n Distribution comparison plot saved as: {dist_plot_file}")