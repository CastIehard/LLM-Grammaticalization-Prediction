import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from Prompting.Prompting import predictor_model, train_examples

def extract_level(text):
    if not text:
        return "?"
    match = re.search(r"\b([1-4])\b", str(text))
    return match.group(1) if match else "?"

results = []

print("🔍 Starting Prediction:")
for i, ex in tqdm(enumerate(train_examples[:50]), total=50, desc="Running Predict"):
    word = ex["word"]
    sentence = ex["sentence"]
    pos = ex["pos"]
    index = ex["index"]
    gold = ex["true_level"]

    print(f"\n[#{i+1}] Word: '{word}' | POS: {pos} | Index: {index}")
    print(f"Sentence: {sentence}")

    try:
        output = predictor_model(sentence=sentence, word=word, pos=pos, index=index)
    except Exception as e:
        print(f" Error: {e}")
        output = None

    level_text = output.level if output and hasattr(output, "level") else "?"
    reasoning = output.reasoning if output and hasattr(output, "reasoning") else "No reasoning"
    pred_level = extract_level(level_text)

    print(f" Predicted Level: {level_text}")
    print(f" Reasoning:\n{reasoning}\n{'-'*80}")
    
    results.append({
        "word": word,
        "pos": pos,
        "index": index,
        "sentence": sentence,
        "true_level": gold,
        "predicted_level": pred_level,
        "reasoning": reasoning
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("grammaticalization_predictions.csv", index=False)

# Evaluate
valid = df[df["predicted_level"] != "?"].copy()
valid["predicted_level"] = valid["predicted_level"].astype(int)
valid["true_level"] = valid["true_level"].astype(int)

print("\n Classification Report:")
print(classification_report(valid["true_level"], valid["predicted_level"]))
