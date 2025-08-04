import os
import re
import requests
import pandas as pd
from tqdm import tqdm

# === Config ===
LLM_ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "google/gemma-3-12b"
INPUT_FILE = "./data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv"

#please change the description every time you run the code with different prompting or arguments
description = "definitions_and_few_shot_prompting"
OUTPUT_PATH = f"./evaluation/input_csv (only dev2 nothing else)/{description}.csv"

# === Prompt Definitions ===
DEFINITIONS = """Level 1: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of "pure" function words."""

def build_prompt(keyword):
    return f"""You are a highly trained text data annotation tool capable of providing subjective responses.
Your task is to rate the degree of grammaticalization of a bracketed German preposition in the given sentence.

Sentence:
{keyword}

Focus only on the structure and functional reduction of the bracketed preposition as it appears in this sentence.
Please provide a judgment as a single grammaticalization level using the below definition, where:

{DEFINITIONS}

Respond strictly in the format to indicate the level of grammaticalization:

Label: [1] or Label: [2] or Label: [3] or Label: [4]

Expected format:  
Label: [<level>]"""

def get_llm_response(prompt):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 100,
        "stream": False
    }
    try:
        response = requests.post(LLM_ENDPOINT, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return ""
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

# === Load data ===
df = pd.read_csv(INPUT_FILE)

print(f"\nRunning inference on {len(df)} examples...\n")
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    keyword = row["keyword"]
    gramm_score = row["gramm_score"]
    
    prompt = build_prompt(keyword)
    response = get_llm_response(prompt)
    prediction = extract_label(response)
    
    results.append({
        "keyword": keyword,
        "gramm_score": gramm_score,
        "predictions": prediction
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"✓ Predictions saved to {OUTPUT_PATH}")
