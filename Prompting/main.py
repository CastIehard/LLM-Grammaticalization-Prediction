import os
import re
import time
import logging
import itertools
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
import requests

load_dotenv()

# === Config ===
USE_OPENAI = True
if USE_OPENAI:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL_NAME = "gpt-4o"
else:
    openai_client = None
    #change for ollama or whatever local LLM you are using
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
    MODEL_NAME = os.getenv("LLM_MODEL", "google/gemma‑3‑12b")

INPUT_FILE = "./data/dev data 2 (for testing)/dev_data_2_testing_metrics.csv"


#please change the description every time you run the code with different prompting or arguments
description = "gpt4o_with_definitions_and_structure"
OUTPUT_PATH = f"./evaluation/input_csv (only dev2 nothing else)/{description}.csv"

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

def extract_label(response_text: str) -> int | None:
    if not response_text:
        return None
    m = re.search(r"Label:\s*\[?([1-4])\]?", response_text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b([1-4])\b", response_text)
    return int(nums[-1]) if nums else None

def _call_openai(prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()

def _call_local_llm(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(LLM_ENDPOINT, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# Select the right LLM function
get_llm_response = _call_openai if USE_OPENAI else _call_local_llm

def run_inference(df: pd.DataFrame, concurrency: int = 8) -> pd.DataFrame:
    """
    If df is small (<1000 rows), use serial. For >1000,
    use parallel batches to speed up (optional: use multiprocessing/asyncio).
    Currently: standard loop with tqdm.
    """
    results = []
    with tqdm(total=len(df)) as pbar:
        for sentence, true_score in zip(df["keyword"], df["gramm_score"]):
            prompt = build_prompt(sentence)
            try:
                ai_out = get_llm_response(prompt)
            except Exception as e:
                logging.error(f"LLM call failed for: {sentence!r} — {e}")
                label = None
                ai_out = ""
            else:
                label = extract_label(ai_out)
            results.append({"keyword": sentence, "gramm_score": true_score, "predictions": label})
            pbar.update(1)
    return pd.DataFrame(results)

def main():
    df = pd.read_csv(INPUT_FILE, dtype=str)
    logging.info(f"Loaded {len(df)} examples")
    out_df = run_inference(df)
    out_df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved inference results to {OUTPUT_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
