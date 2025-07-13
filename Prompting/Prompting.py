import json
import dspy

# === Step 1: Load and process the data ===
with open("sdewac_sampled_output.json", encoding="utf-8") as f:
    raw_data = json.load(f)

examples = []
for entry in raw_data:
    if entry.get("keyword") and entry.get("sentence") and entry.get("pos"):
        keyword = entry["keyword"]
        tokens = entry["sentence"]
        pos_list = entry["pos"]

        tokens_lower = [tok.lower() for tok in tokens]
        if keyword.lower() in tokens_lower:
            idx = tokens_lower.index(keyword.lower())
            pos_tag = pos_list[idx]

            # Optional: highlight keyword in sentence
            highlighted_sentence = " ".join(
                f"[{tok}]" if i == idx else tok
                for i, tok in enumerate(tokens)
            )

            examples.append({
                "word": keyword,
                "sentence": highlighted_sentence,  # or " ".join(tokens) if you don't want brackets
                "pos": pos_tag,
                "index": idx,
                "true_level": int(entry["grammaticalisation_level"])
            })

# === Step 2: Define DSPy signature ===
class Grammaticalization(dspy.Signature):
    """
    Predict the grammaticalization level (1–4) of a target word in a sentence.

    Level 1: fully lexical, no grammatical use.
    Level 2: slight grammatical use but mostly lexical.
    Level 3: mainly grammatical use (e.g. auxiliary, modal).
    Level 4: fully grammaticalized (e.g. function word, clitic).

    Consider the POS tag, the context in the sentence, and the position.
    """

    sentence = dspy.InputField(desc="The full sentence containing the word.")
    word = dspy.InputField(desc="The target word.")
    pos = dspy.InputField(desc="POS tag of the word.")
    index = dspy.InputField(desc="Position of the word in the sentence.")
    level = dspy.OutputField(desc="Grammaticalization level (1–4)")
    reasoning = dspy.OutputField(desc="Explanation of the level")


# === Step 3: Connect to LLM ===
lm = dspy.LM(
    model="ollama_chat/gemma:2b",
    api_base="http://localhost:11434",
    api_key=""
)
dspy.configure(lm=lm)

# === Step 4: Define predictor ===
predictor_model = dspy.ChainOfThought(Grammaticalization)

# Export for use in prediction script
train_examples = examples
