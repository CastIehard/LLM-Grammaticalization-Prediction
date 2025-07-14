import json
import dspy

# === Step 1: Load and process ONLY 63rd entry ===
with open("sdewac_sampled_output.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# Get only the 63rd entry (index 63)
entry = raw_data[63]

examples = []

if entry.get("keyword") and entry.get("sentence") and entry.get("pos"):
    keyword = entry["keyword"].lower()
    tokens = entry["sentence"]
    pos_list = entry["pos"]
    tokens_lower = [tok.lower() for tok in tokens]

    # Find all occurrences of the keyword
    keyword_indices = [i for i, tok in enumerate(tokens_lower) if tok == keyword]

    for idx in keyword_indices:
        pos_tag = pos_list[idx]

        # Highlight this occurrence
        highlighted_tokens = [
            f"[{tok}]" if i == idx else tok
            for i, tok in enumerate(tokens)
        ]
        highlighted_sentence = " ".join(highlighted_tokens)

        examples.append({
            "word": keyword,
            "sentence": highlighted_sentence,
            "pos": pos_tag,
            "index": idx,
            "true_level": int(entry["grammaticalisation_level"])
        })

# === Step 2: Define DSPy signature ===
class Grammaticalization(dspy.Signature):
    """
    Predict the grammaticalization level (1–4) of a target word in a sentence.
    """
    sentence = dspy.InputField()
    word = dspy.InputField()
    pos = dspy.InputField()
    index = dspy.InputField()
    level = dspy.OutputField(desc="Grammaticalization level (1–4)")
    reasoning = dspy.OutputField(desc="Explanation for the predicted level")

# === Step 3: Connect to local LLaMA2 via Ollama (compatible version) ===
lm = dspy.LM(
    model="ollama_chat/llama2",  # Or "llama3", "llama3:instruct" if you pulled that
    api_base="http://localhost:11434",
    api_key=""  # Not required for Ollama
)
dspy.configure(lm=lm)

# === Step 4: Define predictor ===
predictor = dspy.ChainOfThought(Grammaticalization)

# === Step 5: Run prediction for this example ===
for example in examples:
    result = predictor(
        sentence=example["sentence"],
        word=example["word"],
        pos=example["pos"],
        index=str(example["index"])
    )

    print("\n---")
    print(f" Word: {example['word']} at index {example['index']}")
    print(f" Sentence: {example['sentence']}")
    print(f" POS tag: {example['pos']}")
    print(f" Predicted Level: {result.level}")
    print(f" Reasoning: {result.reasoning}")
    print(f"True Label (for reference): {example['true_level']}")
