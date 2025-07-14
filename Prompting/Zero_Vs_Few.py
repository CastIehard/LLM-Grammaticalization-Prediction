import json
import dspy

# ========== Signature ==========
class Grammaticalization(dspy.Signature):
    sentence = dspy.InputField()
    word = dspy.InputField()
    pos = dspy.InputField()
    index = dspy.InputField()
    level = dspy.OutputField(desc="Grammaticalization level (1–4)")

# ========== Prompt Logger ==========
last_prompt = None

class LoggingLM(dspy.LM):
    def __call__(self, *args, **kwargs):
        global last_prompt
        last_prompt = kwargs.get("prompt") or kwargs.get("messages")
        return super().__call__(*args, **kwargs)

# ========== Load data ==========
with open("sdewac_sampled_output.json", encoding="utf-8") as f:
    raw_data = json.load(f)

def extract_examples(data):
    examples = []
    for entry in data:
        if entry.get("keyword") and entry.get("sentence") and entry.get("pos"):
            keyword = entry["keyword"].lower()
            tokens = entry["sentence"]
            pos_list = entry["pos"]
            tokens_lower = [tok.lower() for tok in tokens]

            keyword_indices = [i for i, tok in enumerate(tokens_lower) if tok == keyword]
            for idx in keyword_indices:
                pos_tag = pos_list[idx]
                highlighted_tokens = [
                    f"[{tok}]" if i == idx else tok for i, tok in enumerate(tokens)
                ]
                highlighted_sentence = " ".join(highlighted_tokens)

                examples.append({
                    "word": keyword,
                    "sentence": highlighted_sentence,
                    "pos": pos_tag,
                    "index": idx,
                    "true_level": str(entry["grammaticalisation_level"])
                })
    return examples

all_examples = extract_examples(raw_data)

# ========== Prepare few-shot demos and test case ==========
few_shot_demos = [
    {
        "sentence": ex["sentence"],
        "word": ex["word"],
        "pos": ex["pos"],
        "index": str(ex["index"]),
        "level": ex["true_level"]
    }
    for ex in all_examples[:5]  # use 5 for now to keep output manageable
]

test_example = all_examples[5] if len(all_examples) > 5 else None

# ========== Connecting to the LM ==========
# lm = dspy.LM(
#     model="ollama_chat/llama2",  # Or "llama3", "llama3:instruct" if you pulled that
#     api_base="http://localhost:11434",
#     api_key=""  # Not required for Ollama
# )
# dspy.configure(lm=lm)

# ========== Connecting to logging LM ==========
lm = LoggingLM(
    model="ollama_chat/llama2",
    api_base="http://localhost:11434",
    api_key=""
)
dspy.configure(lm=lm)

# ========== Predictors ==========
few_shot_predictor = dspy.Predict(Grammaticalization, demos=few_shot_demos)
zero_shot_predictor = dspy.Predict(Grammaticalization)

# ========== Few-shot prediction ==========
if test_example:
    print("\n==== Few-Shot Prediction ====")
    result = few_shot_predictor(
        sentence=test_example["sentence"],
        word=test_example["word"],
        pos=test_example["pos"],
        index=str(test_example["index"])
    )
    print(f"Word: {test_example['word']}, True Level: {test_example['true_level']}")
    print(f"Predicted Level: {result.level}")
    print("\n[Prompt Used by DSPy]")
    print(last_prompt)

# ========== Zero-shot prediction ==========
print("\n==== Zero-Shot Prediction ====")
for example in all_examples[:2]:
    result = zero_shot_predictor(
        sentence=example["sentence"],
        word=example["word"],
        pos=example["pos"],
        index=str(example["index"])
    )
    print(f"\n---")
    print(f"Word: {example['word']}, True Level: {example['true_level']}")
    print(f"Sentence: {example['sentence']}")
    print(f"Predicted Level: {result.level}")

print("\n[Prompt Used for Last Zero-Shot Prediction]")
print(last_prompt)
