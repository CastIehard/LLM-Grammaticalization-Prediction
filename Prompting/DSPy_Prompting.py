import json
import dspy
import matplotlib.pyplot as plt

# === DSPy Signature ===
class GrammaticalizationWithMeta(dspy.Signature):
    sentence = dspy.InputField(desc="Full sentence with the target word bracketed (e.g., 'Ich komme [aus] Berlin.')")
    word = dspy.InputField(desc="Target keyword to evaluate")
    index = dspy.InputField(desc="Position of the keyword in the tokenized sentence")
    collocation_strength = dspy.InputField(desc="Numerical value; higher = more fixed expression")
    context_entropy = dspy.InputField(desc="Numerical value; lower = more grammaticalized")
    level = dspy.OutputField(desc="Predicted grammaticalization level (1–4)")
    reasoning = dspy.OutputField(desc="Explanation of the prediction")

# === Logging LM Wrapper ===
last_prompt = None
class LoggingLM(dspy.LM):
    def __call__(self, *args, **kwargs):
        global last_prompt
        last_prompt = kwargs.get("prompt") or kwargs.get("messages")
        return super().__call__(*args, **kwargs)

# === Helpers to Load and Format JSONL ===
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def prepare_data(raw_data):
    formatted = []
    for entry in raw_data:
        try:
            sentence = entry["sentence_raw"]
            keyword = entry["keyword"]
            index = sentence.split().index(f"[{keyword}]")  # Assumes word is bracketed
            formatted.append({
                "sentence": sentence,
                "word": keyword,
                "index": str(index),
                "collocation_strength": entry["collocation_strength"],
                "context_entropy": entry["context_entropy"],
                "true_level": str(entry["gramm_score"])
            })
        except Exception:
            continue
    return formatted

# === Load the Dev and Test Sets ===
raw_dev = load_jsonl("dataset_dev.jsonl")
raw_test = load_jsonl("data_test.jsonl")

few_shot_demos = prepare_data(raw_dev)
few_shot_tests = prepare_data(raw_test)

# === Connect DSPy to Your Local LM ===
lm = LoggingLM(model="ollama_chat/llama2", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

# === Define Predictors ===
few_shot_predictor = dspy.Predict(GrammaticalizationWithMeta, demos=few_shot_demos)
zero_shot_predictor = dspy.Predict(GrammaticalizationWithMeta)
cot_predictor = dspy.ChainOfThought(GrammaticalizationWithMeta)

# === Evaluation ===
fs_correct = zs_correct = cot_correct = 0
fs_total = zs_total = cot_total = len(few_shot_tests)

for ex in few_shot_tests:
    true = int(ex["true_level"])

    # Few-Shot
    result_fs = few_shot_predictor(
        sentence=ex["sentence"],
        word=ex["word"],
        index=ex["index"],
        collocation_strength=ex["collocation_strength"],
        context_entropy=ex["context_entropy"]
    )
    fs_correct += (int(result_fs.level) == true)

    # Zero-Shot
    result_zs = zero_shot_predictor(
        sentence=ex["sentence"],
        word=ex["word"],
        index=ex["index"],
        collocation_strength=ex["collocation_strength"],
        context_entropy=ex["context_entropy"]
    )
    zs_correct += (int(result_zs.level) == true)

    # Chain-of-Thought
    result_cot = cot_predictor(
        sentence=ex["sentence"],
        word=ex["word"],
        index=ex["index"],
        collocation_strength=ex["collocation_strength"],
        context_entropy=ex["context_entropy"]
    )
    cot_correct += (int(result_cot.level) == true)

# === Accuracy Calculation ===
fs_acc = fs_correct / fs_total
zs_acc = zs_correct / zs_total
cot_acc = cot_correct / cot_total

# === Plot Results ===
strategies = ["Few-Shot", "Zero-Shot", "Chain-of-Thought"]
accuracies = [fs_acc, zs_acc, cot_acc]

plt.figure(figsize=(8, 5))
bars = plt.bar(strategies, accuracies, color=['skyblue', 'salmon', 'limegreen'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Evaluation Accuracy by Prompting Strategy")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.2%}", ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
