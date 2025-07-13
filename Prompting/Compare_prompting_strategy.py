import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import dspy
from Prompting.Prompting import train_examples, Grammaticalization

# === 1. Extract predicted level ===
def extract_level(text):
    if not text:
        return "?"
    if isinstance(text, list):
        text = text[0] if text else ""
    match = re.search(r"\\b([1-4])\\b", str(text))
    return match.group(1) if match else "?"

# === 2. Manual FewShot or CoT with prompt formatting ===
class FewShotManual(dspy.Module):
    def __init__(self, examples, lm, use_chain_of_thought=False):
        super().__init__()
        self.examples = examples
        self.lm = lm
        self.use_cot = use_chain_of_thought

    def forward(self, sentence, word, pos, index):
        # Build few-shot prompt with clear instruction
        shots = (
            "You are a linguistic model that predicts grammaticalization levels (1–4) of words in context.\n"
            "Levels:\n1 = purely lexical\n2 = partially grammatical\n"
            "3 = mostly grammatical\n4 = fully grammaticalized (e.g. auxiliaries, modals, determiners)\n\n"
            "Here are some examples:\n\n"
        )

        for ex in self.examples:
            shots += (
                f"Sentence: {ex['sentence']}\n"
                f"Word: {ex['word']}\n"
                f"POS: {ex['pos']}\n"
                f"Index: {ex['index']}\n"
                f"Level: {ex['true_level']}\n"
            )
            if self.use_cot:
                shots += f"Reasoning: {ex.get('reasoning', '...')}\n"
            shots += "\n"

        # Test input
        prompt = shots
        prompt += (
            f"Sentence: {sentence}\n"
            f"Word: {word}\n"
            f"POS: {pos}\n"
            f"Index: {index}\n"
            f"Level:"
        )
        if self.use_cot:
            prompt += "\nReasoning:"
        prompt += "\nPlease answer with only the level number (1, 2, 3, or 4)."

        try:
            raw = self.lm(prompt)
            response = raw[0] if isinstance(raw, list) else raw
            return dspy.Prediction(level=response.strip(), reasoning=response)
        except Exception as e:
            print(f" Error in prompt call: {e}")
            return dspy.Prediction(level="?", reasoning="error")

# === 3. Build model depending on strategy ===
def make_model(kind="zero-shot", few_shot_examples=None, lm=None):
    if kind == "zero-shot":
        return dspy.Predict(Grammaticalization)
    elif kind == "cot":
        return dspy.ChainOfThought(Grammaticalization)
    elif kind == "few-shot":
        return FewShotManual(few_shot_examples, lm, use_chain_of_thought=False)
    elif kind == "cot-few":
        return FewShotManual(few_shot_examples, lm, use_chain_of_thought=True)

# === 4. Run prediction and compute accuracy ===
def run_predictions(model, examples, max_samples=50, label=""):
    results = []

    print(f"\n🔍 Running: {label}")
    for i, ex in tqdm(enumerate(examples[:max_samples]), total=max_samples):
        word = ex["word"]
        sentence = ex["sentence"]
        pos = ex["pos"]
        index = ex["index"]
        gold = ex["true_level"]

        try:
            output = model(sentence=sentence, word=word, pos=pos, index=index)
            level_text = output.level if hasattr(output, "level") else "?"
        except Exception as e:
            print(f" {label} failed at #{i}: {e}")
            level_text = "?"

        pred = extract_level(level_text)
        results.append({"true": gold, "pred": pred})

    df = pd.DataFrame(results)
    df = df[df["pred"] != "?"]
    df["true"] = df["true"].astype(int)
    df["pred"] = df["pred"].astype(int)

    acc = accuracy_score(df["true"], df["pred"])
    print(f" {label} Accuracy: {acc:.3f}")
    return acc

# === 5. Plot accuracy bar chart ===
def plot_accuracies(acc_dict):
    plt.figure(figsize=(8, 5))
    methods = list(acc_dict.keys())
    accuracies = [acc_dict[m] for m in methods]

    plt.bar(methods, accuracies, color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of Prompting Methods")
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("prompting_accuracy_comparison.png")
    plt.show()

# === 6. Main Execution ===
if __name__ == "__main__":
    # Connect to Ollama (Gemma)
    lm = dspy.LM(
        model="ollama_chat/gemma:2b",
        api_base="http://localhost:11434",
        api_key="",
        return_chat_completions=False
    )
    dspy.configure(lm=lm)

    few_shot_examples = train_examples[:5]

    models = {
        "Zero-Shot": make_model(kind="zero-shot"),
        "Few-Shot": make_model(kind="few-shot", few_shot_examples=few_shot_examples, lm=lm),
        "CoT-Zero": make_model(kind="cot"),
        "CoT + Few": make_model(kind="cot-few", few_shot_examples=few_shot_examples, lm=lm),
    }

    accuracies = {}
    for label, model in models.items():
        acc = run_predictions(model, train_examples, max_samples=50, label=label)
        accuracies[label] = acc

    plot_accuracies(accuracies)