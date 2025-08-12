# dspy_frameworks.py
# Four ZERO-SHOT variants:
#  1) Zero-shot WITHOUT definitions (no optimizer)
#  2) Zero-shot WITH definitions (no optimizer)  [dynamic or constant]
#  3) Zero-shot WITHOUT definitions + instruction optimization (0-Shot MIPRO)
#  4) Zero-shot WITH definitions + instruction optimization (0-Shot MIPRO)
#
# Usage examples are at the bottom.

from typing import List, Dict, Optional, Iterable
import re
import dspy
from dspy.teleprompt import MIPROv2

# -------------------------
# Metric for DSPy optimizers
# -------------------------
def accuracy_metric(example, prediction, trace=None):
    """Return 1.0 if predicted integer == gold integer in {1..4}, else 0.0."""
    try:
        yhat = int(prediction)
    except Exception:
        m = re.search(r"\b([1-4])\b", str(prediction))
        yhat = int(m.group(1)) if m else -1
    return 1.0 if yhat == int(example["label"]) else 0.0

# -------------------------
# Optional: canonical label definitions (you can replace text or pass your own)
# -------------------------
LABEL_DEFS = """Level 1: Very Low Degree of Grammaticalization
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words.
"""

# -------------------------
# Signatures
# -------------------------
class ZS_NoDefs(dspy.Signature):
    """Evaluate the given German preposition and output a label 1–4."""
    preposition: str = dspy.InputField(desc="German preposition (keyword)")
    label: int = dspy.OutputField(desc="One of {1,2,3,4} (grammaticalization level)")

class ZS_WithDefs(dspy.Signature):
    """Use the provided label definitions to rate grammaticalization (1–4)."""
    label_definitions: str = dspy.InputField(desc="Natural-language definitions for labels 1–4")
    preposition: str = dspy.InputField(desc="German preposition (keyword)")
    label: int = dspy.OutputField(desc="One of {1,2,3,4}")

# -------------------------
# 1) Zero-shot WITHOUT definitions (no optimizer)
# -------------------------
def make_zs_nodefs():
    """Return a no-tuning, zero-shot DSPy model without label definitions."""
    return dspy.Predict(ZS_NoDefs)

# -------------------------
# 2) Zero-shot WITH definitions (no optimizer)
#    - dynamic: pass defs at call-time
#    - constant: freeze defs once and avoid passing them every call
# -------------------------
def make_zs_withdefs_dynamic():
    """Return a zero-shot DSPy model that expects label_definitions each call."""
    return dspy.Predict(ZS_WithDefs)

def make_zs_withdefs_const(defs: str = LABEL_DEFS):
    """Return a zero-shot DSPy module with fixed label_definitions."""
    class WithConstDefs(dspy.Module):
        def __init__(self, defs_text: str):
            super().__init__()
            self.predict = dspy.Predict(ZS_WithDefs)
            self._defs = defs_text

        def forward(self, preposition: str):
            return self.predict(preposition=preposition, label_definitions=self._defs)

    return WithConstDefs(defs)

# -------------------------
# Helpers for optimization
# -------------------------
def _ensure_examples(
    trainset: Iterable[Dict],
    use_defs: bool,
    defs: Optional[str]
) -> List[dspy.Example]:
    """
    Normalize a user-provided trainset (list of dicts or dspy.Example) to List[dspy.Example].
    Each dict must have keys: 'preposition' (str) and 'label' (int 1..4).
    If use_defs is True, inject 'label_definitions' into examples.
    """
    examples: List[dspy.Example] = []

    for row in trainset:
        if isinstance(row, dspy.Example):
            ex = row
        else:
            kw = str(row["preposition"])
            lb = int(row["label"])
            if use_defs:
                ex = dspy.Example(preposition=kw, label=lb, label_definitions=(defs or LABEL_DEFS))
                ex = ex.with_inputs("preposition", "label_definitions")
            else:
                ex = dspy.Example(preposition=kw, label=lb).with_inputs("preposition")
        examples.append(ex)

    # If caller passed dspy.Example objects but forgot to mark inputs, fix them:
    fixed: List[dspy.Example] = []
    for ex in examples:
        # Heuristic: ensure correct input fields are set
        if use_defs:
            fixed.append(ex.with_inputs("preposition", "label_definitions"))
        else:
            fixed.append(ex.with_inputs("preposition"))
    return fixed

def _compile_mipro(
    program: dspy.Module,
    trainset: List[dspy.Example],
    auto: str = "medium"
):
    """Compile with MIPROv2 (0-Shot instruction optimization). Fallback to dspy.compile if needed."""
    teleprompter = MIPROv2(
        metric=accuracy_metric,
        max_bootstrapped_demos=0,   # keep it zero-shot: instruction-only
        max_labeled_demos=0,
        auto=auto,
    )
    try:
        return teleprompter.compile(
            program,
            trainset=trainset,
            requires_permission_to_run=False
        )
    except Exception as e:
        compile_fn = getattr(dspy, "compile", None)
        if compile_fn is not None:
            return compile_fn(
                program=program,
                trainset=trainset,
                metric=accuracy_metric,
                teleprompter=teleprompter,
            )
        print(f"[DSPy] Instruction optimization unavailable ({e}). Returning None.")
        return None

# -------------------------
# 3) Zero-shot + instruction optimization (0-Shot MIPRO)
#    a) WITHOUT definitions
#    b) WITH definitions
# -------------------------
def compile_zero_shot_instruction_optimized(
    trainset: Iterable[Dict],
    use_defs: bool = True,
    defs: Optional[str] = None,
    auto: str = "medium",
):
    """
    Returns a compiled DSPy model whose *instruction* has been optimized on `trainset`
    (instruction-only; no demonstrations in prompt).

    Args:
        trainset: iterable of dicts or dspy.Example with fields:
                  - preposition (str)
                  - label (int 1..4)
        use_defs: if True, optimize the signature that includes label_definitions.
        defs:     optional definitions text; if None and use_defs=True, uses LABEL_DEFS.
        auto:     MIPRO auto setting ('conservative' | 'medium' | 'aggressive').

    Returns:
        A compiled DSPy Module (optimized program) or None if compilation is unavailable.
    """
    # Build the appropriate zero-shot program
    program = dspy.Predict(ZS_WithDefs) if use_defs else dspy.Predict(ZS_NoDefs)

    # Normalize/prepare examples and compile
    examples = _ensure_examples(trainset, use_defs=use_defs, defs=defs)
    optimized_program = _compile_mipro(program, trainset=examples, auto=auto)
    return optimized_program

# -------------------------
# Usage examples (remove if you don't want inline docs)
# -------------------------
"""
# 0) Configure DSPy elsewhere in your codebase:
# import dspy
# dspy.configure(lm=dspy.LM('ollama/llama2:7b', api_base='http://localhost:11434'))

# 1) Zero-shot, no definitions:
zs_nodefs = make_zs_nodefs()
pred = zs_nodefs(preposition="über").label

# 2a) Zero-shot, with definitions (dynamic):
zs_withdefs_dyn = make_zs_withdefs_dynamic()
pred = zs_withdefs_dyn(preposition="über", label_definitions=LABEL_DEFS).label

# 2b) Zero-shot, with definitions (constant):
zs_withdefs_const = make_zs_withdefs_const(LABEL_DEFS)
pred = zs_withdefs_const(preposition="über").label

# 3a) Optimized, no definitions:
trainset = [
    {"preposition": "über", "label": 3},
    {"preposition": "bis", "label": 1},
    # ...
]
opt_nodefs = compile_zero_shot_instruction_optimized(trainset, use_defs=False, auto="medium")
if opt_nodefs:
    pred = opt_nodefs(preposition="über").label

# 3b) Optimized, with definitions:
opt_withdefs = compile_zero_shot_instruction_optimized(trainset, use_defs=True, defs=LABEL_DEFS, auto="medium")
if opt_withdefs:
    pred = opt_withdefs(preposition="über", label_definitions=LABEL_DEFS).label
"""
