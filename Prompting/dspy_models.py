# dspy_models_variants.py — three DSPy variants
# 1) Baseline (no optimization)
# 2) Zero-shot optimized (MIPROv2, instructions only; no demos)
# 3) Few-shot optimized (JOINT MIPROv2: tunes BOTH instructions and few-shot demos)

from typing import List, Dict, Optional, Iterable, Callable, Union
import os
import csv
import re
import threading
import dspy
from dspy.teleprompt import MIPROv2

# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------
BASIC_INSTRUCTION = (
    "You are a highly trained linguist specializing in German grammaticalization and capable of providing subjective responses. "
    "Rate the degree of grammaticalization of the target German word by utilizing the following level description - label between *'s followed by its definition. "
    "Your response should align with a human’s succinct judgment. Please respond in the format: Level : <value>"
)

LABEL_DEFS = """*Level 1*: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

*Level 2*: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

*Level 3*: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

*Level 4*: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

# ---------------------------------------------------------------------------
# Metric and signature(s)
# ---------------------------------------------------------------------------
def accuracy_metric(example, prediction, trace=None):
    try:
        m = re.search(r"(?:[Ll]evel\s*:?\s*)?([1-4])\b", str(prediction))
        yhat = int(m.group(1)) if m else -1
    except Exception:
        yhat = -1
    return 1.0 if yhat == int(example["level"]) else 0.0

class WithDefs(dspy.Signature):
    """{BASIC_INSTRUCTION}"""
    label_definitions: str = dspy.InputField(desc="Text describing label meanings 1–4")
    preposition: str = dspy.InputField(desc="The German preposition")
    level: str = dspy.OutputField(desc="Output in format: Level : <value>")

WithDefs.__doc__ = BASIC_INSTRUCTION

# ---------------------------------------------------------------------------
# Logging hook helpers (for raw LLM I/O)
# ---------------------------------------------------------------------------
class _LMIOLogger:
    def __init__(self, path: str):
        self.path = path
        self._rows = []
        self._lock = threading.Lock()
        self._ctx = threading.local()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["strategy", "keyword", "raw_input", "raw_output"])
                w.writeheader()

    def set_context(self, **kwargs):
        self._ctx.kv = kwargs

    def record(self, raw_input: str, raw_output: str):
        with self._lock:
            kv = getattr(self._ctx, "kv", {})
            row = {
                "strategy": kv.get("strategy", ""),
                "keyword": kv.get("keyword", ""),
                "raw_input": raw_input or "",
                "raw_output": raw_output or "",
            }
            self._rows.append(row)

    def flush(self):
        with self._lock:
            if not self._rows:
                return
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["strategy", "keyword", "raw_input", "raw_output"])
                w.writerows(self._rows)
            self._rows.clear()

class _LoggingLM(dspy.BaseLM):
    def __init__(self, base_lm, logger: _LMIOLogger):
        super().__init__(model=base_lm.model)
        self._base = base_lm
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __call__(self, *args, **kwargs):
        raw_input = None
        if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
            raw_input = kwargs["prompt"]
        elif "messages" in kwargs and isinstance(kwargs["messages"], (list, tuple)):
            parts = []
            for m in kwargs["messages"]:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            raw_input = "\n".join(parts)

        out = self._base(*args, **kwargs)

        raw_output = (
            out if isinstance(out, str)
            else getattr(out, "text", None) or getattr(out, "completion", None) or str(out)
        )

        self._logger.record(raw_input=raw_input, raw_output=raw_output)
        return out

_logger_instance = None
_original_lm = None

def _get_current_lm():
    if hasattr(dspy, "settings") and getattr(dspy.settings, "lm", None) is not None:
        return dspy.settings.lm
    if hasattr(dspy, "config") and getattr(dspy.config, "lm", None) is not None:
        return dspy.config.lm
    return None

def enable_lm_io_logging(log_csv_path: str):
    global _logger_instance, _original_lm
    current_lm = _get_current_lm()
    if current_lm is None:
        raise RuntimeError("No LM is configured. Use dspy.configure(lm=...) before logging.")
    if _logger_instance is None:
        _original_lm = current_lm
        _logger_instance = _LMIOLogger(log_csv_path)
        wrapped = _LoggingLM(_original_lm, _logger_instance)
        dspy.configure(lm=wrapped)
    return _logger_instance

def set_lm_io_context(**kwargs):
    if _logger_instance is not None:
        _logger_instance.set_context(**kwargs)

def flush_lm_io_logs():
    if _logger_instance is not None:
        _logger_instance.flush()

# ---------------------------------------------------------------------------
# Baseline (no optimization)
# ---------------------------------------------------------------------------
class WithDefsBaseline(dspy.Module):
    def __init__(self, defs_text: str = LABEL_DEFS):
        super().__init__()
        self._defs = defs_text
        self._predict = dspy.Predict(WithDefs)

    def forward(self, preposition: str):
        return self._predict(preposition=preposition, label_definitions=self._defs)

# ---------------------------------------------------------------------------
# Zero-shot optimized (instruction-only)
# ---------------------------------------------------------------------------
class WithDefsZeroShotOptimized(dspy.Module):
    def __init__(self, optimized_program, defs_text: str = LABEL_DEFS):
        super().__init__()
        self._prog = optimized_program
        self._defs = defs_text

    def forward(self, preposition: str):
        return self._prog(preposition=preposition, label_definitions=self._defs)

def compile_zero_shot(trainset, defs=LABEL_DEFS, auto="medium", metric=accuracy_metric):
    ds = [
        dspy.Example(preposition=row["preposition"], level=row["level"], label_definitions=defs)
        .with_inputs("preposition", "label_definitions")
        for row in trainset
    ]
    program = dspy.Predict(WithDefs)
    teleprompter = MIPROv2(metric=metric, max_bootstrapped_demos=0, max_labeled_demos=0, auto=auto)
    return WithDefsZeroShotOptimized(teleprompter.compile(program, trainset=ds, requires_permission_to_run=False), defs)

# ---------------------------------------------------------------------------
# Few-shot optimized (joint)
# ---------------------------------------------------------------------------
class WithDefsJointOptimized(dspy.Module):
    def __init__(self, optimized_program, defs_text: str = LABEL_DEFS):
        super().__init__()
        self._prog = optimized_program
        self._defs = defs_text

    def forward(self, preposition: str):
        return self._prog(preposition=preposition, label_definitions=self._defs)

def compile_joint_mipro(trainset, defs=LABEL_DEFS, auto="medium", metric=accuracy_metric,
                         bootstrapped_demos=60, selected_demos_per_module=8):
    ds = [
        dspy.Example(preposition=row["preposition"], level=row["level"], label_definitions=defs)
        .with_inputs("preposition", "label_definitions")
        for row in trainset
    ]
    program = dspy.Predict(WithDefs)
    teleprompter = MIPROv2(metric=metric, auto=auto,
                           max_bootstrapped_demos=bootstrapped_demos,
                           max_labeled_demos=selected_demos_per_module)
    return WithDefsJointOptimized(teleprompter.compile(program, trainset=ds, requires_permission_to_run=False), defs)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_baseline(defs=LABEL_DEFS):
    return WithDefsBaseline(defs_text=defs)

def make_zero_shot_optimized(trainset, defs=LABEL_DEFS, auto="medium"):
    return compile_zero_shot(trainset=trainset, defs=defs, auto=auto)

def make_few_shot_optimized(trainset, demos=None, defs=LABEL_DEFS, auto="medium",
                             bootstrapped_demos=60, selected_demos_per_module=8):
    return compile_joint_mipro(
        trainset=trainset,
        defs=defs,
        auto=auto,
        bootstrapped_demos=bootstrapped_demos,
        selected_demos_per_module=selected_demos_per_module,
    )
