# prompts.py
import pandas as pd
import random
from typing import List, Tuple, Dict

class PromptBuilder:
    DEFINITIONS = """Level 1: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

    # ======================
    # DYNAMIC EXAMPLES (INTERLEAVE)
    # ======================
    @staticmethod
    def _read_and_clean(csv_path: str) -> pd.DataFrame:
        """
        Read CSV and return DataFrame with columns:
          - keyword: str
          - level: int in {1,2,3,4}
        Accepts common column variants.
        """
        df = pd.read_csv(csv_path)
        kw_col = next((c for c in ["keyword", "Keyword", "preposition", "Preposition"] if c in df.columns), None)
        lb_col = next((c for c in ["gramm_score", "Gramm_Score", "label", "Label", "level", "Level"] if c in df.columns), None)
        if kw_col is None or lb_col is None:
            raise ValueError(
                f"Expected keyword and label columns. Found {list(df.columns)}. "
                f"Looking for one of keyword/preposition and gramm_score/label/level."
            )
        df = df[[kw_col, lb_col]].dropna().copy()
        df.rename(columns={kw_col: "keyword", lb_col: "level"}, inplace=True)
        df["keyword"] = df["keyword"].astype(str)
        df["level"] = df["level"].astype(float).round().astype(int)
        df = df[df["level"].between(1, 4)]
        return df

    @staticmethod
    def make_examples_block_from_csv_interleave(
        csv_path: str,
        per_level: int = 2,
        seed: int = 42,
        dedup_keywords: bool = True,
    ) -> str:
        """
        Build few-shot examples with INTERLEAVED ordering:
          1a,2a,3a,4a, 1b,2b,3b,4b  (generalizes if per_level > 2 via round-robin)
        Returns a string like:
          Preposition: während → Level: 2
          Preposition: von → Level: 4
          ...
        """
        df = PromptBuilder._read_and_clean(csv_path)
        if dedup_keywords:
            df = df.drop_duplicates(subset=["keyword"])

        rng = random.Random(seed)
        sampled: Dict[int, List[str]] = {}
        for lvl in [1, 2, 3, 4]:
            pool = df[df["level"] == lvl]["keyword"].tolist()
            if len(pool) == 0:
                raise ValueError(f"No examples found for level {lvl} in {csv_path}.")
            rng.shuffle(pool)
            take = min(per_level, len(pool))
            sampled[lvl] = pool[:take]
            if len(sampled[lvl]) < per_level:
                # pad if necessary by cycling (rare; just to be safe)
                needed = per_level - len(sampled[lvl])
                sampled[lvl].extend((pool * ((needed // max(1, len(pool))) + 1))[:needed])

        # interleave: round-robin over levels
        seq: List[Tuple[str, int]] = []
        for i in range(per_level):
            for lvl in [1, 2, 3, 4]:
                seq.append((sampled[lvl][i], lvl))

        lines = [f"Preposition: {kw} → Level: {lvl}" for kw, lvl in seq]
        return "\n".join(lines)

    # ======================
    # PROMPT BUILDERS
    # ======================
    @staticmethod
    def build_prompt_a(Keyword: str):
        return f"""You have been given a keyword in the form of preposition. 
Your task is to evaluate the given German preposition and rate the degree of grammaticalization 
in the form of single label from 1 to 4.
Preposition: {Keyword}

"""
    @staticmethod
    def build_prompt_b(Keyword: str):
        return f"""
Task: evaluate the given German preposition and rate the degree of grammaticalization.
Input: Preposition: {Keyword}
Expected Output: Label: <1/2/3/4>
"""

    @staticmethod
    def build_prompt_c(Keyword: str):
        return f"""You are a highly trained linguistic tool specializing in German syntax and grammaticalization.  
Your task is to rate the degree of grammaticalization of the preposition given as the keyword input.

Preposition: {Keyword}

Please provide the judgment as a single integer (1, 2, 3, or 4) corresponding to the increasing degrees of grammaticalization."""



    @staticmethod
    def build_prompt_d(Keyword: str):
        return f"""
Task: evaluate the given German preposition and rate the degree of grammaticalization.
Label Description:{PromptBuilder.DEFINITIONS}
Input: Preposition: {Keyword}
Expected Output: Label: <1/2/3/4>
"""

    @staticmethod
    def build_prompt_e(keyword: str, examples_block: str):
        # unchanged signature & format
        return f"""
Task: evaluate the given German preposition and rate the degree of grammaticalization.
{examples_block}
Preposition: {keyword} → Level:"""

    @staticmethod
    def build_prompt_f(keyword: str, examples_block: str):
        # unchanged signature & format
        return f"""Task: evaluate the given German preposition and rate the degree of grammaticalization.
Label Description:{PromptBuilder.DEFINITIONS}
{examples_block}

Preposition: {keyword} → Level:"""
