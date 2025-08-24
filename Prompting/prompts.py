# Updated prompts.py (No JSON format, using updated Prompt A/B/C from your new spec)

import pandas as pd
import random
from typing import List, Tuple, Dict

class PromptBuilder:
    DEFINITIONS = """*Level 1*: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

*Level 2*: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

*Level 3*: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

*Level 4*: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

    @staticmethod
    def _read_and_clean(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        kw_col = next((c for c in ["keyword", "Keyword", "preposition", "Preposition"] if c in df.columns), None)
        lb_col = next((c for c in ["gramm_score", "Gramm_Score", "label", "Label", "level", "Level"] if c in df.columns), None)
        if kw_col is None or lb_col is None:
            raise ValueError("CSV missing keyword or label column.")
        df = df[[kw_col, lb_col]].dropna().copy()
        df.rename(columns={kw_col: "keyword", lb_col: "level"}, inplace=True)
        df["keyword"] = df["keyword"].astype(str)
        df["level"] = df["level"].astype(float).round().astype(int)
        df = df[df["level"].between(1, 4)]
        return df

    @staticmethod
    def make_examples_block_from_csv_interleave(csv_path: str, per_level: int = 2, seed: int = 42, dedup_keywords: bool = True) -> str:
        df = PromptBuilder._read_and_clean(csv_path)
        if dedup_keywords:
            df = df.drop_duplicates(subset=["keyword"])

        rng = random.Random(seed)
        sampled: Dict[int, List[str]] = {}
        for lvl in [1, 2, 3, 4]:
            pool = df[df["level"] == lvl]["keyword"].tolist()
            rng.shuffle(pool)
            sampled[lvl] = pool[:per_level] if len(pool) >= per_level else (pool * ((per_level // len(pool)) + 1))[:per_level]

        examples = []
        for i in range(per_level):
            for lvl in [1, 2, 3, 4]:
                kw = sampled[lvl][i]
                examples.append(f"[User]\nKeyword (target word): {kw}\n[Assistant]\nLevel: {lvl}")

        return "\n".join(examples)

    @staticmethod
    def build_prompt_a(keyword: str) -> str:
        return f"""[SYSTEM]
You are a highly trained linguist specializing in German grammaticalization and capable of providing subjective responses. Rate the degree of grammaticalization of the target German word. Output levels: For very low grammaticalization, assign Level 1, for a medium grammaticalized word, assign Level 2, for high grammaticalization, assign Level 3, for full grammaticalization, assign Level 4. Your response should align with a human’s succinct judgment. Please respond in the format:
Level : <value>

[USER]
Keyword (target word): {keyword}
Please provide a judgment at a single level between (1-4). For example, if your judgment is fully grammaticalized, then provide Level: 4. If your judgment is very low grammaticalization, provide Level: 1.
"""

    @staticmethod
    def build_prompt_b(keyword: str) -> str:
        return f"""[SYSTEM]
You are a highly trained linguist specializing in German grammaticalization and capable of providing subjective responses. Rate the degree of grammaticalization of the target German word by utilizing the following level description - label between *'s followed by its definition:
{PromptBuilder.DEFINITIONS}
Your response should align with a human’s succinct judgment. Please respond in the format:
Level : <value>

[USER]
Keyword (target word): {keyword}
Please provide a judgment at a single level. For example, if your judgment is fully grammaticalized, then provide Level: 4. If your judgment is very low grammaticalization, provide Level: 1.
"""

    @staticmethod
    def build_prompt_c(keyword: str, examples_block: str) -> str:
        return f"""[SYSTEM]
    You are a highly trained linguist specializing in German grammaticalization and capable of providing subjective responses. Rate the degree of grammaticalization of the target German word by utilizing the following level description - label between *'s followed by its definition:
    {PromptBuilder.DEFINITIONS}
    Your response should align with a human’s succinct judgment. Please respond in the format:
    Level : <value>

    [USER]
    Keyword (target word): {keyword}
    ###Examples###
    {examples_block}
    """

