import pandas as pd
import random

class PromptBuilder:
    DEFINITIONS = """Level 1: Very Low Degree of Grammaticalization  
Prepositional phrases at this level have a very low degree of grammaticalization and always consist of three words. Either the structure is «preposition + NP + preposition». A second possible structure is «preposition + NP», in which an article always appears.

Level 2: Slightly Higher Degree of Grammaticalization  
Formations at this level show a slightly higher degree of grammaticalization, as they can occur both with and without an article. In some cases, the use with an article still outweighs the use without one, while in other formations, the share of occurrences without an autonomous article already dominates.

Level 3: Strengthened Grammaticalization  
Level 3 describes a state in which grammaticalization is already well advanced. The central feature of this stage is that the expressions (prepositional phrases) in the corpus examined were not assigned an attributive extension. This means that a descriptive adjective can no longer be inserted between the components of the phrase.

Level 4: Highest Degree of Grammaticalization  
Prepositions with the form of a function word have the highest degree of grammaticalization to be recorded, and the grammaticalization process is for the most part complete. Due to phonological and/or semantic erosion, the original structure is no longer recognizable, and these forms thus receive the status of “pure” function words."""

    examples_block = """Preposition: während → Level: 2
Preposition: von → Level: 4

Preposition: trotz → Level: 1
Preposition: durch → Level: 3

Preposition: ohne → Level: 2
Preposition: zu → Level: 4

Preposition: anhand → Level: 1
Preposition: gegen → Level: 3"""

    @staticmethod
    def build_prompt_a(Keyword):
        return f"""You have been given a keyword in the form of preposition. 
Your task is to evaluate the given German preposition and rate the degree of grammaticalization 
in the form of single label from 1 to 4, where 1–4 represent increasing degrees of grammaticalization.
Preposition: {Keyword}

"""

    @staticmethod
    def build_prompt_b(Keyword):
        return f"""You are a highly trained linguistic tool specializing in German syntax and grammaticalization.  
Your task is to rate the degree of grammaticalization of the preposition given as the keyword input.

Preposition: {Keyword}

Please provide the judgment as a single integer (1, 2, 3, or 4) corresponding to the increasing degrees of grammaticalization."""

    @staticmethod
    def build_prompt_c(Keyword):
        return f"""You are a highly trained linguistic annotation model specializing in German syntax and grammaticalization.  
Your task is to evaluate the degree of grammaticalization of a given preposition, and assign it an appropriate level from 1 (least grammaticalized) to 4 (most grammaticalized).

Input: {Keyword} 
Expected Output: Label: <1/2/3/4>
"""
    
    @staticmethod
    def build_prompt_d(Keyword):
        return f"""Consider the following label definitions for evaluating the degree of grammaticalization of a preposition:
{PromptBuilder.DEFINITIONS}
 
Your task is to rate the degree of grammaticalization of the preposition given as the keyword input in the form of single label from 1 to 4.

Preposition: {Keyword}

"""
    
    @staticmethod
    def build_prompt_f(keyword: str, examples_block: str):
        return f"""Your task is to evaluate the given German preposition and rate the degree of grammaticalization 
in the form of single label from 1 to 4.Use the following examples to guide your evaluation:

{examples_block}

Preposition: {keyword} → Level:"""

    @staticmethod
    def build_prompt_e(keyword: str, examples_block: str):
        return f"""{examples_block}

Preposition: {keyword} → Level:"""
