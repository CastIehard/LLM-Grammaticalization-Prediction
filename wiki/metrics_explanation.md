# Explanation of Keyword Metrics

## 1. Keyword itself 

*   **Explanation:** This is not a calculated metric but the primary **identifier** for the item being analyzed. It represents a specific expression, which may include one or more orthographic or morphological variants. For consistency, all variants are converted to lowercase.
*   **Example:** `auf grund/aufgrund`

## 2. Occurrences

*   **Explanation:** This is the most basic frequency-based metric. It represents the raw count of how many times the keyword, including all of its specified variants, appears in the entire corpus. A higher number indicates a more frequent expression. For context the corpus contains 884356312 words.
*   **Formula:**
    Let $V(\text{kw})$ be the set of all variants for a given keyword `kw`. The total occurrences are the sum of the counts of each variant in the corpus.
    $$
    \text{Occurrences}(\text{kw}) = \sum_{v \in V(\text{kw})} \text{count}(v, \text{Corpus})
    $$

## 3. Average Character Count

*   **Explanation:** This metric captures a physical property of the keyword: its average length. It is calculated by summing the character lengths of all defined variants of a keyword and dividing by the number of variants. This can be a weak proxy for lexical complexity or reduction over time.
*   **Formula:**
    Let $V(\text{kw})$ be the set of variants and $|V(\text{kw})|$ be the number of variants. Let $\text{len}(v)$ be the number of characters in a variant string $v$.
    $$
    \text{AvgCharCount}(\text{kw}) = \frac{\sum_{v \in V(\text{kw})} \text{len}(v)}{|V(\text{kw})|}
    $$

## 4. Amount of Distinct Neighbors

*   **Explanation:** This metric measures the lexical diversity of the keyword's immediate context. It counts the number of **unique** words that appear directly before or after any occurrence of the keyword. A high value suggests the keyword is used in a wide variety of different lexical contexts, indicating semantic flexibility. A low value suggests its usage is more restricted and idiomatic.
*   **Formula:**
    Let $\text{Neighbors}(\text{kw})$ be the multiset (list) of all tokens appearing immediately to the left or right of any occurrence of `kw`. The metric is the cardinality (size) of the set of these neighbors.
    $$
    \text{AmountDistinctNeighbors}(\text{kw}) = |\{ w \mid w \in \text{Neighbors}(\text{kw}) \}|
    $$

## 5. Word Entropy

*   **Explanation:** Word Entropy is a more sophisticated measure of contextual diversity than simply counting distinct neighbors. It applies the concept of **Shannon Entropy** to the probability distribution of the neighboring words. It quantifies the unpredictability of the keyword's context.
    *   A **high entropy** value means the context is highly unpredictable; many different neighbors appear with relatively uniform frequencies. This suggests high semantic and syntactic freedom.
    *   A **low entropy** value means the context is highly predictable; a small number of neighbors account for most of the occurrences. This points towards a more fixed, collocational usage.
*   **Formula:**
    Let $W_{\text{kw}}$ be the set of unique neighboring words for a keyword `kw`. Let $P(w)$ be the probability of a neighbor $w$ appearing in the context, calculated as the count of $w$ divided by the total number of all neighbors.
    $$
    \text{WordEntropy}(\text{kw}) = - \sum_{w \in W_{\text{kw}}} P(w) \log_2 P(w)
    $$

## 6. Collocation Strength

*   **Explanation:** This metric aims to quantify how strongly a keyword is associated with its context words, beyond what would be expected by chance. It is calculated as a frequency-weighted sum of the **Pointwise Mutual Information (PMI)** for each neighbor. PMI compares the actual probability of a keyword and a neighbor appearing together with the probability of them appearing together if they were independent. A high, positive value indicates a strong collocation. By weighting this by the bigram frequency, we give more importance to strong collocations that are also frequent.
*   **Formula:**
    First, the Pointwise Mutual Information (PMI) between a keyword `kw` and a neighboring word `w` is:
    $$
    \text{PMI}(\text{kw}, w) = \log_2\left(\frac{P(\text{kw}, w)}{P(\text{kw})P(w)}\right)
    $$
    Where $P(\text{kw}, w)$ is the probability of the co-occurrence, and $P(\text{kw})$ and $P(w)$ are the individual probabilities in the corpus. The final metric is the average PMI weighted by co-occurrence counts. Let $BC(\text{kw}, w)$ be the count of the bigram `(kw, w)`.
    $$
    \text{CollocationStrength}(\text{kw}) = \frac{1}{\text{Occurrences}(\text{kw})} \sum_{w \in \text{Neighbors}(\text{kw})} BC(\text{kw}, w) \times \text{PMI}(\text{kw}, w)
    $$

## 7. Synthetic Context Adversity (SCA)

*   **Explanation:** This metric measures the **syntactic flexibility** of a keyword. Instead of looking at the neighboring words themselves, it looks at their Part-of-Speech (POS) tags. The SCA is the total number of unique POS tags of the words appearing immediately before the keyword plus the number of unique POS tags of the words appearing immediately after. A high SCA value indicates that the keyword can appear in many different syntactic constructions (e.g., it can be preceded by nouns, verbs, and prepositions), suggesting syntactic versatility. A low value suggests a more rigid grammatical role.
*   **Formula:**
    Let $\text{PreTags}(\text{kw})$ be the set of unique POS tags of tokens immediately preceding `kw`, and $\text{PostTags}(\text{kw})$ be the set of unique POS tags of tokens immediately succeeding `kw`.
    $$
    \text{SCA}(\text{kw}) = | \text{PreTags}(\text{kw}) | + | \text{PostTags}(\text{kw}) |
    $$