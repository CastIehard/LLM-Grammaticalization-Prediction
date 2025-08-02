# LLMs-for-Classification-of-Grammaticalization-Degrees

This repository provides state-of-the-art experiments and resources for classifying grammaticalization degrees using Large Language Models (LLMs). Grammaticalization, the linguistic process by which words develop into grammatical elements, poses significant challenges for traditional computational methods. Leveraging powerful language models, this project aims to provide a nuanced understanding and automatic classification of different grammaticalization stages.

---

## 📂 Project Structure & Files

The project consists of several main steps and data components, managed through the scripts in the [`corpus_processing/`](corpus_processing/) directory.

### 1. 🔧 Corpus Processing & Metric Calculation

We process the large-scale SdeWaC corpus to extract meaningful features related to grammaticalization.

*   **Keyword Ground Truth:**
    *   File: [`data/general data (some additional stuff)/keyword_groundtruth.csv`](data/general%20data%20(some%20additional%20stuff)/keyword_groundtruth.csv)
    *   Contains 206 keywords with their expert-annotated grammaticalization scores (1-4).

*   **Raw Corpus Data:**
    *   Source: [SdeWaC Corpus](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sdewac/)
    *   A large (approx. 880 million tokens) German web corpus.

*   **Full Keyword Metrics:**
    *   File: [`data/full data (only for storing, do not use)/keywords_metrics_full.csv`](data/full%20data%20(only%20for%20storing,%20do%20not%20use)/keywords_metrics_full.csv)
    *   Includes calculated metrics for all 206 keywords, such as:
        *   `occurrences`
        *   `avg_character_count`
        *   `amount_distinct_neighbors`
        *   `word_entropy`
        *   `collocation_strength`
        *   `synthetic_context_adversity`
    *   For a detailed explanation of these metrics, see the [`corpus_processing/README.MD`](corpus_processing/README.MD).

### 2. 📊 Three-Way Data Split

To ensure a robust and leak-free evaluation, we have implemented a three-way data split. The datasets are **mutually exclusive** at the keyword level.

*   **Dev Data 1 (For Few-Shot Prompting)**
    *   **Location**: [`data/dev data 1 (for prompting)/`](data/dev%20data%201%20(for%20prompting)/)
    *   **Contents**: Contains exactly **8 keywords** (2 from each grammaticalization level).
    *   **Purpose**: Exclusively for sourcing examples for few-shot prompting.

*   **Dev Data 2 (For Development & Tuning)**
    *   **Location**: [`data/dev data 2 (for testing)/`](data/dev%20data%202%20(for%20testing)/)
    *   **Contents**: **50%** of the remaining keywords.
    *   **Purpose**: The primary dataset for model development and hyperparameter tuning.

*   **Test Data (For Final Evaluation)**
    *   **Location**: [`data/test data (only use at the end)/`](data/test%20data%20(only%20use%20at%20the%20end)/)
    *   **Contents**: The final **50%** of keywords.
    *   **Purpose**: A hold-out set for final model evaluation.

Each data split folder contains:
*   A `.jsonl` file with example sentences.
*   A `.csv` file with the keyword metrics for that specific split.
*   A `gramm_score_distribution.png` plot.

### 3. 📈 Evaluation

Evaluation is performed independently on the **Full**, **Dev Data 2**, and **Test** sets.

*   **Evaluation Tables**:
    *   Full: [`data/full data (only for storing, do not use)/evaluation_summary_table_full.csv`](data/full%20data%20(only%20for%20storing,%20do%20not%20use)/evaluation_summary_table_full.csv)
    *   Dev 2: [`data/dev data 2 (for testing)/evaluation_summary_table_dev2_testing.csv`](data/dev%20data%202%20(for%20testing)/evaluation_summary_table_dev2_testing.csv)
    *   Test: [`data/test data (only use at the end)/evaluation_summary_table_test.csv`](data/test%20data%20(only%20use%20at%20the%20end)/evaluation_summary_table_test.csv)
*   **Metrics**: We use Spearman's Rank Correlation (ρ) and Average Precision (AP) to assess the performance of our calculated metrics.

### 4. 🔍 Leak Checks

We are conducting checks to identify any potential overlap between training data and our evaluation set.

*   See folder: [`data_leakage_check`](data_leakage_check/)

---

## 👥 Contributors

*   **Prasanna Bhat** ([prasanna.bhat@utn.de](mailto:prasanna.bhat@utn.de)), University of Technology Nuremberg, Germany
*   **Divyansh Kaushik** ([divyansh.kaushik@utn.de](mailto:divyansh.kaushik@utn.de)), University of Technology Nuremberg, Germany
*   **Luca Burghard** ([luca.burghard@utn.de](mailto:luca.burghard@utn.de)), University of Technology Nuremberg, Germany
*   **Dominik Schlechtweg** ([dominik.schlechtweg@gmx.de](mailto:dominik.schlechtweg@gmx.de)), University of Stuttgart, Germany
*   **Anne Breitbarth** ([anne.breitbarth@ugent.be](mailto:anne.breitbarth@ugent.be)), Ghent University, Belgium

---

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.