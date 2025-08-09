# LLM-based Grammaticalization Classification

This script (`main.py`) uses a Large Language Model (LLM) to classify the grammaticalization degree of German keywords based on a set of detailed linguistic definitions. It performs zero-shot classification by providing the keyword and the definitions directly to the model.

## Prerequisites

1.  **Running LLM Server**: You must have a local LLM inference server running that is compatible with the OpenAI Chat Completions API format. The script is pre-configured to connect to `http://127.0.0.1:1234/v1/chat/completions`, which is common for servers like `llama-cpp-python`, `vLLM`, or `Ollama` with an OpenAI-compatible adapter.

2.  **Python Packages**: Install the required Python packages:
    ```bash
    pip install requests pandas tqdm
    ```

## Configuration

Before running the script, you must configure the following variables at the top of `main.py`:

-   `LLM_ENDPOINT`: The URL of your local LLM server's chat completions endpoint.
-   `MODEL_NAME`: The identifier for the model you are using on your server.
-   `INPUT_FILE`: The path to the input CSV containing the keywords to be classified. It should contain `keyword` and `gramm_score` columns.
-   `description`: **(Important)** A short, descriptive name for your experiment (e.g., `zero_shot_gemma_3b`). This description is used as the filename for the output CSV, which will be saved in the `evaluation/input_csv (only dev2 nothing else)/` directory.

## How to Run

1.  Ensure your local LLM server is running.
2.  Navigate to the `Prompting/` directory.
3.  Run the script from the command line:
    ```bash
    python main.py
    ```

The script will display a progress bar as it processes each keyword.

## Output

The script will generate a single CSV file named `{description}.csv` inside the `evaluation/input_csv (only dev2 nothing else)/` directory. This file will contain the following columns:

-   `keyword`: The word or phrase that was classified.
-   `gramm_score`: The ground truth grammaticalization score.
-   `predictions`: The grammaticalization score (1-4) predicted by the LLM.

This output file is formatted to be used directly with the `evaluation/evaluation_pipeline.py` script to assess the model's performance.
