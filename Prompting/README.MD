# LLM-based Grammaticalization Classification

This project uses Large Language Models (LLMs) to classify the degree of grammaticalization for German prepositions in context. It supports zero-shot prompting, few-shot prompting with examples, and DSPy-based optimized prompting strategies.

## Prerequisites

1. Python Environment

   Install required packages:
   ```bash
   pip install pandas requests tqdm python-dotenv
   ```

2. LLM Backend

   You can use either:
   - OpenAI API (e.g., gpt-4o)
   - A local LLM server (e.g., Ollama, vLLM, llama.cpp) that supports the OpenAI-compatible Chat Completions API.

   Create a `.env` file with the following:
   ```dotenv
   OPENAI_API_KEY=your-openai-key
   LLM_ENDPOINT=http://127.0.0.1:1234/v1/chat/completions
   LLM_MODEL=your-local-model-name
   ```

## Project Structure

```
.
├── main.py               # Entry point for running inference
├── prompts.py            # Prompt templates and few-shot example generator
├── dspy_models.py        # DSPy-based baseline and optimized prompting
├── requirements.txt
└── evaluation/
    └── input_csv (only dev2 nothing else)/  # Individual output file is located here
```

## Prompting Strategies

The project supports the following prompting types:

- **Prompt A**: Instruction-only prompt.
- **Prompt B**: Instruction + label definitions.
- **Prompt C**: Instruction + label definitions + few-shot examples (2 per class).
- **DSPy Variants**: Includes baseline prompting, instruction-only optimization (zero-shot), and joint instruction + demo optimization (few-shot), using MIPROv2.

All strategies rely on the same set of label definitions and output format.

## Configuration (main.py)

You must customize the following fields in `main.py` before running:

- `USE_OPENAI`: Set `True` for OpenAI or `False` for local LLM.
- `INPUT_FILE`: Path to CSV with columns `keyword` and `gramm_score`.
- `description`: Descriptive name for the experiment run.
- `OUTPUT_PATH`: Auto-generated output path using `description`.

## Running Inference

Run the inference script:

```bash
python main.py
```

The script will show a progress bar and output predictions to a CSV file.

## Output

The script generates a CSV file in:

```
evaluation/input_csv (only dev2 nothing else)/
```

The output file includes:

- `keyword`: The sentence containing the preposition.
- `gramm_score`: Ground truth label.
- `predictions`: LLM-predicted grammaticalization level (1–4).

Individual model and prompt results are located in the `evaluation/input_csv (only dev2 nothing else)/` directory. Each output file is named using the `description` provided in the config.

## Evaluation

The output CSV is compatible with evaluation pipelines (e.g., `evaluation_pipeline.py`) to compute accuracy and other metrics.

## Notes

- Regex is used to extract predicted labels in the format `Label: [<1–4>]` or `Level: <1–4>`.
- DSPy models can log raw inputs and outputs for inspection.
- Prompt construction and DSPy logic are modular and extensible.
