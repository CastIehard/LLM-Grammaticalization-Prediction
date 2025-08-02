# Evaluation

## Setup Instructions

Please add CSV files from the dev2 set that are formatted as follows:

```csv
keyword,gramm_score,prediction
example_word,0.75,high
another_word,0.23,low
test_word,0.91,high
```

For reference, see `evaluation/input_csv/random_baseline_dev2.csv`.

## File Requirements

- CSV format with comma delimiters
- Required columns: `keyword`, `gramm_score`, `prediction`
- First row should contain column headers
- Only dev2 dataset files should be placed in `evaluation/input_csv/`
- Files should follow the dev2 dataset naming convention

