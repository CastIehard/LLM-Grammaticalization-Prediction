import csv
import re
from tqdm import tqdm

def split_testset(input_path, output_path):
    result = []

    with open(input_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            keyword_field, score = row[0].strip(), row[1].strip()
            parts = keyword_field.split("/")
            for part in parts:
                clean = part.replace("_", " ").strip()
                result.append([clean, score])

    # Schreibe gesplittete Datei
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Keyword", "Score"])
        writer.writerows(result)

    print(f"✅ Gesplittetes Testset gespeichert unter: {output_path} ({len(result)} Einträge)")

# This code was run once to change the format of the testset so it can be used to search for keywords in the training set.
#input_file = "data/testset.csv"
#output_file = "data/testset_splitted.csv"
#split_testset(input_file, output_file)

# Load testset with clean keywords and grammaticalization levels see above
def load_testset(testset_path):
    keyword_level = []
    with open(testset_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keyword = row["Keyword"].strip()
            level = row["Score"].strip()
            keyword_level.append((keyword, level))
    return keyword_level

# Scan sentences for keyword matches
def match_sentences(data_path, keyword_level):
    results = []
    total_lines = sum(1 for _ in open(data_path, encoding='utf-8'))
    with open(data_path, encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Scanning sentences"):
            if '\t' not in line:
                continue
            _, sentence = line.strip().split('\t', 1)
            for keyword, level in keyword_level:
                if re.search(rf'\b{re.escape(keyword)}\b', sentence):
                    results.append([sentence, keyword, level])
    return results

# Write output CSV
def write_output(path, data):
    with open(path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sentence", "Keyword", "Grammaticalization_Level"])
        writer.writerows(data)

# File paths
data_file = "data/data.txt"               # ID<TAB>Sentence
testset_file = "data/testset.csv"         # Keyword,Score
output_file = "data/annotated_data.csv"

# Run pipeline
keyword_level = load_testset(testset_file)
matched_rows = match_sentences(data_file, keyword_level)
write_output(output_file, matched_rows)

print(f"\n✅ {len(matched_rows)} matches saved to '{output_file}'")