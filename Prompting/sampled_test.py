import json
import random
from collections import defaultdict

# === Config ===
INPUT_FILE = "data_dev.jsonl"
OUTPUT_FILE = "test_set_sampled_150_min40.jsonl"
TARGET_TOTAL = 200
MIN_PER_LEVEL = 40

# === Load and Group Data ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

by_level = defaultdict(list)
for item in data:
    level = int(item["gramm_score"])
    by_level[level].append(item)

# === Step 1: Take MIN_PER_LEVEL from each level ===
selected = []
leftovers = []

for level in range(1, 5):
    level_items = by_level[level]
    if len(level_items) < MIN_PER_LEVEL:
        raise ValueError(f"Not enough examples for level {level}: found {len(level_items)}, need at least {MIN_PER_LEVEL}")
    
    sampled_min = random.sample(level_items, MIN_PER_LEVEL)
    selected.extend(sampled_min)

    # Keep the rest for optional sampling
    leftovers.extend([item for item in level_items if item not in sampled_min])

# === Step 2: Fill remaining slots (150 - 120 = 30) from leftover pool ===
remaining_slots = TARGET_TOTAL - len(selected)
if remaining_slots > len(leftovers):
    raise ValueError(f"Not enough leftover examples to fill {remaining_slots} more slots.")

selected.extend(random.sample(leftovers, remaining_slots))

# === Final Shuffle ===
random.shuffle(selected)

# === Save to File ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f" Saved {len(selected)} examples to {OUTPUT_FILE}")
