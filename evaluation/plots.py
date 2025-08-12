import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/luca/Desktop/UTN/LLMs-for-Classification-of-Grammaticalization-Degrees/data/general data (some additional stuff)/keyword_groundtruth.csv"
data = pd.read_csv(file_path)

# Count the occurrences of each gramm_score
gramm_score_counts = data['gramm_score'].value_counts().sort_index()

# Plot the distribution
plt.figure(figsize=(6, 3))
gramm_score_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Grammaticalization Score', fontsize=12)
plt.ylabel('Number of Keywords', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig("evaluation/output/keyword_distribution.png")
plt.show()