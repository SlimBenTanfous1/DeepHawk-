# src/get_malicious_samples.py
import pandas as pd
import os

EXPANDED = os.path.expanduser("~/Projects/deephawk/data/processed/ember_expanded.csv")
OUT = os.path.expanduser("~/Projects/deephawk/data/processed/malicious_samples.csv")

df = pd.read_csv(EXPANDED)
mal = df[df["label"] == 1]
print(f"Found {len(mal)} malicious samples")

# Save top N
N = 100
mal.sample(frac=1, random_state=42).head(N).to_csv(OUT, index=False)
print(f"Saved {min(N, len(mal))} malicious feature vectors to {OUT}")
