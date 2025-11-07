# src/top_confident_malicious.py
import pandas as pd, joblib, os, numpy as np

MODEL = os.path.expanduser("~/Projects/deephawk/data/processed/ember_model.pkl")
EXP = os.path.expanduser("~/Projects/deephawk/data/processed/ember_expanded.csv")
OUT = os.path.expanduser("~/Projects/deephawk/data/processed/top_malicious_by_confidence.csv")

model = joblib.load(MODEL)
df = pd.read_csv(EXP)
mal = df[df["label"] == 1].reset_index(drop=True)
X = mal.drop(columns=["label"])
prob = model.predict_proba(X)[:,1]   # probability for class 1 (malicious)
mal["mal_prob"] = prob
top = mal.sort_values("mal_prob", ascending=False).head(100)
top.to_csv(OUT, index=False)
print(f"Saved {len(top)} top malicious samples to {OUT}")
