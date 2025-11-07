# src/synthesize_malicious_like.py
import pandas as pd, numpy as np, os

EXP = os.path.expanduser("~/Projects/deephawk/data/processed/ember_expanded.csv")
OUT = os.path.expanduser("~/Projects/deephawk/data/processed/synth_malicious.csv")

df = pd.read_csv(EXP)
# take a benign template
template = df[df.label==0].sample(1, random_state=42).drop(columns=["label"]).iloc[0].copy()

# Aggressive changes (example column names — adapt to your features)
for col in ["strings_numstrings", "strings_avlength"]:
    if col in template.index:
        template[col] = template[col] * 10 + 50

# inflate some histogram/entropy numeric columns if present
for c in df.columns:
    if "byteentropy" in c or "histogram" in c:
        template[c] = template[c] if (pd.notna(template[c]) and template[c]!=0) else 10
# build a small set
synth = pd.DataFrame([template.values], columns=template.index)
synth["label"] = 1
synth.to_csv(OUT, index=False)
print("Saved synthetic malicious-like vector to", OUT)
