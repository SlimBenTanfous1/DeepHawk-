import pandas as pd
import json, re, os
from tqdm import tqdm

RAW_PATH = os.path.expanduser("~/Projects/deephawk/data/processed/ember_balanced.csv")
OUTPUT_PATH = os.path.expanduser("~/Projects/deephawk/data/processed/ember_expanded.csv")

def clean_to_json(s: str):
    """Clean and parse weird EMBER-style dict strings."""
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s or not s.startswith("{"):
        return {}
    s = re.sub(r"u'([^']*)'", r'"\1"', s)
    s = s.replace("'", '"').replace("False", "false").replace("True", "true").replace("None", "null")
    try:
        return json.loads(s)
    except Exception:
        return {}

def flatten_dict(d, prefix=""):
    """Recursively flatten nested dicts and lists into numeric columns."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix == "" else f"{prefix}_{k}"
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        elif isinstance(v, list):
            for i, val in enumerate(v):
                if isinstance(val, (int, float, bool)):
                    items[f"{new_key}_{i}"] = val
        elif isinstance(v, (int, float, bool)):
            items[new_key] = v
    return items

print(f"[+] Loading {RAW_PATH}")
df = pd.read_csv(RAW_PATH)
print(f"[+] Loaded shape: {df.shape}")

labels = df["label"].copy()
df = df.drop(columns=["label"], errors="ignore")

expanded_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Flattening"):
    combined = {}
    for col, val in row.items():
        parsed = clean_to_json(val) if isinstance(val, str) and val.strip().startswith("{") else None
        if parsed:
            combined.update(flatten_dict(parsed, col))
    expanded_rows.append(combined)

expanded_df = pd.DataFrame(expanded_rows)
expanded_df["label"] = labels.values
expanded_df = expanded_df.fillna(0)

print(f"[✓] Expanded shape: {expanded_df.shape}")
expanded_df.to_csv(OUTPUT_PATH, index=False)
print(f"[✓] Saved to {OUTPUT_PATH}")
