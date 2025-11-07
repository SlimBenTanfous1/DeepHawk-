import json
import pandas as pd
from tqdm import tqdm
import os

BASE_DIR = os.path.expanduser("~/Projects/deephawk/data/raw/ember/dataset/ember2018")
OUTPUT_PATH = os.path.expanduser("~/Projects/deephawk/data/processed/ember_balanced.csv")

TARGET_PER_CLASS = 10000  # 10k benign + 10k malicious

def collect_samples(path_list):
    data, labels = [], []
    counts = {0: 0, 1: 0}
    for path in path_list:
        print(f"[+] Reading {path}")
        with open(path, "r") as f:
            for line in tqdm(f):
                if all(c >= TARGET_PER_CLASS for c in counts.values()):
                    break
                sample = json.loads(line)
                label = sample.pop("label", None)
                if label not in (0, 1):
                    continue
                if counts[label] < TARGET_PER_CLASS:
                    data.append(sample)
                    labels.append(label)
                    counts[label] += 1
        print(f"Current counts: {counts}")
        if all(c >= TARGET_PER_CLASS for c in counts.values()):
            break
    return data, labels

if __name__ == "__main__":
    paths = [os.path.join(BASE_DIR, f"train_features_{i}.jsonl") for i in range(6)]
    features, labels = collect_samples(paths)
    df = pd.DataFrame(features)
    df["label"] = labels
    print(f"[✓] Final balanced shape: {df.shape} (counts={pd.Series(labels).value_counts().to_dict()})")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Saved to {OUTPUT_PATH}")

