#!/usr/bin/env python3
# src/eval_or_create_test_sets.py
import os
import pandas as pd
import joblib
import numpy as np

PROJECT = os.path.expanduser("~/Projects/deephawk")
MODEL_PATH = os.path.join(PROJECT, "data", "processed", "ember_model.pkl")
EXPANDED_PATH = os.path.join(PROJECT, "data", "processed", "ember_expanded.csv")
OUT_DIR = os.path.join(PROJECT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def safe_load_model(path):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

def create_malicious_samples(expanded_path, out_path, N=100):
    df = pd.read_csv(expanded_path)
    mal = df[df["label"] == 1]
    if mal.empty:
        print("[!] No malicious samples found in expanded dataset.")
        return False
    mal.sample(frac=1, random_state=42).head(N).to_csv(out_path, index=False)
    print(f"[+] Saved {min(N, len(mal))} malicious samples to {out_path}")
    return True

def create_top_confident_malicious(expanded_path, model, out_path, N=100):
    df = pd.read_csv(expanded_path)
    mal = df[df["label"] == 1].reset_index(drop=True)
    if mal.empty:
        print("[!] No malicious samples found for top-confident creation.")
        return False
    X = mal.drop(columns=["label"], errors="ignore")
    probs = model.predict_proba(X)[:,1]
    mal = mal.copy()
    mal["mal_prob"] = probs
    mal.sort_values("mal_prob", ascending=False).head(N).to_csv(out_path, index=False)
    print(f"[+] Saved top {min(N, len(mal))} malicious-by-confidence to {out_path}")
    return True

def create_synthetic_malicious(expanded_path, out_path):
    df = pd.read_csv(expanded_path)
    if "label" not in df.columns:
        print("[!] Expanded dataset missing 'label' column.")
        return False
    # pick a benign template
    benign = df[df["label"]==0]
    if benign.empty:
        print("[!] No benign samples to synthesize from.")
        return False
    template = benign.sample(1, random_state=42).iloc[0].drop(labels=["label"], errors="ignore").copy()
    # Identify candidate columns to inflate (strings_numstrings, any _entropy_, histogram_, imports_count etc.)
    candidates = [c for c in template.index if ("string" in c.lower() or "entropy" in c.lower() or "histogram" in c.lower() or "import" in c.lower() or "num" in c.lower())]
    if not candidates:
        # fallback: use numeric columns
        candidates = [c for c in template.index if np.issubdtype(type(template[c]), np.number)]
    # amplify a selection of those
    for i, c in enumerate(candidates[:20]):
        val = template[c]
        try:
            template[c] = float(val) * (5 + (i % 5)) + 50
        except Exception:
            # if non-numeric, set to 100
            template[c] = 100.0
    synth = pd.DataFrame([template.values], columns=template.index)
    synth["label"] = 1
    synth.to_csv(out_path, index=False)
    print(f"[+] Saved 1 synthetic malicious-like vector to {out_path}")
    return True

def evaluate_file(model, path):
    df = pd.read_csv(path)
    # Drop known non-feature columns
    for col in ["label", "mal_prob"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Align to model's expected feature list
    model_features = model.booster_.feature_name()
    X = df[model_features] if all(c in df.columns for c in model_features) else df.iloc[:, :len(model_features)]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Samples: {len(preds)}")
    print(f"Predicted malicious count: {int(preds.sum())}")
    print(f"Mean malicious probability: {probs.mean():.4f}")
    top_idx = np.argsort(-probs)[:5]
    print("Top 5 probs:", probs[top_idx])
    return {"n": len(preds), "pred_mal": int(preds.sum()), "mean_prob": float(probs.mean())}


def main():
    # load model
    try:
        model = safe_load_model(MODEL_PATH)
        print(f"[+] Loaded model from {MODEL_PATH}")
    except FileNotFoundError as e:
        print("ERROR:", e)
        return

    # ensure expanded dataset exists
    if not os.path.isfile(EXPANDED_PATH):
        print(f"[!] Expanded dataset not found at {EXPANDED_PATH}")
        return

    files = {
        "malicious_samples.csv": os.path.join(OUT_DIR, "malicious_samples.csv"),
        "top_malicious_by_confidence.csv": os.path.join(OUT_DIR, "top_malicious_by_confidence.csv"),
        "synth_malicious.csv": os.path.join(OUT_DIR, "synth_malicious.csv"),
    }

    # create files if missing
    if not os.path.isfile(files["malicious_samples.csv"]):
        create_malicious_samples(EXPANDED_PATH, files["malicious_samples.csv"], N=100)
    if not os.path.isfile(files["top_malicious_by_confidence.csv"]):
        create_top_confident_malicious(EXPANDED_PATH, model, files["top_malicious_by_confidence.csv"], N=100)
    if not os.path.isfile(files["synth_malicious.csv"]):
        create_synthetic_malicious(EXPANDED_PATH, files["synth_malicious.csv"])

    # evaluate each
    results = {}
    for name, path in files.items():
        if os.path.isfile(path):
            results[name] = evaluate_file(model, path)
        else:
            print(f"[!] Missing {path} — skipping.")

    print("\nSummary:")
    for k,v in results.items():
        print(f" - {k}: samples={v['n']}, predicted_mal={v['pred_mal']}, mean_prob={v['mean_prob']:.4f}")

if __name__ == "__main__":
    main()
