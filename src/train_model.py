import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = os.path.expanduser("~/Projects/deephawk/data/processed/ember_expanded.csv")
MODEL_PATH = os.path.expanduser("~/Projects/deephawk/data/processed/ember_model.pkl")

print("[+] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"[+] Dataset loaded: {df.shape}")

# Keep only labeled samples (0=benign, 1=malicious)
df = df[df["label"].isin([0, 1])]
df = df.fillna(0)

# --- Filter numeric columns only ---
numeric_cols = df.select_dtypes(include=["int", "float", "bool"]).columns
X = df[numeric_cols].drop(columns=["label"], errors="ignore")
y = df["label"]

print(f"[+] Using {X.shape[1]} numeric features out of {df.shape[1]} total columns")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model ---
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=64,
    random_state=42
)
print("[+] Training model...")
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict(X_test)
print("[+] Evaluation:")
print(classification_report(y_test, preds))

# --- Save model ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"[✓] Model saved to {MODEL_PATH}")

