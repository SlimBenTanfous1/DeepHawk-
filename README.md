# 🦅 DeepHawk — AI-Assisted Malware Classification Dashboard  

> **DeepHawk** is a SOC-grade malware analysis dashboard powered by machine learning (LightGBM) and explainable AI (SHAP).  
> It enables cybersecurity analysts to detect, interpret, and visualize malicious behaviors in Windows binaries — entirely inside a secure virtualized lab.

---

## 🚀 Overview

DeepHawk combines **AI malware detection** with **SOC-style visualization** in a unified Streamlit dashboard.  
It leverages the **EMBER dataset** (a public benchmark for static malware classification) and transforms raw binary features into interpretable insights for analysts.

| Module | Description |
|--------|-------------|
| 🧠 **Model Training** | Trains a LightGBM classifier on structured EMBER features |
| 📊 **Batch Evaluation** | Bulk analyze multiple files and visualize model performance |
| 🔍 **Explainability (SHAP)** | Understand feature impact for any given sample |
| 📈 **Metrics & Logs** | Monitor real-time predictions and confidence levels |
| 🖥️ **SOC-Themed UI** | Matte-dark design inspired by modern SIEM dashboards (Kibana, Wazuh, Sentinel) |

---

## 🧩 Architecture

```
deephawk/
│
├── dashboard/
│   └── app.py                 # Streamlit front-end with dark SOC theme
│
├── src/
│   ├── load_ember_subset.py   # Loads subset of the EMBER dataset
│   ├── train_model.py         # Trains and evaluates LightGBM model
│   ├── expand_features.py     # Expands complex JSON fields into numeric vectors
│   ├── eval_or_create_test_sets.py  # Creates and tests malicious samples
│   └── synthesize_malicious_like.py # Generates synthetic malicious-like data
│
├── data/
│   ├── raw/                   # Original dataset (ignored in .gitignore)
│   └── processed/             # Model artifacts and processed CSVs
│
├── venv/                      # Virtual environment (ignored)
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SlimBenTanfous1/DeepHawk-.git
cd DeepHawk-
```

### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
If you don’t have a requirements file yet, run:
```bash
pip install lightgbm shap pandas numpy scikit-learn streamlit matplotlib joblib tqdm
```

---

## 🧠 Training the Model

To train or retrain the model:
```bash
python src/train_model.py
```

Once trained, the model will be saved to:
```
data/processed/ember_model.pkl
```

---

## 🧪 Running the Dashboard

Launch the Streamlit SOC interface:
```bash
cd dashboard
streamlit run app.py
```

Then open your browser at:
👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Features in Action

### 🧠 Single Sample Prediction
Upload a single CSV sample and instantly get:
- ✅ **Benign / 🦠 Malicious** classification  
- Probability & confidence visualization  
- Logged events in real time  

### 📦 Batch Evaluation
Test hundreds of samples at once — visualize model accuracy, recall, and class distribution.

### 🔍 Explainability
Uses **SHAP** values to explain *why* a sample was classified as malicious or benign.

### 📈 Metrics & History
Displays total predictions, average confidence, class balance, and live charts.

---

## 🎨 UI Preview

*(You can replace these placeholders with real screenshots)*

| SOC Dashboard | Explainability View |
|----------------|--------------------|
| ![Dashboard](docs/dashboard_preview.png) | ![Explainability](docs/explainability_view.png) |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit (custom SOC-dark theme) |
| **ML Engine** | LightGBM |
| **Explainability** | SHAP |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Streamlit native charts |
| **Dataset** | EMBER 2018 subset |

---

## 🧑‍💻 Author

**Slim Ben Tanfous**  
🎓 Cybersecurity Engineering Student @ ESPRIT  
🔐 Specializing in DFIR, SOC Automation, and AI-driven Threat Analysis  
📍 Based in Tunisia — open to international internships  

[🌐 LinkedIn](https://linkedin.com/in/slim-ben-tanfous-971b19244) • [💻 GitHub](https://github.com/SlimBenTanfous1)

---

## 🏁 License

This project is released under the **MIT License**.  
You are free to modify, distribute, and use this code for both educational and research purposes.

---

## ⭐ Acknowledgements
- **EMBER Dataset** — Endgame Malware Benchmark for Research  
- **Streamlit** — for rapid SOC-style UI prototyping  
- **LightGBM & SHAP** — for interpretable ML

---

> _“In a world of black boxes, DeepHawk brings light to the unseen logic of malware.”_ 🦅

