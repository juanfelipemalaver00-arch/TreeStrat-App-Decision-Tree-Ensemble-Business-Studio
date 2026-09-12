# 🌳 TreeStrat: Decision Tree & Ensemble Classification Studio for Business Strategy

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%201.31%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI Pipeline](https://github.com/your-org/TreeStrat/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

> **TreeStrat** is an open-source, interactive web application designed for data scientists, business analysts, and managers. It provides a guided, no-code/low-code environment to prepare data, train decision trees and ensemble algorithms, evaluate performance through cost-weighted business metrics, interpret key drivers, and convert ML results into **actionable executive business strategy roadmaps via Generative AI prompts**.

---

## 📸 Overview & Value Proposition

Traditional machine learning tools often end by reporting generic metrics like `Accuracy: 88%`. **TreeStrat** bridges the gap between statistical modeling and executive decision-making:
- **No-Code Guided Workflow**: Step-by-step wizard from raw data upload to model deployment.
- **Leakage-Free ML**: Strict train/test splitting prior to feature scaling and encoding to ensure unbiased generalization.
- **Business Cost-Weighted Metrics**: Evaluate models based on real-world misclassification costs (e.g., False Negatives costing 3x False Positives).
- **Generative AI Strategy Engine**: Automatically synthesizes model outputs into structured prompts for ChatGPT, Gemini, or Claude to generate 10-point operational action plans.

---

## 🏗️ Technical Architecture & Justification

### Why Python over R?
While R offers strong statistical packages (`tidymodels`, `ranger`), **Python** was selected as the core language for TreeStrat due to:
1. **Industry-Standard ML Ecosystem**: Unmatched performance and flexibility with `scikit-learn`, `xgboost`, `lightgbm`, and `shap`.
2. **Generative AI Integration**: Superior integration capabilities with modern LLM SDKs and prompt formatting pipelines.
3. **Deployment Friction**: Frictionless zero-setup deployment via Streamlit Community Cloud and Docker containers.

### Why Streamlit?
**Streamlit** provides a stateful, highly reactive UI (`st.session_state`) without web frontend boilerplate, allowing rapid open-source iterations and immediate publishing on GitHub.

---

## 🛠️ Stack & Recommended Libraries

| Stage | Recommended Libraries | Purpose |
| :--- | :--- | :--- |
| **Ingestion** | `pandas`, `openpyxl`, `xlrd` | Ingest CSV, TXT, XLSX, XLS with automatic delimiter/sheet parsing |
| **Quality & Audit** | `pandas`, `numpy` | Detect missing values, duplicates, outliers (IQR), and constant features |
| **EDA & Data Viz** | `plotly`, `seaborn`, `matplotlib` | Interactive histograms, boxplots, correlation matrices, category bar charts |
| **Preprocessing & Splits** | `scikit-learn` (`ColumnTransformer`, `Pipeline`, `OneHotEncoder`, `StandardScaler`) | Stratified splits and leakage-free data transformation pipelines |
| **Model Zoo** | `scikit-learn`, `xgboost`, `lightgbm` | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM |
| **Tuning & CV** | `scikit-learn` (`GridSearchCV`, `StratifiedKFold`) | Hyperparameter optimization and K-Fold cross-validation |
| **Evaluation & Metrics** | `scikit-learn` (`confusion_matrix`, `f1_score`, `roc_auc_score`) | Multi-metric benchmarking and cost-weighted error scoring |
| **Interpretability** | `shap`, `scikit-learn` (`export_text`) | Gini/MDI feature importance, permutation importance, tree decision rules |
| **Strategy & Export** | `joblib`, `io`, `json` | Model serialization (`.pkl`), data export (`.csv`, `.xlsx`), AI prompt synthesis |

---

## 🔄 12-Step Guided Workflow Breakdown

1. **Data Upload & Audit**: Ingest CSV, TXT, Excel. Automatic detection of rows, cols, duplicates, constant columns, missingness, and high cardinality text.
2. **Data Quality Audit**: Summary report per column with smart imputation strategies (Median, Mean, Mode, or `"Missing"` category).
3. **Variable Type Management**: Interactive data dictionary with explicit type casting and safety checks.
4. **Exploratory Data Analysis (EDA)**: Interactive Plotly histograms, boxplots, correlation heatmaps, and target class distributions.
5. **Target Definition & Diagnostics**: Select target variable. Automatic detection of binary vs. multiclass, target leakage warnings, and class imbalance checks.
6. **Stratified Train/Test Split**: Interactive split ratio slider (e.g., 80/20) with random seed and class distribution verification.
7. **Model Zoo Selection**: Choose from 6 tree-based algorithms (Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM).
8. **Interactive Hyperparameters**: Fine-tune max depth, estimators, learning rate, and split criteria without writing code.
9. **Cross-Validation & Benchmarking**: Stratified K-Fold CV to measure stability and overfit gaps before testing on reserved test data.
10. **Test Set Evaluation & Cost Matrix**: Performance leaderboard featuring Accuracy, Precision, Recall, F1, ROC-AUC, interactive confusion matrices, and custom business cost weights.
11. **Model Interpretability**: Feature importance bar charts (Gini/MDI), permutation importance, and human-readable decision rules.
12. **Business Strategy & Export Center**: Auto-generate a structured AI prompt for GenAI models to create a 10-point business roadmap. Download clean datasets, trained model pickles (`.pkl`), and reports.

---

## 📂 Project Structure

```text
TreeStrat/
├── .github/
│   └── workflows/
│       └── ci.yml                # Automated GitHub Actions test pipeline
├── data/
│   └── sample_churn.csv         # Benchmark customer churn sample dataset
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py         # File ingestion & initial schema audit
│   ├── data_quality.py           # Missingness, outliers, imputation & type casting
│   ├── eda.py                    # Plotly interactive chart generators
│   ├── preprocessing.py         # Stratified split & scikit-learn ColumnTransformer
│   ├── modeling.py              # Tree model zoo & Stratified K-Fold CV
│   ├── evaluation.py            # Metrics, confusion matrix & business cost matrix
│   ├── interpretability.py      # Feature importances & tree decision rules
│   ├── business_strategy.py     # Model recommendation engine & AI strategy prompt generator
│   └── exporter.py               # Downloads (CSV, Excel, JSON, PKL model pickle)
├── tests/
│   ├── test_data_quality.py      # Unit tests for data quality
│   ├── test_preprocessing.py     # Unit tests for splitting & preprocessing
│   └── test_modeling.py         # Unit tests for training & metrics
├── app.py                        # Streamlit web application entry point
├── requirements.txt              # Production dependencies
├── LICENSE                       # MIT Open Source License
└── README.md                     # Application documentation
```

---

## ⚡ Quick Start & Installation

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone Repository
```bash
git clone https://github.com/your-username/TreeStrat.git
cd TreeStrat
```

### 2. Create & Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

To run the automated test suite:
```bash
pytest tests/ -v
```

---

## 🤖 AI Business Strategy Generator Example

TreeStrat converts technical model metrics into a structured prompt for LLMs (ChatGPT, Gemini, Claude) to produce a **10-Point Business Strategy Roadmap**:
1. **Executive Summary**
2. **Key Predictive Findings & Drivers**
3. **Target Segment Profiling**
4. **Recommended Business Actions**
5. **Action Prioritization Matrix (Impact vs Effort)**
6. **Expected Business Impact & Quantified Financial ROI**
7. **Key Performance Indicators (KPIs - 30/60/90 days)**
8. **Phased Implementation Roadmap**
9. **Operational & Model Drift Risk Mitigation**
10. **Stakeholder Action Items (Marketing, Ops, Finance)**

---

## 🔌 Extensibility & Future Roadmap

TreeStrat is built modularly under `src/`. Future enhancements include:
- **SHAP Interactive Explainer Cards**: Adding SHAP waterfall plots per individual row prediction.
- **Automated Hyperparameter Search**: Integrating Optuna for Bayesian hyperparameter optimization.
- **Linear & Neural Benchmarks**: Adding Logistic Regression and Neural Networks as baseline comparison models.
- **REST API Endpoint**: Exposing trained model pipelines via FastAPI.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
