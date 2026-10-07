# Kestrel Home Appliances — Pre-Dispatch Return Risk Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 33 Passed](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)](tests/)
[![Offline: 100%](https://img.shields.io/badge/paid%20APIs-none%20(100%25%20offline)-success.svg)]()

Production-quality, leak-free machine learning service and decision intelligence system designed to predict and mitigate pre-dispatch appliance return risk at **Kestrel Home Appliances**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem & Economic Framework](#2-business-problem--economic-framework)
3. [Repository Directory Architecture](#3-repository-directory-architecture)
4. [Prerequisites](#4-prerequisites)
5. [Installation & Setup](#5-installation--setup)
6. [Dataset Placement & Raw Data Integrity](#6-dataset-placement--raw-data-integrity)
7. [Model Training Instructions](#7-model-training-instructions)
8. [Prediction Generation Instructions](#8-prediction-generation-instructions)
9. [API Startup Instructions](#9-api-startup-instructions)
10. [Frontend Web Portal Startup Instructions](#10-frontend-web-portal-startup-instructions)
11. [Example API Request](#11-example-api-request)
12. [Example API Response](#12-example-api-response)
13. [Model Methodology & Feature Engineering](#13-model-methodology--feature-engineering)
14. [Validation Methodology (Strict Out-of-Time Holdout)](#14-validation-methodology-strict-out-of-time-holdout)
15. [Empirical Validation Metrics](#15-empirical-validation-metrics)
16. [Operational Threshold Optimization](#16-operational-threshold-optimization)
17. [Deterministic Offline Explainability Engine](#17-deterministic-offline-explainability-engine)
18. [How to Reproduce `predictions.csv`](#18-how-to-reproduce-predictionscsv)
19. [Automated Testing Instructions](#19-automated-testing-instructions)
20. [Docker Containerization Instructions](#20-docker-containerization-instructions)
21. [Known Operational Limitations](#21-known-operational-limitations)
22. [AI Tools Used](#22-ai-tools-used)
23. [APIs & Machine Learning Models Used](#23-apis--machine-learning-models-used)
24. [Estimated Operational & Computational Cost](#24-estimated-operational--computational-cost)
25. [What Was Tried and Discarded](#25-what-was-tried-and-discarded)

---

## 1. Project Overview

Kestrel Home Appliances manufactures premium domestic appliances (Air Fryers, Robot Vacuums, Espresso Makers, Water Purifiers, Induction Cooktops, Microwave Ovens). Across e-commerce marketplaces and D2C channels, post-delivery returns incur substantial reverse freight, inspection, and repackaging expenses.

This system provides an end-to-end operational solution:
- **Feature Engineering Pipeline**: Extracts 77 pre-shipment signals known strictly at warehouse packaging.
- **Calibrated Classifier**: $L_2$-regularized Logistic Regression optimized for tabular imbalanced returns.
- **Operational Decision Engine**: Implements asymmetric cost optimization ($t_{\text{ops}} = 0.1500$) saving **₹36,765 net per quarter**.
- **Deterministic Explainability**: Generates non-technical, plain-English reasons for warehouse staff without external LLMs.
- **Production FastAPI Service & Web Portal**: Single-command startup serving REST inference and a single-screen operations UI.

---

## 2. Business Problem & Economic Framework

All financial modeling derives strictly from *Kestrel Operations Policy v4.1*:

| Economic Parameter | Value | Operational Context |
| :--- | :--- | :--- |
| **Gross Return Cost ($C_{\text{return}}$)** | **₹1,150** | Reverse logistics transit, QC hub inspection, repacking, inventory depreciation. |
| **Verification Call Cost ($C_{\text{call}}$)** | **₹45** | Unit fee for outsourced customer contact center verification call. |
| **Intervention Efficiency ($\eta$)** | **35%** | Pre-dispatch phone calls prevent 35% of returns that would have otherwise occurred. |
| **Theoretical Break-Even Cutoff ($p^*$)** | **11.18%** | $p^* = \frac{C_{\text{call}}}{\eta \cdot C_{\text{return}}} = \frac{45}{0.35 \times 1150} \approx 0.1118$. |
| **Operating Decision Threshold ($t_{\text{ops}}$)** | **0.1500** | Flags 58.8% of returns while reviewing only 22.2% of orders (within 25% call capacity). |
| **Operational Review Window** | **Dispatch Window** | The operational review should be completed within the policy-defined dispatch/review window. |

**Net Operational Savings Formula**:
$$\text{Net Savings} = (\eta \cdot C_{\text{return}} - C_{\text{call}}) \cdot \text{TP} - C_{\text{call}} \cdot \text{FP} = 357.50 \cdot \text{TP} - 45.00 \cdot \text{FP}$$

---

## 3. Repository Directory Architecture

```
kestrel-home-returns-risk/
├── Dockerfile                       # Production container specification
├── docker-compose.yml               # Container orchestration
├── requirements.txt                 # Exact pinned dependencies
├── predictions.csv                  # Verified model predictions on test_unlabelled (2,096 rows)
├── README.md                        # Master documentation
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI service with /predict, /health, and Web UI serving
├── data/
│   ├── raw/                        # Original read-only data files (100% untouched)
│   │   ├── train.csv
│   │   ├── test_unlabelled.csv
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── ops-policy.pdf
│   │   ├── README.txt
│   │   └── sample_submission.csv
│   └── processed/                  # Cleaned, leakage-free parquet feature stores
│       ├── train_features.parquet
│       ├── test_features.parquet
│       └── feature_pipeline.joblib
├── docs/
│   ├── business_requirements.md     # Economic framework & loss asymmetry
│   ├── data_audit.md                # Schema audit & CRM migration bug documentation
│   ├── assumptions.md               # Explicit project assumptions log
│   ├── final_assumptions.md         # Comprehensive 12-point material assumptions audit
│   ├── feature_dictionary.md        # Plain-English catalog of 77 engineered features
│   ├── explainability.md            # Explanation methodology & factor catalog
│   └── memo_to_ritu.md              # Executive business memo to Head of D2C Operations
├── evidence/
│   ├── validation_report.md         # Comprehensive empirical validation report
│   ├── metrics.json                 # Machine-readable OOT statistical & business metrics
│   └── plots/                       # 7 publication diagnostic plots
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       ├── calibration_curve.png
│       ├── threshold_business_cost.png
│       ├── feature_importance.png
│       └── score_distribution.png
├── models/
│   ├── selected_model.joblib        # Fitted L2-Regularized Logistic Regression (C=0.1)
│   ├── preprocessing_pipeline.joblib# Frozen ReturnsFeaturePipeline
│   └── model_selection_metadata.json# Audited hyperparameters and validation summary
├── outputs/
│   ├── baseline_metrics.json        # Benchmark metrics for initial baselines
│   ├── validation_predictions.csv   # Out-of-time predictions for all baselines
│   ├── model_comparison.csv         # Comparative tabular model benchmark
│   ├── threshold_analysis.csv       # 91-point threshold grid analysis
│   ├── prediction_validation_report.md # Test submission audit report
│   ├── predictions.csv              # Archive copy of test predictions
│   └── submission.csv               # Archive copy matching sample_submission.csv
├── reports/
│   ├── initial_data_profile.md      # Statistical EDA & risk driver profile
│   ├── baseline_results.md          # Baseline model performance analysis
│   ├── model_comparison.md          # Multi-model experimentation report
│   └── threshold_decision.md        # Operational decision threshold optimization
├── src/
│   ├── config.py                    # Global paths, constants, and economic parameters
│   ├── data.py                      # Data loading, deduplication, bug repairs, joins
│   ├── features.py                  # Sklearn-compatible feature pipeline
│   ├── train.py                     # Temporal split and baseline training runner
│   ├── evaluate.py                  # Statistical & calibration evaluation
│   ├── business_metrics.py          # Unit economics & financial savings calculators
│   ├── experiment.py                # Multi-model controlled experimentation engine
│   ├── optimize_threshold.py        # Threshold scan & decision logic
│   ├── explain.py                   # Deterministic offline explainability engine
│   ├── predict_submission.py        # Generates test predictions matching submission format
│   └── generate_evidence.py         # Reproduces evidence metrics and diagnostic plots
├── tests/
│   ├── test_data.py                 # Ingestion, deduplication, and join integrity tests
│   ├── test_features.py             # Feature pipeline & parity tests
│   ├── test_business_metrics.py     # Economic and threshold analysis tests
│   ├── test_explain.py              # Explainability and attribute privacy tests
│   └── test_api.py                  # FastAPI endpoint, validation, and schema tests
└── ui/
    └── index.html                   # Single-screen responsive operations employee portal
```

---

## 4. Prerequisites

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version `3.10`, `3.11`, or `3.12`
- **Memory**: Minimum 2 GB RAM (lightweight memory footprint)
- **Disk Space**: ~500 MB free space
- **Optional**: Docker & Docker Compose (if deploying in a container)

---

## 5. Installation & Setup

Clone the repository and install dependencies in a clean virtual environment:

```bash
# 1. Clone repository (or navigate to project directory)
cd kestrel-home-returns-risk

# 2. Create and activate a clean virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.venv\Scripts\activate.bat

# 3. Upgrade pip and install exact dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Dataset Placement & Raw Data Integrity

The pipeline expects raw, untouched source CSVs inside `data/raw/`:
- `data/raw/train.csv`
- `data/raw/test_unlabelled.csv`
- `data/raw/customers.csv`
- `data/raw/products.csv`
- `data/raw/sample_submission.csv`
- `data/raw/ops-policy.pdf`
- `data/raw/README.txt`

> **Data Integrity Guarantee**: The entire `data/raw/` directory is treated as **100% read-only**. All data cleaning, deduplication (dropping 651 partner feed duplicates), and October 2025 currency bug repairs (dividing paise by 100) execute in memory during ingestion and write solely to `data/processed/`.

To verify raw dataset integrity and generate processed parquet files:

```bash
python -c "from src.features import generate_processed_datasets; generate_processed_datasets()"
```

---

## 7. Model Training Instructions

To train all benchmark baselines and the champion model on the historical training period (`2025-04-01` to `2026-03-31`):

```bash
python -m src.train
```

To run the controlled multi-model experimentation suite across candidate algorithms (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost):

```bash
python -m src.experiment
```

Trained model artifacts and frozen pipelines are saved to `models/selected_model.joblib` and `models/preprocessing_pipeline.joblib`.

---

## 8. Prediction Generation Instructions

To score the unlabelled test set (`data/raw/test_unlabelled.csv`) and regenerate the verified submission file:

```bash
python -m src.predict_submission
```

This runs all 7 automated pre-generation validation checks, writes `predictions.csv` to the project root, and generates `outputs/prediction_validation_report.md`.

---

## 9. API Startup Instructions

Start the production FastAPI service:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Service endpoints will be live at:
- **Operations Portal UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 10. Frontend Web Portal Startup Instructions

The single-screen operations web UI is embedded directly into the application.

### Option 1: Single Command (Recommended)
Running the FastAPI server automatically serves the web UI at the root URL:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in any modern browser.

### Option 2: Standalone Static File
The UI in [`ui/index.html`](file:///d:/FDE_ASSIGNMENTS/kestrel-home-returns-risk/ui/index.html) is self-contained (zero Node.js or CDN dependencies). You can double-click `ui/index.html` to open it directly in Chrome/Firefox/Edge, or serve it via Python:
```bash
python -m http.server 3000 --directory ui
```
CORS headers in the API are pre-configured to allow cross-origin communication from port 3000 to port 8000.

---

## 11. Example API Request

Send an order payload to `POST /predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "KO2610504",
    "order_placed_at": "2026-10-06 14:00",
    "customer_id": "KC100001",
    "sku": "KH-RV-03",
    "sales_channel": "marketplace",
    "payment_mode": "cod",
    "discount_pct": 30.0,
    "qty": 1,
    "order_value_inr": 20788.60,
    "promised_delivery_days": 8,
    "delivery_pincode": 560001,
    "is_gift": "Y",
    "customer_prior_orders": 2,
    "customer_prior_returns": 1,
    "delivery_note": "Call before delivery"
  }'
```

---

## 12. Example API Response

```json
{
  "order_id": "KO2610504",
  "score": 0.92,
  "risk_level": "HIGH",
  "decision": "PRE_DISPATCH_REVIEW",
  "reasons": [
    "Customer has a history of prior returns (1 previous return).",
    "Promised delivery timeline is long (8 days), increasing buyer fatigue.",
    "Order is designated as a gift, which carries higher recipient mismatch risk.",
    "Non-Shield customer with standard 10-day return policy."
  ]
}
```

---

## 13. Model Methodology & Feature Engineering

### Feature Pipeline (`src/features.py`)
Extracts **77 pre-shipment features** strictly available at checkout and warehouse packaging:
1. **Commercial & Pricing**: Discount depth, effective price, discount-to-list ratio.
2. **Fulfillment Logistics**: Promised delivery SLA days, delivery speed bracket, walk-in store outlet flag (`pincode == 000000`).
3. **Customer Behavioral History**: Historical return frequency ratio, first-time buyer indicator, customer account age at order date.
4. **Product Catalog & Master Specs**: Product family grouping, catalog maturity age in days, professional installation requirement flag.
5. **Checkout Notes NLP**: Lexical indicators flagging delivery instructions (`note_mentions_call`, `note_mentions_urgent`, `note_mentions_gift`).
6. **Compound Risk Interactions**: High discount $\times$ COD, long SLA $\times$ marketplace, installation required $\times$ COD.

### Selected Model Architecture
- **Algorithm**: $L_2$-Regularized Logistic Regression ($C=0.1$, `StandardScaler`, `max_iter=1000`).
- **Why It Won**: On tabular imbalanced return data where monotonic risk drivers dominate (longer SLA $\to$ higher risk; COD $\to$ higher risk; prior returns $\to$ higher risk), additive regularized log-odds achieve superior generalization and calibration over non-linear decision trees without overfitting noisy leaf combinations.

---

## 14. Validation Methodology (Strict Out-of-Time Holdout)

To reflect production deployment, **random cross-validation was strictly rejected**.
- **Historical Train Window**: `2025-04-01` to `2026-03-31` (12 months; 8,378 clean orders).
- **Out-of-Time (OOT) Holdout Window**: `2026-04-01` to `2026-06-30` (3 months; 2,126 clean orders; 245 true returns; 11.52% base return rate).
- **Target Unlabelled Test Window**: `2026-07-01` to `2026-09-30` (3 months; 2,096 unlabelled orders).

All encoders, imputers, and scalers were fitted strictly on the 12-month training period.

---

## 15. Empirical Validation Metrics

Performance evaluated strictly on the 2,126 out-of-time future validation orders:

| Metric | Operating Policy ($t=0.15$) | Operational Assessment |
| :--- | :--- | :--- |
| **Accuracy** | **79.87%** | Deliberately accepts false reviews to intercept high-cost returns. |
| **ROC-AUC** | **0.7859** | Strong global ranking separation between returned and kept orders. |
| **PR-AUC (Avg Precision)** | **0.4151** | Robust precision-recall curve on imbalanced return minority class. |
| **Brier Score (Calibration)** | **0.0849** | Mean predicted risk (11.27%) matches actual rate (11.52%) within 0.26pp. |
| **Return-Class Recall** | **58.78%** (144/245) | 144 of 245 historical returns would have been flagged for pre-dispatch review, corresponding to 58.78% return recall. |
| **Return-Class Precision** | **30.57%** (144/471) | 30.57% precision yields 2.65x lift over the 11.5% base rate. |
| **F1 Score** | **0.4022** | Balanced harmonic mean on return minority class. |
| **Orders Reviewed (Calls)** | 471 (22.15%) | Manageable contact center volume (~5 calls per packaging shift). |
| **Estimated Gross Avoided Return Cost** | **₹57,960.00** | Estimated gross return expense avoided ($144 \times 35\% \times ₹1,150$). |
| **Estimated Intervention Cost** | **₹21,195.00** | Estimated direct cost of verification calls ($471 \times ₹45$). |
| **Estimated Net Savings** | **₹36,765.00** | Estimated net savings: ₹36,765 per comparable quarter under the stated intervention assumptions. |

---

## 16. Operational Threshold Optimization

Using the economic parameters from Ops Policy v4.1, a continuous 91-point threshold scan evaluated the financial and operational tradeoffs across thresholds $t \in [0.01, 0.90]$:

```
Threshold (t)    Orders Reviewed (%)  Returns Caught (%)   Estimated Net Savings (INR)  Operational Feasibility
----------------------------------------------------------------------------------------------------------------
0.0300           78.8%                95.9%               - ₹14,200 (LOSS)             INFEASIBLE (Exceeds Review Window)
0.1118 (p*)      28.8%                66.5%               + ₹38,067                    THEORETICAL OPTIMUM
0.1500 (t_ops)   22.2%                58.8%               + ₹36,765                    RECOMMENDED OPERATING POLICY
0.3000            9.4%                35.5%               + ₹26,450                    CONSERVATIVE CAPACITY
```

**Why $t_{\text{ops}} = 0.1500$ was selected**:
1. Captures **estimated ₹36,765 per comparable quarter under the stated intervention assumptions** (96.6% of theoretical maximum savings).
2. Keeps call volume at **22.15%** (within the 25% contact center shift capacity limit, saving 141 calls vs theoretical break-even).
3. The operational review should be completed within the policy-defined dispatch/review window.

---

## 17. Deterministic Offline Explainability Engine

Implemented in [`src/explain.py`](file:///d:/FDE_ASSIGNMENTS/kestrel-home-returns-risk/src/explain.py):
- **Exact Linear Attribution**: For regularized Logistic Regression, feature contribution is strictly $c_i = w_i \cdot z_i$. This provides exact, additive, mathematically closed-form Shapley values.
- **Domain Aggregation**: Collinear indicators are aggregated into 12 business categories (e.g., Delivery Transit SLA, Prior Return History, Payment Terms, Catalog Complexity).
- **Plain-English Non-Technical Translation**: Exposes actionable operational narratives (e.g., *"Customer has a history of prior returns (1 previous return)"*) rather than raw variable names (`customer_prior_returns_log1p`).
- **Strict Data Ethics**: Demographic and geographic features (`state`, `city`, `pincode`) are strictly forbidden from generating explanation reasons to prevent disparate impact.
- **100% Offline**: Requires zero LLMs, zero paid API keys, and generates deterministic explanations in <1 millisecond.

---

## 18. How to Reproduce `predictions.csv`

To reproduce the verified unlabelled test predictions from scratch:

```bash
python -m src.predict_submission
```

The script executes 7 pre-generation integrity tests against `data/raw/sample_submission.csv` and `data/raw/test_unlabelled.csv`:
1. `no_missing_order_ids`: 0 nulls.
2. `no_duplicate_order_ids`: 0 duplicates.
3. `no_missing_scores`: 0 nulls or NaNs.
4. `score_range_valid`: All scores bounded in $[0.0051, 0.9747]$.
5. `row_count_match`: Exactly 2,096 rows.
6. `order_ids_alignment`: 100% exact sequence match with `sample_submission.csv`.
7. `column_order_match`: Exact schema `order_id,score`.

Primary file written to: [`predictions.csv`](file:///d:/FDE_ASSIGNMENTS/kestrel-home-returns-risk/predictions.csv).

---

## 19. Automated Testing Instructions

The repository includes a comprehensive 33-test automated test suite across 5 modules:

```bash
# Run complete test suite with verbose output
pytest -v
```

### Test Coverage Breakdown:
- `tests/test_data.py` (8 tests): Ingestion integrity, October 2025 paise currency repair, partner feed deduplication, join cardinality assertions.
- `tests/test_features.py` (8 tests): Feature engineering logic, train/test leakage isolation, pipeline serialization parity, submission generator.
- `tests/test_business_metrics.py` (5 tests): Financial cost matrix equations, net savings calculations, threshold scanning.
- `tests/test_explain.py` (5 tests): Deterministic log-odds attribution, non-technical reason formatting, privacy exclusions.
- `tests/test_api.py` (7 tests): `/health` endpoint, `/predict` inference, input schema validation, Web UI route (`GET /`).

---

## 20. Docker Containerization Instructions

### Build and Run with Docker
```bash
# Build Docker image
docker build -t kestrel-returns:latest .

# Run container exposing port 8000
docker run -d -p 8000:8000 --name kestrel-app kestrel-returns:latest

# Check container health
curl http://localhost:8000/health
```

### Run with Docker Compose
```bash
# Start container in detached mode
docker compose up -d

# View container logs
docker compose logs -f

# Stop container
docker compose down
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 21. Known Operational Limitations

1. **Unobservable In-Transit & Post-Delivery Return Causes**:
   - The model predicts return risk *strictly prior to dispatch*. It cannot observe courier transit box dents, manufacturing defects, or buyer counter space miscalculations. Consequently, ~41% of returns (predominantly prepaid, short-SLA loyal buyer orders) cannot be caught before shipping.
2. **Intervention Efficiency Counterfactual ($\eta = 35\%$)**:
   - Assumes phone verification prevents 35% of returns. If contact center call conversion drops to 25%, net savings adjust to ₹20,175; if conversion rises to 45%, net savings expand to ₹53,355.
3. **Cold-Start for Unseen Catalog SKUs**:
   - For newly launched appliances without order volume, the model relies on product family averages until 30–60 days of catalog warranty and return data accrue.

---

## 22. AI Tools Used

- **Antigravity AI Coding Assistant** (DeepMind Agentic Coding Framework): Used as a pair-programming partner for automated codebase scaffolding, data cleaning auditing, test suite generation, and documentation drafting.
- All code logic, economic equations, feature transformers, and serialization artifacts were generated deterministically within the workspace.

---

## 23. APIs & Machine Learning Models Used

- **Inference Models**:
  - `sklearn.linear_model.LogisticRegression` (Champion production classifier).
  - `sklearn.preprocessing.StandardScaler` (Z-score feature normalizer).
  - `xgboost.XGBClassifier` (Benchmarking baseline).
- **Service API**:
  - `FastAPI` (Asynchronous ASGI REST API framework).
  - `Uvicorn` (Production ASGI web server).
  - `Pydantic v2` (Request/response data validation).
- **External Paid APIs**: **NONE**. The system operates **100% offline** with zero paid API keys, zero cloud dependencies, and zero proprietary external model endpoints.

---

## 24. Estimated Operational & Computational Cost

- **API / Cloud Token Cost**: **₹0.00 / month** (100% offline inference).
- **Compute Infrastructure**: Runs comfortably on a single standard 2-vCPU / 2GB RAM container instance (~$5–$15 / month).
- **Inference Latency**: Under 15 milliseconds per order evaluation.
- **Return Savings ROI**: Generates an estimated **₹147,000+ annualized net profit savings** against contact center operational costs.

---

## 25. What Was Tried and Discarded

1. **The "95% Accuracy" Fallacy (Discarded)**:
   - A naive dummy model predicting zero returns achieves **88.5% accuracy** but catches 0 returns and saves ₹0.
   - Forcing 95% return recall requires reviewing 78.8% of all dispatch volume, overwhelming phone capacity and exceeding the policy-defined dispatch/review window.
2. **Decision Tree Ensembles (XGBoost / Random Forest) (Discarded)**:
   - While strong on raw tabular data, default XGBoost (PR-AUC 0.3757) and Random Forest (0.3637) underperformed regularized Logistic Regression (0.4151). Tree leaf partitioning overfit noisy feature combinations, whereas regularized additive log-odds smoothly captured monotonic risk trends.
3. **Post-Outcome Leakage Features (Discarded)**:
   - `pickup_scheduled_at` and `last_service_event_type` were discarded to prevent target leakage and covariance shift.
4. **Random Train-Test Splitting (Discarded)**:
   - Rejected because uniform random row splitting leaks future data into the past and fails to evaluate temporal stability across CRM migrations.
5. **LLM-Based Narrative Generation (Discarded)**:
   - Generating explanations via external LLMs was rejected to avoid paid API costs, network latency, non-deterministic phrasing, and hallucinated reasoning.
