# Final Submission Audit Report — Kestrel Home Returns Risk

**Audit Date:** October 7, 2026  
**Author:** Lead Machine Learning Engineer & System Architect  
**Project:** Kestrel Home Appliances Pre-Dispatch Return Risk System  
**Audit Purpose:** Pre-submission verification across data integrity, leakage safeguards, out-of-time validation, business economics, and production test suites.

---

## 1. Data Integrity & Reconciliation

### Exact Row Reconciliation Table
Every row transition from the raw files to the modeling partitions reconciles mathematically:

| Accounting Step | Row Count | Operational / Engineering Rationale |
| :--- | :---: | :--- |
| **`original_train_rows`** | **11,155** | Raw orders loaded directly from `data/raw/train.csv` (19 columns). |
| **`duplicate_rows_removed`** | **- 651** | 651 duplicate ingestion rows removed based on duplicate order_id records (`partner_feed`). Primary `crm` records retained. |
| **`invalid_rows_removed`** | **0** | Zero unparseable records; October 2025 paise amounts ($100\times$) rescaled rather than discarded. |
| **`missing_target_rows_removed`**| **0** | Zero missing values in `returned` column across historical orders. |
| **`unmatched_customer_rows`** | **0** | Referential integrity: 100% of customer IDs match `data/raw/customers.csv`. |
| **`unmatched_product_rows`** | **0** | Referential integrity: 100% of SKUs match `data/raw/products.csv`. |
| **`other_rows_removed`** | **0** | No ad-hoc row dropping. |
| **`final_modeling_rows`** | **10,504** | Deduplicated clean modeling dataset ($11,155 - 651 = 10,504$). |
| **`fit_rows` (Train Partition)** | **8,378** | Chronological partition: Orders placed before `2026-04-01` ($79.76\%$). |
| **`validation_rows` (Holdout)** | **2,126** | Chronological partition: Orders placed between `2026-04-01` and `2026-06-30` ($20.24\%$). |
| **`test_rows` (Unlabelled)** | **2,096** | Separate unlabelled evaluation dataset in `data/raw/test_unlabelled.csv`. |

$$\text{Reconciliation Asserted:} \quad 11,155 - 651 = 10,504 = 8,378 + 2,126$$

---

## 2. Leakage Prevention & Temporal Safeguards

### Excluded Post-Outcome Attributes
The following features were audited and strictly purged from feature engineering in `src/data.py`:
- **`pickup_scheduled_at` (100% Leakage):** Reverse logistics scheduling timestamp. Correlated with 91.62% return rate in train; 100% missing in unlabelled test data.
- **`last_service_event_type` (Post-Outcome Lifecycle Contamination):** Reflects post-delivery technician events (`REVERSE_PICKUP` has 100.0% return rate). In test data, only pre-dispatch states (`NONE`, `INSTALL_BOOKED`) appear.
- **`source` (Ingestion Metadata):** Discarded as administrative metadata.
- **`returned` (Target):** Kept strictly as the training outcome; never passed to inference.
- **`order_id` & raw `customer_id`:** Preserved as keys/joins only; excluded from predictive matrices to prevent identifier memorization.

### Temporal Feature Safeguards
- **Chronological Split:**
  - **Fit Period:** April 1, 2025 $\to$ March 31, 2026 ($N = 8,378$).
  - **Validation Period:** April 1, 2026 $\to$ June 30, 2026 ($N = 2,126$).
  - **Test Period:** July 1, 2026 $\to$ September 30, 2026 ($N = 2,096$).
- **Customer History Invariance:** `customer_prior_orders`, `customer_prior_returns`, and `prior_return_rate` represent point-in-time facts stamped at checkout before dispatch.
- **Bayes Smoothing Invariance:** Empirical Bayes smoothing parameters ($\alpha = 0.456, \beta = 3.544$) are learned strictly on the training period. Unit test `test_bayes_smoothing_invariance_to_future_data` mathematically proves that appending future returns to the evaluation stream does not alter historical transformed priors.

---

## 3. Validation Methodology & Out-of-Time Performance

Evaluated on the contiguous 3-month holdout ($N = 2,126$ orders; $245$ actual returns, base return rate = $11.52\%$):

| Metric | Majority Baseline | Logistic Regression ($L_2$) | XGBoost | HistGradientBoosting | Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy ($t=0.50$)** | 88.48% | **89.56%** | 89.28% | 89.18% | 88.85% |
| **ROC-AUC** | 0.5000 | **0.7859** | 0.7661 | 0.7525 | 0.7553 |
| **PR-AUC (Avg Precision)** | 0.1152 | **0.4151** | 0.3713 | 0.3486 | 0.3650 |
| **Brier Score** | 0.1020 | **0.0849** | 0.0879 | 0.0894 | 0.0892 |
| **Return Recall ($t=0.15$)** | 0.00% | **58.78%** | 55.92% | 52.65% | 54.69% |
| **Precision ($t=0.15$)** | 0.00% | **30.57%** | 27.85% | 29.05% | 28.09% |
| **Flagged Volume ($t=0.15$)**| 0.00% | **22.15% (471)** | 23.14% (492) | 20.88% (444) | 22.44% (477) |
| **Est. Net Savings ($t=0.15$)**| ₹0.00 | **₹36,765.00** | ₹33,002.50 | ₹31,942.50 | ₹32,470.00 |

**Model Selection:** $L_2$-Regularized Logistic Regression is selected as the champion because it is the **best-performing tested candidate under this validation setup**, delivering higher PR-AUC, superior probability calibration, lower latency (<2ms), and deterministic linear interpretability.

---

## 4. Operational Threshold Justification ($t = 0.1500$)

The threshold was determined by balancing three concrete operational constraints:

1. **Theoretical Economic Break-Even ($p^* = 0.1118$):**
   $$p \times C_{\text{return}} \times \eta = C_{\text{call}} \implies p \times 1150 \times 0.35 = 45 \implies p^* \approx 0.1118 \quad (11.18\%)$$
2. **Contact Center Capacity Constraint ($\le 25\%$ of Dispatch Volume):**
   - At $t = 0.1118$, the model flags **31.56% of orders (671 calls)**, which exceeds shift capacity.
   - At $t = 0.1200$, the model flags **28.79% of orders (612 calls)**.
   - At $t = 0.1500$, the model flags **22.15% of orders (471 calls)**, safely adhering to the 25% capacity ceiling.
3. **Model Performance & Financial Yield:**
   - At $t = 0.1500$, 144 of 245 historical returns would have been flagged for pre-dispatch review, corresponding to 58.78% return recall and **30.57% precision** (2.65x lift over baseline).
   - Estimated net savings: ₹36,765 per comparable quarter under the stated intervention assumptions (capturing 96.6% of theoretical peak unconstrained savings).

---

## 5. Economic Model & Unit Financial Impact

Derived strictly from *Operations Policy v4.1*:
- **Gross Cost of a Return ($C_{\text{return}}$):** ₹1,150.00 (logistics freight, transit packaging, QC hub inspection, restocking).
- **Confirmation Call Cost ($C_{\text{call}}$):** ₹45.00 (3-minute pre-dispatch phone verification).
- **Assumed Call Prevention Rate ($\eta$):** 35% (completed call prevents 35% of potential returns).

### Estimated Economic Impact Summary ($t = 0.1500$, Out-of-Time Quarter)
- **Baseline Return Expense (No Intervention):** $245 \times ₹1,150 = \mathbf{₹281,750.00}$
- **Estimated Prevented Returns:** $144 \times 35\% = \mathbf{50.4\text{ returns}}$
- **Estimated Gross Avoided Return Cost:** $50.4 \times ₹1,150 = \mathbf{₹57,960.00}$
- **Estimated Intervention Call Cost:** $471 \times ₹45 = \mathbf{₹21,195.00}$
- **Estimated Net Savings:** $₹57,960.00 - ₹21,195.00 = \mathbf{₹36,765.00}$ per comparable quarter
- **Estimated Intervention Cost per Prevented Return:** $₹21,195.00 / 50.4 = \mathbf{₹420.54}$

---

## 6. Production Architecture & Service Verification

- **API Service (`api/main.py`):**
  - Framework: FastAPI with lifespan context (model and pipelines loaded once at startup).
  - Validation: Pydantic v2 schemas enforcing domain invariants (e.g. `prior_returns <= prior_orders`).
  - Offline Operation: **100% offline; ₹0 paid API dependencies**.
  - Decision Terminology: Flagged orders map to `decision = "PRE_DISPATCH_REVIEW"`.
- **Operations Web Cockpit:**
  - Canonical source: `ui/index.html` (mirrored to `frontend/index.html`).
  - Features: Color-coded risk meter, pre-dispatch review action banner, deterministic reason bullets, and quick-fill test presets.
- **Test Suite Results:**
  - Test command: `pytest -v`
  - Result: **41 passed in 16.01s** (100% pass rate).

---

## 7. Submission Artifact Validation (`predictions.csv`)

Pre-generation validation assertions against `data/raw/sample_submission.csv`:
- `test_rows`: 2,096
- `prediction_rows`: 2,096
- `missing_ids`: 0
- `extra_ids`: 0
- `duplicate_ids`: 0
- `null_scores`: 0
- `min_score`: 0.0051
- `max_score`: 0.9747
- `score_mean`: 0.1120

---

## 8. Missing File Disclosure (`email-thread.txt`)

The provided assignment package available to us did not include `email-thread.txt`, so no assumptions were invented from that source. All operational assumptions used by the solution are documented from the supplied dataset, policy document (`ops-policy.pdf`), `README.txt`, and available assignment materials.

---

## 9. Key Limitations & Uncertainties

1. **Unobservable Post-Shipment Causes:** The model operates strictly prior to dispatch and cannot predict courier transit damage, customer kitchen counter sizing mismatches, or spontaneous buyer remorse.
2. **Intervention Efficiency Counterfactual ($\eta = 35\%$):** Financial savings depend on the contact center call conversion rate. We recommend a 5% randomized holdout group during live operations to measure true causal lift.
3. **Cold-Start for New SKUs:** For newly launched appliance catalog SKUs without historical order volume, predictions rely on product family priors until 30–60 days of catalog returns data accrue.

