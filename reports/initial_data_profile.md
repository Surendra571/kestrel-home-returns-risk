# Initial Data Profile & Exploratory Analysis Report — Kestrel Home Returns Risk

## 1. Executive Summary

This report delivers a comprehensive statistical profile of the Kestrel Home Returns Risk data pack, establishing baseline distributions, data quality diagnostics, risk drivers, and temporal dynamics across 10,504 deduplicated historical orders (train) and 2,096 unlabelled orders (test).

### Key Takeaways:
1. **Class Distribution**: 11.42% overall return rate in train (1,200 returned vs 9,304 retained).
2. **Top Risk Drivers**:
   - **Payment Mode**: Cash on Delivery (COD) orders return at **18.81%**, compared to **7.68%** for prepaid UPI.
   - **Customer Prior Returns**: Dramatic monotonic risk escalation: 0 returns = **9.06%**, 1 return = **19.94%**, 2 returns = **40.81%**, 3 returns = **53.66%**, 4+ returns = **71.0%+**.
   - **Promised Delivery SLA**: Delivery promises >8 days experience a **16%–33%** return rate (buyer's remorse / delivery fatigue) vs **4.5%** for 1-day delivery.
   - **Product Category**: Complex, high-value appliances exhibit elevated return rates: Robot Vacuums (**19.50%**) and Water Purifiers (**14.82%**) vs Ceiling Fans (**6.66%**) and Mixer Grinders (**6.75%**).
   - **Kestrel Shield**: Shield members return at **18.61%** (vs **9.38%** for non-members) due to zero-friction 30-day return policy.
3. **Data Integrity Hazards**:
   - Discovered 651 exact duplicate records in train from partner outlet re-imports.
   - Discovered exactly 700 orders in October 2025 where order values were recorded in paise (100x rupees).
   - Identified catastrophic post-outcome leakage in `pickup_scheduled_at` and `last_service_event_type`.

---

## 2. Dataset Dimensions & Volume Overview

| Dataset | File Path | Raw Rows | Dedup Rows | Columns | Time Horizon |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Orders** | `data/raw/train.csv` | 11,155 | 10,504 | 19 | 2025-04-01 to 2026-06-30 (15 months) |
| **Test Orders** | `data/raw/test_unlabelled.csv`| 2,096 | 2,096 | 18 | 2026-07-01 to 2026-09-30 (3 months) |
| **Customers** | `data/raw/customers.csv` | 9,000 | 9,000 | 5 | 2022-10-15 to 2026-08-14 |
| **Products** | `data/raw/products.csv` | 21 | 21 | 6 | Launch dates: 2023 to 2025 |
| **Submission** | `data/raw/sample_submission.csv`| 2,096 | 2,096 | 2 | Mirrors test order IDs |

---

## 3. Univariate Profile of Target Variable (`returned`)

```
Train Target Counts (Deduplicated, N = 10,504):
Class 0 (Retained):  9,304 (88.58%)
Class 1 (Returned):  1,200 (11.42%)
Imbalance Ratio:    ~7.75 to 1
```

- **Operational Baseline**: A naive zero-rule classifier predicting non-return achieves **88.58% accuracy**, proving that accuracy is an uninformative metric for return prevention.
- **Monthly Return Rate Stability**:
  - Min Month: September 2025 (8.49%)
  - Max Month: April 2025 (14.98%)
  - Interquartile Range: 10.60% to 12.06%
  - Monthly Order Volume: Highly consistent at 661 to 743 orders per month.

---

## 4. Key Bivariate Risk Drivers & Return Rate Analysis

### A. Payment Mode & Checkout Friction
Cash on Delivery is a dominant source of returns due to zero financial commitment at checkout:

| Payment Mode | Order Volume | Share of Orders | Return Count | Return Rate | Relative Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cash on Delivery (`cod`)** | 3,152 | 30.01% | 593 | **18.81%** | **2.45x vs UPI** |
| **EMI (`emi`)** | 1,278 | 12.17% | 116 | **9.08%** | 1.18x vs UPI |
| **Prepaid Card (`prepaid_card`)** | 2,153 | 20.50% | 190 | **8.82%** | 1.15x vs UPI |
| **Prepaid UPI (`prepaid_upi`)** | 3,921 | 37.33% | 301 | **7.68%** | 1.00x (Baseline) |

---

### B. Customer Prior History (Reputation & Propensity)
Customer prior return behavior is the single strongest behavioral predictor in the dataset:

| Prior Returns | Order Count | Return Rate | Prior Orders | Order Count | Return Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 9,119 | **9.06%** | **0** | 2,996 | **8.88%** |
| **1** | 1,038 | **19.94%** | **1** | 3,579 | **9.44%** |
| **2** | 223 | **40.81%** | **2** | 2,308 | **11.31%** |
| **3** | 82 | **53.66%** | **3** | 990 | **15.86%** |
| **4** | 31 | **70.97%** | **4** | 378 | **23.55%** |
| **5** | 8 | **87.50%** | **5** | 158 | **29.11%** |
| **6** | 3 | **100.00%** | **6+** | 95 | **41.05%** |

---

### C. Promised Delivery Days (Delivery SLA Sensitivity)
Customer cancellation and return probability increases monotonically with promised transit days:

| Promised Days | Order Count | Return Rate | Risk Profile |
| :--- | :--- | :--- | :--- |
| **1 day** | 177 | **4.52%** | Rapid fulfillment, high conversion |
| **2 days** | 678 | **8.85%** | Low risk |
| **3–4 days** | 3,593 | **7.88%** | Standard reliable window |
| **5–6 days** | 3,772 | **12.27%** | Moderate risk |
| **7–8 days** | 1,827 | **15.71%** | High risk |
| **9–10 days** | 393 | **20.36%** | Severe risk (1 in 5 returned) |
| **11–12 days** | 64 | **29.69%** | Critical risk (nearly 1 in 3 returned) |

---

### D. Product Category & Price Elasticity
High complexity, higher ticket size products suffer substantially higher returns:

| Product Family | Orders | Mean List Price | Return Count | Return Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Robot Vacuum** | 1,554 | ₹23,099 | 303 | **19.50%** |
| **Water Purifier** | 1,451 | ₹15,749 | 215 | **14.82%** |
| **Air Fryer** | 1,516 | ₹6,824 | 182 | **12.01%** |
| **Room Heater** | 1,525 | ₹2,624 | 178 | **11.67%** |
| **Induction Cooktop** | 1,476 | ₹3,149 | 122 | **8.27%** |
| **Mixer Grinder** | 1,451 | ₹4,199 | 98 | **6.75%** |
| **Ceiling Fan** | 1,531 | ₹3,674 | 102 | **6.66%** |

---

### E. Kestrel Shield Membership Dynamics
Shield members have higher return propensity due to policy entitlements (§6):

| Shield Member | Order Count | Share | Return Rate | Strategic Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **Shield Member (`Y`)** | 2,327 | 22.15% | **18.61%** | High LTV: Intervene with white-glove onboarding; never add cancellation friction |
| **Non-Shield (`N`)** | 8,177 | 77.85% | **9.38%** | Standard intervention protocols |

---

### F. Sales Channel & Gift Flag
- **Marketplace**: 2,869 orders $\to$ **13.56% return rate** (lower brand loyalty, easier platform return policies).
- **Web**: 2,766 orders $\to$ **11.17% return rate**.
- **App**: 3,582 orders $\to$ **10.75% return rate**.
- **Partner Outlet**: 1,287 orders $\to$ **9.09% return rate** (physical evaluation at retail partner lowers remorse).
- **Gift Flag (`is_gift == 'Y'`)**: 726 orders $\to$ **16.53% return rate** vs **11.05%** for non-gifts (recipient mismatch).

---

## 5. Text & NLP Profiling: `delivery_note`

- **Missingness**: 24.65% in train (2,589 orders), 25.67% in test (538 orders).
- **Return Rate by Presence**:
  - Note Present: 7,915 orders $\to$ 11.19% return rate.
  - Note Null: 2,589 orders $\to$ 12.13% return rate.
- **Top Instruction Categories**:
  1. *Time & Date Constraints* ("Office address, weekdays only", "Customer requested morning slot", "Deliver after 6 pm"): ~11.2% return rate.
  2. *Delivery Access / Friction* ("Third floor, no lift", "Leave with security", "Ring twice"): ~10.5%–11.5% return rate.
  3. *Proactive Contact Requests* ("Call before delivery", "Do not call, WhatsApp only"): "Call before delivery" correlates with **13.17%** return rate (indicates delivery anxiety or location ambiguity).
  4. *Landmark Specifics* ("Landmark: opposite Reliance Fresh / Big Bazaar / HP petrol pump"): Ranges from 8.9% to 17.4%.

---

## 6. Train vs Test Distributional Parity (Covariate Shift Audit)

A systematic comparison verifies whether the unlabelled test set exhibits covariate shift relative to train:

| Feature Dimension | Train (Deduplicated) | Test (Unlabelled) | Drift Verdict |
| :--- | :--- | :--- | :--- |
| **Payment Mode: COD** | 30.01% | 29.82% | Identical parity |
| **Payment Mode: UPI** | 37.33% | 38.12% | Identical parity |
| **Sales Channel: App** | 34.10% | 34.45% | Identical parity |
| **Sales Channel: Mktplace**| 27.31% | 27.10% | Identical parity |
| **Promised Days (Mean)** | 5.03 days | 5.13 days | Negligible shift |
| **Discount % (Mean)** | 10.02% | 10.20% | Negligible shift |
| **Quantity = 2 Units** | 5.93% | 6.82% | Minor shift |
| **Walk-in Pincode (`000000`)**| 8.07% | 8.40% | Consistent parity |
| **Clean Order Value (Mean)**| ₹8,055.67 | ₹8,151.25 | Perfect parity after Oct fix |
| **Repeat Customer Rate** | N/A | 68.69% of test orders | High history overlap |
| **`last_service_event_type`**| 5 post-delivery events | Only `NONE` & `INSTALL_BOOKED` | **Severe Leakage Drift (DROP/REMAP)** |
| **`pickup_scheduled_at`** | 12.38% populated (leakage) | 100% NaN | **Severe Leakage Drift (DROP)** |

The underlying feature generation process between train and test is highly stationary, meaning an out-of-time validated gradient boosted decision tree (LightGBM/XGBoost/CatBoost) will generalize reliably once post-outcome leakage columns are excluded.
