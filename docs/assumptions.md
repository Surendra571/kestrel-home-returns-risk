# Explicit Project Assumptions & Ambiguity Log — Kestrel Home Returns Risk

In accordance with professional machine learning engineering standards and the requirement to resolve ambiguity via explicit, defensible rationale rather than silent heuristics, this document records all assumptions governing the Kestrel Home Returns Risk system.

---

## 1. Documentation & Specification Scope

### Assumption 1.1: Scope of Available Documentation & Missing `email-thread.txt`
- **Observation**: `README.txt` (line 31) and the project prompt reference `email-thread.txt`. However, this file was not provided in `data/raw/` nor present on disk. Line 33 of `README.txt` affirms: *"No other documentation is available."*
- **Explicit Assumption**: The operational context, cost accounting, CRM migration history, and business rules documented in `ops-policy.pdf` (v4.1) and `README.txt` represent the authoritative and complete specification for this engagement. No unstated constraints from the missing email thread will be assumed; all business metrics derive strictly from `ops-policy.pdf`.

---

## 2. Operational Unit of Prediction & Temporal Cutoff

### Assumption 2.1: Prediction Unit and Dispatch Snapshot
- **Observation**: `README.txt` states: *"test_unlabelled.csv: The most recent orders, same columns without returned. This file is the snapshot the warehouse sees at dispatch."*
- **Explicit Assumption**:
  1. The **unit of prediction** is an individual order (`order_id`).
  2. The **prediction point-in-time** is the packaging/dispatch gate at the warehouse prior to loading onto the outbound carrier.
  3. Any feature whose state changes after outbound dispatch cannot legally be used for inference.

### Assumption 2.2: Strict Exclusion of Post-Outcome Target Leakage
- **Observation**:
  - `pickup_scheduled_at`: Populated in `train.csv` when reverse pickup logistics are scheduled (91.6% return rate when present). In `test_unlabelled.csv`, it is 100% NaN.
  - `last_service_event_type`: In `train.csv`, contains post-delivery lifecycle states (`REVERSE_PICKUP`, `INSTALL_DONE`, `DEMO_DONE`, `TECH_VISIT`). In `test_unlabelled.csv`, only pre-dispatch states (`NONE`, `INSTALL_BOOKED`) appear.
- **Explicit Assumption**: Both columns represent post-outcome operational status. Including them directly in training models represents severe target leakage and train-test covariate mismatch.
  - `pickup_scheduled_at` will be completely discarded.
  - `last_service_event_type` cannot be used with its train values (`REVERSE_PICKUP`, etc.). The only pre-dispatch concept is whether installation was booked at checkout (`INSTALL_BOOKED` vs `NONE`). Because historical train logs only kept the *latest* service event (overwriting `INSTALL_BOOKED` with `INSTALL_DONE` etc.), models must either drop `last_service_event_type` or infer installation eligibility strictly from product catalog attributes (`products.csv`).

---

## 3. Data Cleansing & Integrity Corrections

### Assumption 3.1: Deduplication of Re-Imported Partner Feed Records
- **Observation**: `train.csv` contains 651 duplicated pairs of `order_id` (1,302 rows). One record per pair has `source == 'crm'` and the other has `source == 'partner_feed'`. All other 17 columns are 100% identical. `test_unlabelled.csv` contains 0 `partner_feed` records and 0 duplicate IDs.
- **Explicit Assumption**: The `partner_feed` records were inadvertently duplicated into the training extract during batch migration. To avoid artificial weighting and catastrophic data leakage across validation folds, the 651 duplicate records in `train.csv` will be dropped, retaining the primary `crm` record (reducing train size from 11,155 to 10,504 unique orders).

### Assumption 3.2: Correction of October 2025 Paise Currency Bug
- **Observation**: Exactly 700 orders in `train.csv` placed between `2025-10-01` and `2025-10-31` have `order_value_inr` exactly 100.0x the expected list price minus discount. Ops policy §9 notes CRM cutover occurred on `2025-10-01`. In `test_unlabelled.csv`, all order values are normal rupee amounts.
- **Explicit Assumption**: For the 700 orders placed during October 2025, the payment system stored values in paise ($1\text{ INR} = 100\text{ paise}$). In preprocessing and feature engineering, `order_value_inr` for October 2025 orders will be divided by 100 (or derived as $\text{list\_price\_inr} \times \text{qty} \times (1 - \text{discount\_pct}/100)$).

### Assumption 3.3: Walk-in Partner Store Orders (`delivery_pincode == 000000`)
- **Observation**: 848 orders in train (8.07%) and 176 in test (8.40%) have pincode `0`.
- **Explicit Assumption**: Pincode 0 is not a missing data corruption but a legitimate business category indicating in-store walk-in sales without home delivery address capture. It will be modeled as an explicit binary feature `is_walkin_outlet = 1` rather than being imputed with geographic coordinates.

### Assumption 3.4: Orders Placed Prior to Signup Date
- **Observation**: 1,999 train orders and 18 test orders have `order_placed_at < signup_date`.
- **Explicit Assumption**: Kestrel allows guest checkout or offline marketplace/partner purchases where customer CRM accounts are created at a later date. Any customer tenure feature $(\text{order\_date} - \text{signup\_date})$ will clamp negative values to 0 and flag `is_guest_or_backfilled = 1`.

---

## 4. Evaluation & Validation Framework

### Assumption 4.1: Submission Target & Format
- **Observation**: `sample_submission.csv` contains `order_id` and `score` with 2,096 rows.
- **Explicit Assumption**:
  1. The required prediction is a calibrated probability score $P(\text{returned} = 1) \in [0.0, 1.0]$.
  2. Higher scores indicate higher predicted risk of return.

### Assumption 4.2: Temporal Out-Of-Time (OOT) Validation Split
- **Observation**: Train spans `2025-04-01` to `2026-06-30` (15 months). Test spans `2026-07-01` to `2026-09-30` (3 months). Test is strictly contiguous and future relative to Train.
- **Explicit Assumption**: Standard random K-fold cross-validation is invalid because it leaks future data into the past and fails to test temporal generalization across CRM shifts. The primary evaluation must be an **Out-Of-Time (OOT) Holdout**:
  - **Train Period**: `2025-04-01` to `2026-03-31` (12 months, 8,338 orders).
  - **Validation Period**: `2026-04-01` to `2026-06-30` (3 months, 2,166 orders), exactly mirroring the 3-month test horizon.

---

## 5. Business Operations & Economic Threshold

### Assumption 5.1: Cost-Optimal Call Threshold
- **Observation**: Operations Policy v4.1 states:
  - Average return cost = ₹1,150.
  - Outbound confirmation call cost = ₹45.
  - Confirmation call prevents ~35% of returns on called orders.
  - The operational review should be completed within the policy-defined dispatch/review window.
- **Explicit Assumption**:
  1. Net cost-optimal decision threshold for calling is $p^* = \frac{45}{0.35 \times 1150} \approx 11.2\%$.
  2. The operational review should be completed within the policy-defined dispatch/review window.
  3. Shield members have free returns as an explicit contract perk. Calls to Shield members will be strictly framed as white-glove onboarding and address verification to avoid damaging high-LTV customer goodwill.
