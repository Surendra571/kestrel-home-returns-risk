# Comprehensive Project Assumptions & Ambiguity Audit

**Project**: Kestrel Home Appliances — Returns Risk Prediction & Decision Intelligence System  
**Author**: Senior Machine Learning & Decision Intelligence Engineer  
**Audit Date**: October 6, 2026  
**Audited Sources**:
- `data/raw/README.txt`
- `data/raw/ops-policy.pdf` (Operations Policy v4.1)
- Missing file reference: `email-thread.txt`
- `docs/data_audit.md`
- `docs/assumptions.md`
- `docs/business_requirements.md`
- `docs/feature_dictionary.md`
- `reports/model_comparison.md`
- `reports/threshold_decision.md`
- `evidence/validation_report.md`

---

## Executive Summary

Machine learning systems deployed in mission-critical supply chains cannot succeed on automated statistical heuristics alone. In physical appliance logistics, data structures are deeply entangled with operational business processes, CRM system cutovers, partner data feeds, and warehouse workflow boundaries.

This audit provides a comprehensive, transparent record of all **12 material ambiguities** encountered during the design, development, and operationalization of the Kestrel Returns Risk system. For every ambiguity, we document:
1. **What was unclear** in the source data or business brief.
2. **What decision we made** to resolve it.
3. **Why we made that decision** (technical and operational rationale).
4. **How it affects the model or business decision**.
5. **What we would change if Kestrel provided additional information or telemetry**.

---

## 1. Definition and Tracking of a Return (`returned == 1`)

### 1. What was unclear
Neither `README.txt` nor `train.csv` explicitly differentiates between different categories of non-retained merchandise:
- Post-delivery customer-initiated returns within the 10-day (or Shield 30-day) window.
- In-transit courier delivery rejections / Return-to-Origin (RTO) due to customer unavailability or doorstep refusal on Cash on Delivery (COD).
- Pre-dispatch customer cancellations requested between checkout and packaging.
- Warranty repair / replacement requests handled via technician dispatch.

### 2. What decision we made
We treated `returned == 1` strictly as **post-dispatch goods returned to the hub**, requiring reverse logistics freight, quality control (QC) triage, repacking, and inventory write-down ($C_{\text{return}} = ₹1,150$). Pre-dispatch cancellations were assumed to have been filtered out of the dispatch queue, as warehouse packing manifests only generate for active orders.

### 3. Why we made it
Operations Policy v4.1 (§4) associates the ₹1,150 unit loss specifically with physical reverse freight, return center inspection, repacking, and inventory depreciation. Pre-dispatch cancellations incur negligible logistics cost because goods never leave the shelf.

### 4. How it affects the model or business decision
The model predicts the combined probability of doorstep refusal and post-delivery return. Because both outcomes trigger identical reverse logistics expenses (₹1,150), predicting their joint occurrence maximizes avoided supply chain loss.

### 5. What we would change with additional information
If Kestrel provided granular return reason codes (e.g., `RTO_DOORSTEP_REFUSAL`, `DEFECTIVE_ON_ARRIVAL`, `DIMENSION_MISMATCH`, `BUYER_REMORSE`), we would train a multi-task or hierarchical model. Doorstep COD refusals are highly preventable via pre-dispatch SMS/call payment confirmation, whereas manufacturing defects require assembly QC intervention rather than dispatch holds.

---

## 2. Prediction Point-in-Time & Operational Snapshot (Prediction Timing)

### 1. What was unclear
`README.txt` states: *"test_unlabelled.csv: The most recent orders, same columns without returned. This file is the snapshot the warehouse sees at dispatch."*  
However, `train.csv` contained two downstream lifecycle columns:
- `pickup_scheduled_at`: Booking timestamp for reverse pickup logistics (populated in 91.6% of returned orders; 100% NaN in test).
- `last_service_event_type`: Downstream service milestone (`REVERSE_PICKUP`, `INSTALL_DONE`, `DEMO_DONE`, `TECH_VISIT`, `NONE`, `INSTALL_BOOKED`).

### 2. What decision we made
We enforced an absolute temporal firewall at the warehouse packing station:
- Completely eliminated `pickup_scheduled_at` from all data processing.
- Completely eliminated `last_service_event_type` from model training, replacing it with an inferred feature derived strictly from `products.csv` (`requires_installation = 1` for Water Purifiers, Robot Vacuums, Ceiling Fans).

### 3. Why we made it
Including `pickup_scheduled_at` or `last_service_event_type == 'REVERSE_PICKUP'` represents severe post-outcome target leakage. In historical training data, the CRM system retroactively updated these columns when a return occurred weeks after shipment. At the moment a box is on the conveyor belt, reverse pickups have not occurred.

### 4. How it affects the model or business decision
Prevents catastrophic real-world model failure. A model trained with `pickup_scheduled_at` would appear to achieve 99% accuracy in training but collapse to 0% recall in production where test orders have no reverse pickups scheduled.

### 5. What we would change with additional information
If Kestrel logged the exact checkout state of service requests (e.g., an immutable checkout boolean `installation_requested_at_purchase`), we would include it as a clean pre-dispatch feature rather than inferring installation need from catalog metadata.

---

## 3. Availability and Aggregation of Customer History

### 1. What was unclear
`train.csv` provides two historical customer columns: `customer_prior_orders` and `customer_prior_returns`.
- It was unclear whether these counters were point-in-time snapshots as of the order checkout timestamp, or lifetime totals computed at database export time.
- 1,999 train orders (19.0%) had `order_placed_at < signup_date` in `customers.csv`.

### 2. What decision we made
1. Assumed `customer_prior_orders` and `customer_prior_returns` represent faithful point-in-time CRM counters as seen at checkout.
2. For orders where `order_placed_at < signup_date`, we recognized that guest checkouts or partner channel purchases allow buying prior to formal account creation. We clamped customer tenure $(\text{order\_date} - \text{signup\_date})$ to 0 days and created an explicit flag `is_guest_or_backfilled = 1`.

### 3. Why we made it
In real-time inference, the warehouse system queries the CRM at packing time. If counters were export-time totals, they would leak future orders placed by the same customer later in the year. Pydantic validation rules enforce `customer_prior_returns <= customer_prior_orders`.

### 4. How it affects the model or business decision
Customer return ratio ($\text{prior\_returns} / \text{prior\_orders}$) is one of the strongest predictive signals (+0.28 log-odds impact). Clamping guest tenure avoids negative-day calculation bugs without dropping valid transactions.

### 5. What we would change with additional information
If Kestrel provided raw timestamped event logs of previous transactions, we would compute exact time-decayed return rates (e.g., returns in the last 30, 90, and 365 days) rather than lifetime cumulative totals.

---

## 4. Availability of Product Information & Catalog Joins

### 1. What was unclear
`products.csv` provides 21 SKUs with `list_price_inr`, `warranty_months`, and `launch_date`.
- It was unclear whether `list_price_inr` represented historical price at launch or current selling price.
- It was unclear whether catalog specifications (e.g., warranty duration) remained constant across the 15-month training and 3-month validation horizon.

### 2. What decision we made
Treated `products.csv` as an immutable dimension master table. Engineered pre-shipment interaction features combining order-level commercial terms with product master specs:
- `discount_amount_inr = list_price_inr * (discount_pct / 100.0)`
- `effective_price_inr = list_price_inr * (1.0 - discount_pct / 100.0)`
- `product_age_days = (order_placed_at - launch_date).days`
- `requires_installation = 1` for Water Purifiers, Robot Vacuums, and Ceiling Fans.

### 3. Why we made it
Cardinality verification confirmed a strict $1:1$ join with 0 unmatched SKUs across all 11,155 train orders and 2,096 test orders. List price provides the anchor list value necessary to audit order values and detect promotional markdown depth.

### 4. How it affects the model or business decision
Product category and price point strongly modulate return risk. Premium complex appliances (Robot Vacuums at ₹24,000+) exhibit a 17.8% return rate, whereas low-cost kitchen staples (Toasters, Kettles at ₹1,500) exhibit a 7.2% return rate.

### 5. What we would change with additional information
If Kestrel provided historical price change logs or product revision/firmware batch numbers, we would track SKU version deprecation and price elasticity shifts over time.

---

## 5. Missing Data, Partner Duplication, and the October 2025 Currency Bug

### 1. What was unclear
1. **Missing `email-thread.txt`**: Referenced in `README.txt` (line 31), but absent from raw storage.
2. **Duplicate Records**: 651 pairs of orders in `train.csv` had identical order numbers and attributes, with one row labeled `source == 'crm'` and the other `source == 'partner_feed'`.
3. **October 2025 Currency Bug**: Exactly 700 orders placed between October 1 and October 31, 2025 had `order_value_inr` inflated by exactly 100.0x (e.g., an Air Fryer listed at ₹4,999 stored as ₹499,900.00).
4. **Pincode `000000`**: 848 train orders and 176 test orders had delivery pincode 0.

### 2. What decision we made
1. **Missing Documentation**: Relied strictly on `ops-policy.pdf` (v4.1) and `README.txt`, refusing to invent unstated business rules.
2. **Deduplication**: Dropped the 651 duplicate `partner_feed` records, retaining primary `crm` records (reducing clean training orders from 11,155 to 10,504).
3. **Currency Correction**: Divided `order_value_inr` by 100.0 for orders placed in October 2025 where value exceeded 10x list price.
4. **Pincode 0**: Modeled as an explicit business indicator `is_walkin_outlet = 1` (store purchases without home delivery), avoiding flawed spatial imputation.

### 3. Why we made it
- `ops-policy.pdf` (§9) notes that a major CRM system cutover occurred on October 1, 2025, which explains why the payment gateway exported integer paise ($1\text{ INR} = 100\text{ paise}$) during the initial cutover month before being patched.
- `test_unlabelled.csv` contains zero duplicate partner feeds and zero paise-inflated order values. Leaving duplicates and paise values uncorrected would severely distort loss calculations and model weights.

### 4. How it affects the model or business decision
Without correcting the currency bug, gradient and linear models treat October orders as multi-crore luxury transactions, creating severe false-positive price distortions. Deduplication prevents cross-validation leakage between identical rows.

### 5. What we would change with additional information
If Kestrel provided the raw SQL extract logs from the payment gateway migration, we would audit whether any November 2025 transactions experienced partial currency truncation.

---

## 6. Temporal Windows and Out-of-Time Validation Strategy

### 1. What was unclear
`train.csv` spans 15 contiguous months (`2025-04-01` to `2026-06-30`). `test_unlabelled.csv` spans the subsequent 3 months (`2026-07-01` to `2026-09-30`). Standard academic ML workflows often apply random K-fold cross-validation, which shuffles rows uniformly across time.

### 2. What decision we made
We strictly prohibited random splitting and designed a **contiguous Out-Of-Time (OOT) temporal validation split**:
- **Training Period**: `2025-04-01` to `2026-03-31` (First 12 months; 8,378 clean orders).
- **Validation Period**: `2026-04-01` to `2026-06-30` (Last 3 months; 2,126 clean orders; 245 returns).
- Feature pipeline transformations (scaling, encoding) were fitted *strictly* on the training period and frozen before transforming validation and test data.

### 3. Why we made it
In real operations, models predict the *future* from the *past*. Random splits cause temporal leakage: customer historical return rates and seasonal promotions from June 2026 would leak into predictions for May 2025, producing inflated, untrustworthy performance estimates.

### 4. How it affects the model or business decision
Validation metrics accurately reflect real-world production performance across seasonal demand shifts and catalog expansions.

### 5. What we would change with additional information
If Kestrel provided 3+ years of historical data, we would implement expanding-window rolling backtests (e.g., 4 consecutive quarterly test windows) to assess macroeconomic and festive-season (Diwali/Big Billion Days) model drift.

---

## 7. Treatment of New Customers (Cold-Start Problem)

### 1. What was unclear
A significant proportion of orders (~34%) are placed by first-time customers with `customer_prior_orders == 0` and `customer_prior_returns == 0`. It was unclear how their return risk should be evaluated without historical behavioral track records.

### 2. What decision we made
1. Created an explicit indicator feature `is_first_time_customer = 1`.
2. Computed `customer_historical_return_rate = 0.0` for new buyers, ensuring no division-by-zero errors.
3. Allowed the model to evaluate first-time buyers through commercial and fulfillment signals: sales channel (Marketplace vs D2C), payment mode (COD vs Prepaid UPI), discount percentage, and promised delivery transit SLA days.

### 3. Why we made it
A new buyer with zero orders cannot be penalized for having zero returns, but cannot be assumed immune to return risk either. Empirical analysis shows first-time buyers on marketplaces using COD have a 15.8% return rate.

### 4. How it affects the model or business decision
The model does not fail or output nulls for new customers. It shifts its attribution to delivery promised days, discount depth, and payment mode.

### 5. What we would change with additional information
If Kestrel tracked digital session analytics prior to checkout (e.g., user browsing duration, device type, referral UTM source, cart edit count), we could predict new customer intent with substantially higher precision.

---

## 8. Treatment of New Products (SKU Cold-Start Problem)

### 1. What was unclear
When Kestrel launches new appliance models during the year, individual SKUs have zero historical order volume in prior months.

### 2. What decision we made
1. Engineered hierarchical categorical features: product `family` (Air Fryer, Robot Vacuum, etc.) alongside specific `sku`.
2. Encoded product family priors and interaction features (e.g., `family` $\times$ `payment_mode`, `family` $\times$ `requires_installation`).
3. If an unseen SKU appears, the feature pipeline imputes list price from family median and relies on family-level return behavior.

### 3. Why we made it
While a new Robot Vacuum model (`KH-RV-04`) may lack sales history, the broader Robot Vacuum family exhibits consistent operational characteristics (installation dependency, high price point, mapping setup friction).

### 4. How it affects the model or business decision
Prevents inference crashes on newly launched SKUs and ensures rational baseline risk estimates.

### 5. What we would change with additional information
If Kestrel provided engineering product specs (e.g., dimensions, wattage, motor type, packaging box volume), we could use product attribute embeddings to generalize across catalog updates.

---

## 9. Business Cost Accounting Assumptions

### 1. What was unclear
Operations Policy v4.1 specifies:
- Gross return cost: ₹1,150.
- Outbound verification call cost: ₹45.
- Call prevention efficiency: 35%.
- Dispatch hold cancellation risk: 12% if held >24 hours.  
It was unclear whether the ₹1,150 cost includes lost product gross margin or strictly reverse logistics, and whether the ₹45 call fee applies to answered calls or all dialed attempts.

### 2. What decision we made
Adopted the explicit linear cost matrix documented in Ops Policy §4 and §6:
$$\text{Cost} = 1150 \cdot \text{FN} + 45 \cdot \text{FP} + (45 + 0.65 \cdot 1150) \cdot \text{TP} + 0 \cdot \text{TN}$$
$$\text{Net Savings} = (0.35 \cdot 1150 - 45) \cdot \text{TP} - 45 \cdot \text{FP} = 357.50 \cdot \text{TP} - 45.00 \cdot \text{FP}$$
Assumed ₹45 is the blended per-order contact center contract rate (including retries and telephony).

### 3. Why we made it
Adhering strictly to official company policy prevents arbitrary spreadsheet adjustments and aligns ML evaluation directly with the P&L managed by Operations leadership.

### 4. How it affects the model or business decision
Establishes the exact theoretical break-even threshold:
$$p^* = \frac{C_{\text{call}}}{\eta \cdot C_{\text{return}}} = \frac{45}{0.35 \times 1150} \approx 0.1118 \quad (11.18\%)$$
Any order with return probability above 11.18% yields positive expected financial return from a verification call.

### 5. What we would change with additional information
If Kestrel provided product-specific gross margins and reverse shipping freight tiers by weight/distance, we would implement variable unit-level cost matrix optimization ($C_{\text{return}}(i)$ function of product weight and transit zone).

---

## 10. Pre-Dispatch Review Interpretation & Operational Review Window

### 1. What was unclear
The business requirements mention reviewing high-risk orders before dispatch. It was unclear whether a review meant:
- Cancelling the order immediately.
- Pausing dispatch indefinitely until manual review.
- Initiating an outbound phone verification call.

### 2. What decision we made
1. A decision of `PRE_DISPATCH_REVIEW` means: **Initiate an outbound verification phone call within the policy-defined dispatch/review window**.
2. If the customer confirms address and order readiness, the order is dispatched immediately.
3. If the customer indicates buyer remorse or wrong address, the order is cancelled and restocked, saving ₹1,150.
4. If the customer is unreachable within the window, the order is released for standard dispatch.

### 3. Why we made it
The operational review should be completed within the policy-defined dispatch/review window. Phone agents can easily complete ~5–7 calls per shift on reviewed orders (22.2% review volume).

### 4. How it affects the model or business decision
We selected $t_{\text{ops}} = 0.1500$ rather than the unconstrained mathematical optimum ($t=0.1200$). While $t=0.1200$ achieves ₹1,302 more theoretical savings, it reviews 28.8% of all orders (612 calls), pushing warehouse holding bays and call capacity near operational limits. Setting $t=0.1500$ reviews only 22.15% (471 orders), securing 96.6% of maximum savings while guaranteeing compliance with review windows.

### 5. What we would change with additional information
If Kestrel provided shift-by-shift warehouse packing queue telemetry, we would implement dynamic capacity-constrained thresholding (adjusting threshold higher during peak festive rushes when call agents are overwhelmed).

---

## 11. Probability / Score Interpretation & Operational Calibration

### 1. What was unclear
`sample_submission.csv` specifies outputting `order_id` and `score`. It was unclear whether `score` should be:
- A raw, uncalibrated classification margin.
- A rank-ordered percentile.
- A well-calibrated posterior probability $P(\text{returned} = 1 \mid x)$.

### 2. What decision we made
We required `score` to represent a **well-calibrated posterior probability** bounded strictly in $[0.0, 1.0]$. We audited calibration using reliability diagrams and the Brier score loss ($\text{Brier} = 0.0849$), verifying that mean predicted risk (11.27%) matches actual observed return rate (11.52%) within 0.26 percentage points.

### 3. Why we made it
Operational thresholds derived from unit economics ($p^* = \frac{45}{0.35 \times 1150} = 0.1118$) are only valid if the output score represents a true physical probability. Uncalibrated scores would render mathematical threshold optimization meaningless.

### 4. How it affects the model or business decision
Allowed Kestrel operations to interpret risk scores directly in business terms: an order with a score of 0.40 has an estimated 40% probability of returning.

### 5. What we would change with additional information
If downstream operations required isotonic regression or temperature scaling to adapt to regional delivery hubs, we would calibrate models per logistics cluster.

---

## 12. The 95% Accuracy Expectation vs. Supply Chain Reality

### 1. What was unclear
The initial business brief inquired whether 95% classification accuracy was achievable.

### 2. What decision we made
We formally and explicitly documented that:
1. **The model does not achieve 95% accuracy** (achieving 79.87% accuracy at the operational threshold $t=0.15$).
2. **95% accuracy is mathematically and operationally the wrong objective**:
   - A useless "dummy" model that predicts zero returns for every order achieves **88.48% accuracy** simply because 88.48% of customers keep their appliances. However, this dummy model catches **0% of returns** and delivers **₹0 in savings**.
   - A model forced to achieve 95% return recall ($t=0.03$) would hold **78.8% of all dispatch volume**, flooding call centers with false positives, violating the 24-hour SLA, and triggering catastrophic cancellation penalties.

### 3. Why we made it
Transparent engineering integrity requires educating executive stakeholders on the classic accuracy paradox in imbalanced domains. Because false negatives cost ₹1,150 while false positives cost ₹45 (an $8.94:1$ cost ratio after intervention efficiency), an optimal operational model must deliberately trade raw accuracy for return recall.

### 4. How it affects the model or business decision
Prevents the business from adopting an ineffective model. Redirected operational focus toward **PR-AUC (0.4151)**, **Recall (58.78%)**, and **Net Financial Savings (₹36,765/quarter)**.

### 5. What we would change with additional information
If executive leadership establishes fixed corporate tolerance bands for customer contact frequency, we can adjust the operational threshold along the documented Pareto frontier.

---

## Summary Matrix of Audited Ambiguities

| # | Ambiguity Domain | Decision Made | Operational Impact |
| :--- | :--- | :--- | :--- |
| **1** | Return Definition | Joint post-delivery return & RTO refusal | Aligns with ₹1,150 reverse logistics loss |
| **2** | Prediction Point | Strict pre-dispatch packaging gate | Discarded `pickup_scheduled_at` (leakage) |
| **3** | Customer History | Snapshot at checkout; clamped guest tenure | Prevents negative days; captures prior return rate |
| **4** | Product Catalog | Immutable $1:1$ dimension join on SKU | Inferred installation needs; audited price markdown |
| **5** | Data Corruptions | Fixed Oct-2025 paise; deduplicated partner feeds | Eliminated 100x price distortions & fold leakage |
| **6** | Validation Split | Contiguous 3-month OOT holdout | Mirrors production temporal deployment |
| **7** | Cold-Start Buyers | Explicit indicator + commercial signals | Robust scoring for new accounts without history |
| **8** | Cold-Start SKUs | Product family hierarchical grouping | Reliable priors for newly launched appliance models |
| **9** | Unit Economics | Strict Ops Policy v4.1 unit costs | Derived exact break-even threshold ($p^* = 11.18\%$) |
| **10**| Hold Interpretation | 24-hour phone verification window | Operating threshold $t=0.15$ avoids 12% hold penalty |
| **11**| Score Interpretation| Calibrated continuous posterior probability | Validates economic threshold decision rules |
| **12**| 95% Accuracy Goal | Refuted accuracy fallacy; optimized net savings | Catches 58.8% returns, saving ₹36,765/quarter |

