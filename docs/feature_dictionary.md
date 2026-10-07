# Feature Dictionary — Kestrel Home Returns Risk

## 1. Overview & Operational Point-of-View

The **Returns Risk Pipeline** scores customer orders at the **warehouse packing/dispatch gate** immediately prior to loading goods onto the outbound delivery carrier.

### Strict Pre-Shipment Constraint:
To prevent **target leakage** and severe train-test covariate mismatch, every single feature used in this model is derived exclusively from data available at or before outbound dispatch.

### Excluded Post-Outcome Attributes (Strict Zero-Leakage Policy):
1. **`returned`** (Target):
   - Available only in historical training records. Never used as an input feature.
2. **`pickup_scheduled_at`** (100% Leakage):
   - Reverse logistics pickup booking timestamp. Populated only after a return request has been approved in the logistics system. In unlabelled test data, it is 100% missing (`NaN`). Discarded completely.
3. **`last_service_event_type`** (Post-Outcome Contamination):
   - In historical train data, records reflect lifecycle states updated weeks after delivery (`REVERSE_PICKUP`, `INSTALL_DONE`, `DEMO_DONE`, `TECH_VISIT`). In test, only pre-dispatch states (`NONE`, `INSTALL_BOOKED`) exist. Discarded to prevent severe distribution collapse.
4. **`source`** (Metadata):
   - Indicates CRM vs partner outlet feed ingestion origin. Discarded as non-predictive operational metadata.

---

## 2. Master Feature Dictionary

| Category | Total Features | Description |
| :--- | :--- | :--- |
| **Identifiers & Target** | 2 | `order_id` (preserved key), `returned` (train target only) |
| **Pricing & Value** | 11 | Transaction amounts, catalog list prices, discounts, and price ratios |
| **Fulfillment & SLA** | 4 | Promised delivery transit days, SLA thresholds, and value density |
| **Channel & Payment Flags** | 3 | High-level binary flags for checkout channel, COD, and gifts |
| **Customer Behavior & Profile** | 10 | Prior order count, prior returns, Bayes-smoothed return rate, Shield status, tenure |
| **Product Attributes & Tier** | 5 | Warranty, launch age, installation requirements, product tier |
| **Geography & Pincode** | 2 | Walk-in outlet indicators, postal zones |
| **Delivery Note Intent (NLP)** | 7 | Customer delivery note presence, length, and keyword intent flags |
| **Temporal & Ordering Patterns** | 6 | Order hour, day of week, day of month, month, weekend, late-night impulse flags |
| **High-Risk Cross Interactions** | 5 | Compound interaction terms capturing non-linear risk compounding |
| **Encoded Categoricals** | 26 | One-hot encoded channels, payment modes, product families, tiers, and states |
| **TOTAL ENGINEERED FEATURES** | **77** | Available for model training and real-time API scoring |

---

## 3. Comprehensive Feature Catalog

### Primary Identifiers & Target

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `order_id` | String | **Yes** | Unique order identifier (e.g., `KO2600000`). Preserved through all pipeline steps to maintain traceability and link directly to warehouse pick-slips and evaluation submissions. |
| `returned` | Integer | **No (Train Target Only)** | Binary ground truth label (1 = returned, 0 = retained). Kept strictly in `train_features.parquet` as the prediction target and never passed into feature matrices. |

---

### Group 1: Order Transaction & Pricing Dynamics (11 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `qty` | Integer | **Yes** | Total appliance units purchased in the order (1 or 2). Orders with multiple units have a slightly higher risk of partial remorse or over-purchasing. |
| `discount_pct` | Float | **Yes** | Stated promotional percentage discount applied at checkout (0% to 60%). Deep discounts often correlate with bargain-hunting impulse purchases. |
| `order_value_inr` | Float | **Yes** | Cleaned transaction value in INR. Corrected for the historical October 2025 paise ingestion anomaly (where amounts were stored 100x higher). Directly measures financial exposure. |
| `list_price_inr` | Float | **Yes** | Official catalog MRP/list price of the SKU from `products.csv`. Establishes product baseline value regardless of promotions. |
| `total_list_price_inr` | Float | **Yes** | Total catalog list price: $\text{list\_price\_inr} \times \text{qty}$. Total nominal purchase value before discounts. |
| `effective_unit_price` | Float | **Yes** | Net INR amount paid per appliance unit: $\text{order\_value\_inr} / \text{qty}$. |
| `discount_amount` | Float | **Yes** | Absolute rupee savings realized by the customer: $\text{total\_list\_price\_inr} - \text{order\_value\_inr}$. |
| `discount_ratio` | Float | **Yes** | Fraction of catalog value discounted: $\text{discount\_amount} / \text{total\_list\_price\_inr}$. Measures the relative depth of the promotional incentive. |
| `price_ratio` | Float | **Yes** | Ratio of paid order value to list price. Values closer to 1.0 indicate full-price purchases; lower values indicate heavy markdowns. |
| `is_deep_discount` | Binary (0/1) | **Yes** | Flag indicating whether the discount was $\ge 30\%$. Captures high-markdown promotional events where return propensity surges. |
| `is_high_order_value` | Binary (0/1) | **Yes** | Flag indicating order value $\ge ₹15,000$. High ticket items (Robot Vacuums, Water Purifiers) carry high reverse-logistics liability and customer scrutiny. |

---

### Group 2: Fulfillment & SLA Timeline (4 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `promised_delivery_days` | Integer | **Yes** | Delivery transit SLA promised to customer at checkout (1 to 12 days). Orders promised $\ge 8$ days suffer a 16%–30% return rate due to buyer fatigue and competitor alternatives. |
| `is_long_sla` | Binary (0/1) | **Yes** | Flag indicating delivery promise $\ge 7$ days. Strongly flags high-risk transit fatigue and remorse vulnerability. |
| `is_short_sla` | Binary (0/1) | **Yes** | Flag indicating rapid delivery promise $\le 2$ days. Correlates with urgent intent and very low return rates (~4.5%–8.8%). |
| `sla_per_thousand_value` | Float | **Yes** | Ratio of delivery SLA days per thousand rupees of order value. Captures low-value items taking long delivery times, which customers easily abandon. |

---

### Group 3: Channel, Payment, and Gift Flags (3 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `is_gift_order` | Binary (0/1) | **Yes** | Flag indicating order was marked as a gift (`is_gift == 'Y'`). Gifts exhibit a 16.5% return rate (vs 11.0% non-gift) due to recipient preference mismatches or duplicate gifting. |
| `is_cod` | Binary (0/1) | **Yes** | Flag indicating Cash on Delivery payment. COD is the primary driver of refusal and return (18.81% return rate vs 7.68% for prepaid UPI) due to lack of upfront financial commitment. |
| `is_marketplace` | Binary (0/1) | **Yes** | Flag indicating purchase via third-party marketplace channels (Amazon, Flipkart). Marketplaces have lower brand affinity and frictionless app returns. |

---

### Group 4: Customer Profile & Historical Return Propensity (10 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `customer_prior_orders` | Float | **Yes** | Historical completed orders placed by the customer prior to this transaction. |
| `customer_prior_returns` | Float | **Yes** | Historical returned orders initiated by the customer prior to this transaction. Strongest behavioral predictor: monotonically scales risk from 9.06% (0 returns) to 71.0%+ (4+ returns). |
| `is_repeat_customer` | Binary (0/1) | **Yes** | Flag indicating whether the customer has placed at least 1 prior order. |
| `has_prior_returns` | Binary (0/1) | **Yes** | Binary indicator that customer has previously returned at least 1 order. |
| `has_multiple_prior_returns` | Binary (0/1) | **Yes** | Flag indicating customer has previously returned $\ge 2$ orders. Identifies chronic returners. |
| `prior_return_rate` | Float | **Yes** | Raw ratio of past returns to past orders: $\text{prior\_returns} / \text{prior\_orders}$ (0.0 for first-time buyers). |
| `smoothed_prior_return_rate` | Float | **Yes** | Empirical Bayes smoothed prior return rate: $(\text{prior\_returns} + \alpha) / (\text{prior\_orders} + \beta)$, shrinking noisy low-count estimates toward the global baseline. Fitted on train only. |
| `is_shield_member` | Binary (0/1) | **Yes** | Customer is a member of Kestrel Shield loyalty program. Shield members return at 18.61% due to policy entitlement (free 30-day returns) while representing highest LTV. |
| `is_guest_or_backfilled` | Binary (0/1) | **Yes** | Flag indicating order timestamp precedes CRM signup timestamp, identifying guest checkout or backfilled offline account creation. |
| `customer_tenure_days` | Float | **Yes** | Days elapsed between account creation and order placement (clamped to $\ge 0$). Longer tenure generally correlates with account stability. |

---

### Group 5: Product Catalog & Technical Complexity (5 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `warranty_months` | Integer | **Yes** | Manufacturer warranty duration (12 or 24 months). |
| `product_age_days` | Float | **Yes** | Days elapsed between product launch date and order placement. Identifies newly launched SKUs vs mature, established appliances. |
| `requires_installation` | Binary (0/1) | **Yes** | Flag indicating product category requires professional technician setup (Water Purifiers, Robot Vacuums, Ceiling Fans). Unscheduled setup drives return frustration. |
| `product_tier` | Categorical | **Yes** | Extracted appliance tier (`Lite`, `Pro`, `Max`) indicating feature complexity and price tier. |
| `family` | Categorical | **Yes** | High-level appliance family (e.g. Robot Vacuum, Water Purifier, Air Fryer, Induction Cooktop, Ceiling Fan, Mixer Grinder, Room Heater). |

---

### Group 6: Geographic & Address Infrastructure (2 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `is_walkin_pincode` | Binary (0/1) | **Yes** | Flag indicating `delivery_pincode == 000000`, representing partner outlet in-store walk-in sales where no home courier delivery occurs. |
| `pincode_zone` | Integer | **Yes** | First digit of the delivery postal PIN code (0 for walk-in, 1–6 for Indian postal regions), capturing geographical delivery zones. |

---

### Group 7: Delivery Note NLP & Customer Intent (7 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `has_delivery_note` | Binary (0/1) | **Yes** | Indicates whether customer provided delivery instructions at checkout. |
| `note_char_length` | Integer | **Yes** | Character length of the delivery instruction text. |
| `note_mentions_call` | Binary (0/1) | **Yes** | Customer requested phone contact ("Call before delivery", "Ring bell"). Correlates with delivery anxiety, address ambiguity, and 13.2% return rate. |
| `note_mentions_no_call` | Binary (0/1) | **Yes** | Customer explicitly requested no calls ("Do not call, WhatsApp only"). |
| `note_mentions_time_slot` | Binary (0/1) | **Yes** | Delivery time restriction ("morning slot", "deliver after 6 pm", "weekdays only"). Missing slot leads to failed attempts and returns. |
| `note_mentions_friction` | Binary (0/1) | **Yes** | Physical access difficulty ("Third floor, no lift", "Leave with security", "Fragile"). Couriers refusing to climb floors trigger door-step returns. |
| `note_mentions_landmark` | Binary (0/1) | **Yes** | Delivery note provides a specific landmark ("Landmark: opposite Reliance Fresh"), indicating hard-to-find or non-standard addresses. |

---

### Group 8: Temporal & Ordering Patterns (6 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `order_hour` | Integer | **Yes** | Hour of day when order was placed (0 to 23, IST). |
| `order_dayofweek` | Integer | **Yes** | Day of week (0 = Monday, 6 = Sunday). |
| `order_day` | Integer | **Yes** | Day of month (1 to 31). |
| `order_month` | Integer | **Yes** | Calendar month (1 to 12). |
| `is_weekend` | Binary (0/1) | **Yes** | Flag indicating order was placed on Saturday or Sunday. |
| `is_night_order` | Binary (0/1) | **Yes** | Flag indicating order placed late at night (22:00 to 05:00). Late-night mobile purchases have higher rates of impulse-buying remorse. |

---

### Group 9: High-Risk Compound Interactions (5 Features)

| Feature Name | Data Type | Available Before Shipment? | Business Description & Risk Impact |
| :--- | :--- | :--- | :--- |
| `is_cod_and_high_value` | Binary (0/1) | **Yes** | Interaction: `is_cod * is_high_order_value`. High ticket cash orders frequently suffer refusal at doorstep due to cash unavailability. |
| `is_cod_and_long_sla` | Binary (0/1) | **Yes** | Interaction: `is_cod * is_long_sla`. Cash on delivery with prolonged delivery timeline has the highest cancellation rate in ecommerce. |
| `is_first_time_cod` | Binary (0/1) | **Yes** | Interaction: First-time customer selecting COD. Extremely high risk segment needing mandatory pre-dispatch phone verification. |
| `is_shield_and_high_value` | Binary (0/1) | **Yes** | Interaction: Shield member buying high-value appliance ($\ge ₹15,000$). High return propensity coupled with VIP LTV requiring white-glove onboarding. |
| `is_complex_no_installation` | Binary (0/1) | **Yes** | Interaction: Complex appliance requiring installation (`requires_installation`) purchased via online channels rather than partner outlet. |

---

### Group 10: Encoded Categorical Dimensions (26 Features)

All categorical columns are one-hot encoded using a `scikit-learn` `OneHotEncoder` fitted exclusively on training data with `handle_unknown='ignore'`.

| Feature Prefix | Encoded Features | Description |
| :--- | :--- | :--- |
| `sales_channel_*` (4) | `sales_channel_app`<br/>`sales_channel_marketplace`<br/>`sales_channel_partner_outlet`<br/>`sales_channel_web` | Channel origin of the order. Marketplaces show highest return rate (13.56%); partner outlets show lowest (9.09%). |
| `payment_mode_*` (4) | `payment_mode_cod`<br/>`payment_mode_emi`<br/>`payment_mode_prepaid_card`<br/>`payment_mode_prepaid_upi` | Payment method. COD returns at 18.81% vs UPI at 7.68%. |
| `family_*` (7) | `family_Air Fryer`<br/>`family_Ceiling Fan`<br/>`family_Induction Cooktop`<br/>`family_Mixer Grinder`<br/>`family_Robot Vacuum`<br/>`family_Room Heater`<br/>`family_Water Purifier` | Product category. Robot Vacuums (19.50%) and Water Purifiers (14.82%) have highest return rates; Ceiling Fans (6.66%) have lowest. |
| `product_tier_*` (3) | `product_tier_Lite`<br/>`product_tier_Pro`<br/>`product_tier_Max` | Appliance feature tier. |
| `state_*` (8) | `state_DL`, `state_KA`, `state_MH`, `state_MP`, `state_RJ`, `state_TN`, `state_TS`, `state_UP` | Customer geographic state. |

---

## 4. API Prediction & Production Reusability

For real-time scoring at the warehouse dispatch station:

```python
import joblib
from src.features import predict_order_features

# 1. Load fitted pipeline
pipeline = joblib.load("data/processed/feature_pipeline.joblib")

# 2. Receive dispatch order payload
order_payload = {
    "order_id": "KO2612600",
    "order_placed_at": "2026-10-06 12:30",
    "customer_id": "KC100001",
    "sku": "KH-RV-03",
    "sales_channel": "app",
    "payment_mode": "cod",
    "discount_pct": 10,
    "qty": 1,
    "order_value_inr": 26728.2,
    "promised_delivery_days": 6,
    "delivery_pincode": 560001,
    "is_gift": "N",
    "customer_prior_orders": 2,
    "customer_prior_returns": 1,
    "delivery_note": "Ring bell twice"
}

# 3. Transform to exact 77-feature vector (plus order_id)
features_df = predict_order_features(order_payload, pipeline)
# Ready for model.predict_proba(features_df.drop(columns=['order_id']))
```

