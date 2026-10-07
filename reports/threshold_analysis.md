# Operational Threshold & Business Economics Analysis — Kestrel Home Returns Risk

**Document:** `reports/threshold_analysis.md`  
**Author:** Lead Machine Learning Engineer  
**Scope:** Out-of-Time Validation ($N = 2,126$ orders; $245$ true returns, $11.52\%$ base rate)  
**Economics Policy Reference:** *Operations Policy v4.1* (§4, §6, §7)  

---

## 1. Executive Summary & Core Economic Model

The Kestrel Returns Risk model outputs a well-calibrated return probability score $P(\text{returned} = 1)$ for every order at the warehouse packaging station before outbound dispatch.

### 1.1 Unit Economics Parameters (from `ops-policy.pdf`)
- **Gross Cost of a Return ($C_{\text{return}}$):** **₹1,150.00** (reverse logistics freight, transit packaging damage, QC hub testing, restocking, and refurbishment; excluding refund).
- **Outbound Confirmation Call Cost ($C_{\text{call}}$):** **₹45.00** (contact center labor and telephony for a 3-minute pre-dispatch verification call).
- **Intervention Prevention Effectiveness ($\eta$):** **35%** (a completed confirmation call resolves address ambiguities, clarifies delivery SLAs, and confirms COD payment readiness, preventing 35% of returns that would have otherwise occurred).
- **24-Hour Hold Risk:** Holding an order $>24$ hours leads to approximately **12% customer cancellation**. However, the policy **does not provide a direct rupee cost** for that cancellation.
- **Rule on Cost Modeling:** We do **NOT** fabricate an arbitrary rupee penalty for holds/cancellations. Instead, we use the hard ₹1,150 and ₹45 figures, and treat call center capacity (<25% of orders) and the 24-hour turnaround as strict operational guardrails.

---

## 2. Break-Even Probability Derivation

A confirmation call should be triggered whenever the expected avoided return loss exceeds the cost of making the call:

$$\text{Expected Avoided Loss} = p \times C_{\text{return}} \times \eta \ge C_{\text{call}}$$

$$p \times 1150 \times 0.35 = 45$$

$$p \times 402.50 = 45$$

$$p^* = \frac{45}{402.50} \approx 0.1118 \quad (11.18\%)$$

**Theoretical Implication:** For any order where the calibrated return probability exceeds **11.18%**, initiating a pre-dispatch confirmation call generates a positive expected financial payoff.

---

## 3. Threshold Evaluation Spectrum

Evaluated on out-of-time validation data ($N = 2,126$ orders, $245$ actual returns):

| Threshold ($t$) | Orders Flagged | Flagged % | Returns Caught (TP) | Returns Missed (FN) | False Positives (FP) | Call Cost (₹) | Avoided Return Cost (₹) | Expected Net Benefit (₹) | Cost per Prevented Return (₹) | Operational Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0100** | 2,086 | 98.12% | 245 | 0 | 1,841 | ₹93,870.00 | ₹98,612.50 | ₹4,742.50 | ₹1,094.69 | Flags virtually all orders; net savings collapse |
| **0.0300** | 1,675 | 78.79% | 235 | 10 | 1,440 | ₹75,375.00 | ₹94,587.50 | ₹19,212.50 | ₹916.41 | ~96% recall; violates contact center capacity |
| **0.0500** | 1,312 | 61.71% | 226 | 19 | 1,086 | ₹59,040.00 | ₹90,965.00 | ₹31,925.00 | ₹746.40 | >60% flag rate; heavy backlog |
| **0.0800** | 930 | 43.74% | 193 | 52 | 737 | ₹41,850.00 | ₹77,682.50 | ₹35,832.50 | ₹619.54 | Strong recall (78.8%), but high call load |
| **0.1000** | 752 | 35.37% | 174 | 71 | 578 | ₹33,840.00 | ₹70,035.00 | ₹36,195.00 | ₹555.67 | Approaching capacity boundary |
| **0.1118 ($p^*$)** | **671** | **31.56%** | **166** | **79** | **505** | **₹30,195.00** | **₹66,815.00** | **₹36,620.00** | **₹519.71** | **Theoretical Break-Even Threshold** |
| **0.1200** | **612** | **28.79%** | **163** | **82** | **449** | **₹27,540.00** | **₹65,607.50** | **₹38,067.50** | **₹482.73** | **Empirical Maximum Net Savings Point** |
| **0.1500** | **471** | **22.15%** | **144** | **101** | **327** | **₹21,195.00** | **₹57,960.00** | **₹36,765.00** | **₹420.54** | **Recommended Operating Point ($t_{\text{ops}}$)** |
| **0.1800** | 373 | 17.54% | 126 | 119 | 247 | ₹16,785.00 | ₹50,715.00 | ₹33,930.00 | ₹380.61 | Conservative call load; 51.4% recall |
| **0.2000** | 322 | 15.15% | 112 | 133 | 210 | ₹14,490.00 | ₹45,080.00 | ₹30,590.00 | ₹369.64 | Misses >54% of returns |
| **0.3000** | 179 | 8.42% | 78 | 167 | 101 | ₹8,055.00 | ₹31,395.00 | ₹23,340.00 | ₹295.05 | High precision (43.6%), low recall (31.8%) |
| **0.5000** | 51 | 2.40% | 37 | 208 | 14 | ₹2,295.00 | ₹14,892.50 | ₹12,597.50 | ₹177.22 | Default 0.50 cutoff |

*Baseline gross return cost with zero intervention: $245 \times ₹1,150 = ₹281,750.00$.*

---

## 4. Key Business Findings & Operational Judgment

### 4.1 Why the Default 0.50 Threshold is Suboptimal
The default threshold of $0.50$ assumes symmetric misclassification costs. In Kestrel's operations:
- A missed return costs **₹1,150.00**.
- An unnecessary call costs only **₹45.00**.
- The cost ratio is $25.56 : 1$ (or $8.94 : 1$ factoring in the 35% prevention rate).
At $t = 0.50$, the system flags only 51 orders (2.4%), catching only 37 returns and saving ₹12,597.50, leaving significant preventable return costs unaddressed.

### 4.2 Why Automatic Dispatch Hold is Risky vs. Confirmation Call
The client prompt initially states: *"We'll hold dispatch on anything it flags."*
However:
1. The operational review should be completed within the policy-defined dispatch/review window.
2. The policy provides a concrete, empirically verified cost and efficiency model for **pre-dispatch confirmation calls** (₹45 cost, 35% return reduction).
3. Therefore, our operational recommendation is:
   $$\mathbf{HIGH\ RISK} \implies \mathbf{PRE\_DISPATCH\_REVIEW}$$
   Order is routed to a contact center agent for an immediate 3-minute confirmation call. The operational review should be completed within the policy-defined dispatch/review window.

### 4.3 Why $t = 0.1500$ is Superior to $t = 0.1200$
- At $t = 0.1200$, theoretical savings peak at **₹38,067.50**, but **28.79% of all orders (612 orders)** are flagged. This exceeds typical warehouse call center staffing capacity and risks exceeding the policy-defined dispatch/review window.
- At $t = 0.1500$, the system flags **22.15% of orders (471 orders)**, which is 141 fewer calls to make per quarter, yielding estimated net savings: ₹36,765 per comparable quarter under the stated intervention assumptions (96.6% of maximum possible savings).
- Cost per prevented return at $t = 0.1500$ is **₹420.54**, which is less than 37% of the ₹1,150 return expense.

