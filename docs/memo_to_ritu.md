# MEMORANDUM

**TO:** Ritu Deshpande, Head of D2C Operations  
**FROM:** Lead Machine Learning & Decision Intelligence Engineer  
**DATE:** October 7, 2026  
**SUBJECT:** Operational Recommendation: Pre-Dispatch Return Risk Review System  

---

### Executive Overview

You requested an assessment of whether Kestrel should intervene on outbound appliance shipments to reduce return expenses. Below is our direct assessment based on out-of-time validation across 2,126 recent customer orders (April 1 to June 30, 2026).

---

### 1. DECISION: Should Kestrel use the model for dispatch review?

**Yes. Deploy a pre-dispatch return-risk scoring workflow that flags high-risk orders for pre-dispatch confirmation review.**

Key operational clarifications:
- **Confirmation Call, Not Indefinite Hold:** Flagged orders trigger a prompt, 3-minute pre-dispatch confirmation call by contact center agents to reconfirm shipping address, promised delivery date, and payment readiness. The operational review should be completed within the policy-defined dispatch/review window.
- **Why We Do Not Optimize for 95% Accuracy:** The validation analysis demonstrates that chasing 95% accuracy would make the system ineffective. A trivial baseline that approves every order achieves **88.48% accuracy** simply because 88.5% of orders are retained, yet it catches zero returns and delivers ₹0 in savings. Conversely, forcing the model to achieve 95% accuracy (threshold $\approx 0.72$) causes return recall to collapse to 6.1%, missing approximately 94% of returns.
- **Operating Recommendation:** We recommend operating at threshold **0.1500**, which flags **22.15% of dispatch volume** (`PRE_DISPATCH_REVIEW`), remaining comfortably within the assumed 25% call center capacity limit while capturing **58.78% of returns**.

---

### 2. NUMBER: How well does it perform on realistic out-of-time validation?

Evaluated on the contiguous 3-month holdout of 2,126 orders (245 true returns):

| Operational Metric | Validation Result | Practical Operational Meaning |
| :--- | :--- | :--- |
| **Return Catch Rate (Recall)** | **58.78%** | 144 of 245 historical returns would have been flagged for pre-dispatch review, corresponding to 58.78% return recall. |
| **Returns Missed (False Negatives)**| **41.22%** | **101 returns** missed; primarily prepaid orders with unobservable post-delivery issues (transit handling, counter size fit). |
| **Flagged Order Volume** | **22.15%** | **471 orders** flagged for phone review (~5 calls per shift), within the 25% capacity limit. |
| **Review Precision** | **30.57%** | Approximately 1 in every 3.27 confirmation calls reaches an order that would otherwise return (a **2.65x lift** over the 11.52% base rate). |
| **Operating Accuracy** | **79.87%** | Intentionally lower than the 88.48% majority baseline because the system accepts false calls (₹45) to capture 8x costlier return losses (₹1,150). |

---

### 3. RUPEES: What is the estimated financial impact?

Under the policy assumptions ($C_{\text{return}} = \text{₹1,150}$, $C_{\text{call}} = \text{₹45}$, and an assumed $\eta = 35\%$ call prevention efficiency):

```text
Quarterly Baseline Return Expense (No Intervention):   ₹281,750  (245 returns × ₹1,150)
Estimated Gross Avoided Return Cost:                 - ₹57,960  (144 caught × 35% saved × ₹1,150)
Estimated Intervention Call Cost:                    + ₹21,195  (471 calls × ₹45)
------------------------------------------------------------------------------------------
ESTIMATED NET SAVINGS:                                 ₹36,765  per comparable quarter
```

- **Estimated Intervention Cost per Prevented Return:** We spend **₹420.54 in call expenses** for each estimated prevented return ($144 \times 35\% = 50.4$ estimated prevented returns)—substantially lower than the ₹1,150 return expense.
- **Important Financial Caveat:** These figures represent **estimated savings potential** based on historical holdout data and the policy's assumed 35% prevention rate. They are not realized savings until validated through live operations.

---

### 4. NEXT WEEK: What should Ritu actually do next week?

We recommend a disciplined, 2-week controlled operational pilot before full rollout:

1. **Run a Controlled Pilot:** Deploy the pre-dispatch review portal (`http://localhost:8000`) at the primary fulfillment center for high-risk orders ($score \ge 0.15$).
2. **Measure Actual Call Success & Prevention Rate:** Verify whether phone agents achieve the assumed 35% return prevention rate under real contact center conditions.
3. **Monitor Flagged-Order Volume:** Ensure daily flagged volume remains stable at ~22% and does not exceed the 25% call center shift capacity.
4. **Measure False Positives:** Track customer sentiment during confirmation calls to verify that calls feel like courteous VIP customer care rather than fulfillment friction.
5. **Compare Return Rates with a Control Group:** Maintain a 5% randomized holdout of high-risk orders released without a call to empirically establish true causal savings against called orders.
6. **Recalibrate Threshold Using Real Intervention Outcomes:** Adjust the operational threshold after 14 days of live data to maximize net savings based on observed call conversion rates.
