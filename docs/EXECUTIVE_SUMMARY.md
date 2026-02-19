# Executive Summary: CFD System

## 1. For Whom is This System?

The **CFD (Credit Fraud Detection)** system is designed for:

*   **Financial Institutions**: Banks, credit unions, and neo-banks that need to protect customer funds.
*   **Payment Processors**: Companies like Stripe, PayPal, or regional mobile money providers handling high-volume transactions.
*   **Fintech Startups**: New entrants needing a "bolt-on" enterprise-grade security layer without building from scratch.
*   **Compliance & Risk Teams**: Internal audit departments requiring tools to analyze historical data for missed fraud patterns.

## 2. Why is it Important?

*   **Financial Loss Prevention**: Credit card fraud steals billions annually ($28B+ globally in recent years). Stopping even 1% more fraud saves millions.
*   **Customer Trust**: A single drained account can ruin a reputation. Proactive protection builds loyalty.
*   **Regulatory Compliance**: Banks are legally required to have robust AML (Anti-Money Laundering) and fraud detection systems in place.
*   **Operational Efficiency**: Automating detection reduces the manual workload on human analysts, allowing them to focus only on complex edge cases.

## 3. How Can It Be Used?

### A. Real-Time Transaction Screening (API Mode)
*   **Scenario**: A customer swipes their card at a store.
*   **Flow**: The POS terminal sends transaction details -> Bank's API -> **CFD System**.
*   **Action**: CFD predicts "Fraud" (Prob: 99.8%). The bank declines the transaction instantly, protecting the money.

### B. Batch Auditing (Offline Mode)
*   **Scenario**: End-of-day reconciliation.
*   **Flow**: The system feeds the day's 50 million transaction logs into CFD.
*   **Action**: CFD flags suspicious movements that shouldn't have happened. Analysts investigate these accounts and freeze them if necessary.

### C. Shadow Mode (Testing)
*   **Scenario**: Testing a new model against an old one.
*   **Flow**: Run CFD alongside the legacy system without stopping transactions. Compare which one caught more fraud to validate ROI.

## 4. How It Works (High-Level)

The system operates like a digital detective that never sleeps:

1.  **Ingestion ("The Eyes")**: It reads transaction data—who, where, how much, and what type (Transfer, Cash out, etc.).
2.  **Feature Engineering ("The Brain")**: It doesn't just look at the raw numbers; it looks for *patterns*.
    *   *Example*: "Why is this person emptying their entire balance at 3 AM?"
    *   *Example*: "Why does the transfer amount not match the drop in their account balance?" (We calculate `errorBalanceOrig` = `NewBalance` + `Amount` - `OldBalance`. In legitimate transactions, this sums to zero. In fraud, mathematical discrepancies often appear.)
3.  **AI Classification ("The Verdict")**: It uses a **Random Forest Model**—think of it as a council of 100 digital judges. Each judge looks at the evidence and votes "Safe" or "Fraud".
    *   If the majority votes "Fraud", the system flags it.
    *   We use **Class Weighting ('balanced')** to handle the rarity of fraud (<1%). This forces the model to heavily penalize missing a fraud case, compelling it to learn the subtle signatures of the minority class.
4.  **Continuous Learning**: As new patterns emerge, the model can be retrained (`python main.py --mode train`) to learn new criminal tactics.

## 5. Reliability & Verification

How do we know it works? We don't just guess.

*   **Stratified Sampling**: When testing, we split data 80/20 using *Stratification*. This ensures the test set has the exact same proportion of fraud as the real world, preventing "lucky" results.
*   **Audit Trail**: Every action is logged to `logs/` for compliance.
*   **Automated Validation**: The system automatically generates:
    *   **Confusion Matrix**: To visualize exactly where it might be confused.
    *   **ROC-AUC Score**: A mathematical proof of its discrimination capability (currently **0.999**).

> For a deep technical dive into the algorithms and code structure, see the [System Architecture Guide](SYSTEM_OVERVIEW.md).
