Featured Project: Loan Approval Decision Support System
Tools: Python · Scikit-learn · Streamlit · Decision Trees · Pandas
Designed and deployed an end-to-end machine learning system to automate HELOC (Home Equity Line of Credit) screening for a simulated bank environment, replacing a fully manual review process.
What I built:

Trained a Decision Tree classifier on ~10,000 historical loan applications, handling missing data through mean imputation and missing-value indicator engineering
Achieved 84% accuracy on negative predictions — the highest-stakes class for credit risk
Built an interactive Streamlit prototype allowing loan officers to input applicant data and receive instant, explainable decisions
The system identified the 4 most predictive features out of 24 (External Risk Estimate, Average Minimum File, Number of Satisfactory Trades, Net Fraction Revolving Burden) and surfaced them as the core inputs

Business impact:

Reduced loan officer manual review workload by 47%, freeing staff for higher-value decisions
Provided applicants with transparent, rule-based denial explanations — critical for regulatory compliance (Fair Lending standards)
Designed for long-term reliability through periodic retraining, shadow testing, and real-time performance monitoring

Why Decision Trees over alternatives: Evaluated Logistic Regression, Random Forests, XGBoost, SVM, and K-Means before selecting Decision Trees — the only model that satisfied all four constraints: automation, interpretability, accuracy, and regulatory auditability.

What I Bring

Translating ambiguous business problems into structured ML solutions
Balancing model performance with explainability — especially in regulated industries
Building prototypes that non-technical stakeholders can actually use
Thinking beyond the model: deployment, monitoring, and long-term maintenance

