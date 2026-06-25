---

## 🚀 Featured Project: Loan Approval Decision Support System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

> Designed and deployed an end-to-end ML system to automate HELOC screening, replacing a fully manual bank review process.

### 🔧 What I Built
- Trained a **Decision Tree classifier** on ~10,000 historical loan applications
- Engineered missing-value indicators to preserve predictive signal without discarding incomplete records
- Identified the **4 most predictive features** out of 24 and built an interactive **Streamlit prototype** for loan officers
- Provided **rule-based denial explanations** so applicants understand exactly why they were rejected

### 📈 Business Impact
| Metric | Result |
|---|---|
| Accuracy on negative predictions | **84%** |
| Reduction in loan officer workload | **47%** |
| Applications processed in dataset | **~10,000** |

### 🧠 Model Selection Rationale
Evaluated 5 algorithms before choosing Decision Trees:

| Model | Rejected Because |
|---|---|
| Logistic Regression | Assumes linear relationships — too rigid for financial data |
| Random Forest | High accuracy but black-box — can't explain denials |
| XGBoost | Same interpretability problem |
| SVM | Expensive, limited transparency |
| K-Means | Unsupervised — can't justify individual decisions |

✅ **Decision Trees** were the only model satisfying all 4 constraints: automation, interpretability, accuracy, and regulatory auditability.

---

## 💼 What I Bring to a Team

- 🔍 Translating ambiguous business problems into structured ML solutions
- ⚖️ Balancing model performance with explainability — especially in regulated industries
- 🖥️ Building prototypes non-technical stakeholders can actually use
- 🔄 Thinking beyond the model: deployment, monitoring, and long-term maintenance
