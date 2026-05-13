# 🏦 HELOC Loan Risk Classifier

> A machine learning decision support system for automating Home Equity Line of Credit (HELOC) application screening using Decision Trees.

**Course:** CIS432 — Machine Learning for Business Analytics | Spring A 2025  
**Institution:** Simon Business School, University of Rochester  
**Team:** Ashwa Ursani, Sara Alejandra Ramirez Tamez, Kweku Ofori, Daniel Maass

---

## 📌 Project Overview

Simon Bank of Rochester processes HELOC applications manually — a time-consuming task for loan officers. This project builds a predictive ML model and interactive decision support system (DSS) to automate the initial screening of applications.

The model classifies each application as:
- ✅ **Approved** → forwarded to a loan officer for final review
- ❌ **Denied** → with a clear explanation and guidance for future improvement

**Key Results:**
- 🎯 68% overall accuracy
- 🎯 84% accuracy for negative (denial) predictions
- ⚡ 47% reduction in loan officer workload

---

## 📁 Repository Structure

```
heloc-loan-risk-classifier/
│
├── README.md
│
├── data/
│   ├── heloc_dataset_v1.csv            # ~10,000 historical HELOC applications
│   └── heloc_data_dictionary-2.xlsx    # Feature descriptions and definitions
│
├── src/
│   ├── ML_for_Business_ProjectV2.py    # Core ML pipeline and model training
│   └── prototype_final.py             # Interactive Streamlit web application
│
├── docs/
│   └── Project_Report.pdf             # Full project report
│
└── requirements.txt                   # Python dependencies
```

---

## 🧠 Model & Methodology

**Algorithm:** Decision Tree Classifier (scikit-learn)

A Decision Tree was selected over alternatives (Logistic Regression, Random Forest, XGBoost, SVM) because it best satisfies four key requirements:

| Requirement | Why Decision Tree? |
|---|---|
| Automation | Real-time evaluation of applications |
| Interpretability | Clear rule-based decision paths |
| Accuracy | Strong classification performance |
| Regulatory Compliance | Transparent, auditable decisions |

**Top Predictive Features:**
- `ExternalRiskEstimate` — Consolidated risk score
- `AverageMInFile` — Average months in credit file
- `NumSatisfactoryTrades` — Number of satisfactory trades
- `NetFractionRevolvingBurden` — Revolving credit utilization

---

## 🔧 Data Preprocessing

- Original dataset: **10,459 entries, 24 features**
- Removed 588 rows with entirely missing values → **9,871 usable rows**
- Missing values (`-7`, `-8`) handled via mean imputation + missing indicator columns
- Binary target: `RiskPerformance` → `Good (1)` / `Bad (0)`

---

## 🖥️ Streamlit App

An interactive prototype was built with Streamlit allowing users to:
- Preview the dataset
- Visualize the trained decision tree
- Input 4 key features and get an instant loan prediction
- Receive a reason for denial (if applicable)

**To run the app locally:**

```bash
# 1. Clone the repository
git clone https://github.com/AshwaUrsani/ML_project.git.git
cd heloc-loan-risk-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update the file path in prototype_final.py to your local dataset path

# 4. Run the app
streamlit run src/prototype_final.py
```

---

## 📦 Requirements

```
streamlit
pandas
matplotlib
scikit-learn
numpy
openpyxl
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Overall Accuracy | 68% |
| Negative Predictive Value (NPV) | 84% |
| Workload Reduction | 47% |

---

## 🔮 Future Improvements

- Periodic model retraining on new data
- Shadow testing / parallel deployment
- Adaptive decision thresholds for regulatory changes
- Real-time monitoring with automated alerts

---

## 📄 License

This project was developed for academic purposes at the University of Rochester — Simon Business School.
