


Machine Learning Decision Support System (DSS) for Credit Evaluation
Developed an end-to-end Machine Learning Decision Support System (DSS) to automate the preliminary screening and risk assessment of loan applications, successfully shifting operations from a manual review process to an automated pipeline. 
For full methodology, data pipeline details, and performance metrics, please refer to the primary verification file: "Loan Approval ML Report.pdf". 
📈 Core Business Impact & Results
•	47% Workload Reduction: Automated the initial evaluation tier, cutting manual review volume for risk officers by nearly half and optimizing human resource allocation. 
•	High-Precision Risk Filtering: Achieved an overall classification accuracy of 68%, featuring an 84% accuracy rate specifically for identifying and denying high-risk applications. 
•	Regulatory Compliance: Utilized an interpretable architecture to ensure full compliance with financial auditing standards, generating clear, rule-based reasons for loan denials to provide transparency to applicants. 
🛠️ Technical Stack & Implementation
•	Predictive Modeling: Evaluated multiple machine learning algorithms (including Logistic Regression, Random Forests, XGBoost, and SVM). Selected a Decision Tree Classifier to perfectly balance predictive accuracy with structural interpretability and auditability. 
•	Feature Engineering & Preprocessing: Executed data cleaning and missing value engineering on a historical dataset of ~10,000 credit profiles. Implemented mean imputation paired with original missing data indicator flags to preserve hidden predictive patterns in incomplete records. 
•	Interactive Cloud Prototype: Built and deployed an interactive dashboard using Streamlit. To optimize user experience and prevent form-entry fatigue, the interface was designed to focus strictly on the top 4 highest-importance features output by the model (External Risk Estimate, Average Minimum File, Number of Satisfactory Trades, and Net Fraction Revolving Burden). 
🔒 Production Governance & MLOps Strategies
To mitigate model deterioration and performance drift over time in a live banking environment, the system design incorporates four production-stage strategies: 
1.	Periodic Model Retraining: Standard automated pipelines to regularly refresh the model architecture on fresh financial quarters. 
2.	Shadow Testing Frameworks: Deploying updated model variations in a parallel, non-breaking "shadow mode" to validate live performance variations against production benchmarks. 
3.	Adaptive Thresholding: Giving risk management teams the ability to manually tune decision rules in real-time response to macroeconomic or regulatory shifts. 
4.	Continuous Metric Monitoring: Setting up automated, real-time alerts on key statistical metrics (Accuracy, Precision, Recall, and F1-Score) to catch operational drift immediately. 

<img width="468" height="626" alt="image" src="https://github.com/user-attachments/assets/02f34de9-5d57-4f09-9905-5fa6f36bdb04" />
