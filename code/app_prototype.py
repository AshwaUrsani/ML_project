import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline
import numpy as np


file_path = "/Users/ashwaursani/Documents/Machine_Learning/Project/heloc_dataset_v1.csv"

df = pd.read_csv(file_path)
print(len(df))


df_without_missing_rows = pd.DataFrame(columns = df.columns)


print(df_without_missing_rows)

numeric_cols = df.select_dtypes(include='number').columns
mask_all_numeric_missing = df[numeric_cols].eq(-9).all(axis=1)
n_rows_all_numeric_missing = mask_all_numeric_missing.sum()
df_without_missing_rows = df[~mask_all_numeric_missing].copy()

#print(df_without_missing_rows)
print(len(df_without_missing_rows))

X = df.iloc[:, 1:]
#print(X)
Y = df["RiskPerformance"].copy()

Good_loan_count = 0
Bad_loan_count = 0

#print(Y)
#0 is bad, 1 is good
for i in range(len(Y)):
    if Y[i] == 'Bad':
        Y[i] = 0
        Bad_loan_count += 1
    else:
        Y[i] = 1
        Good_loan_count += 1

print(Good_loan_count)
print(Bad_loan_count)
#print(Y)


print
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

X_train = X_train[X_train["ExternalRiskEstimate"] != -9]
Y_train = Y_train.loc[X_train.index]

X_test = X_test[X_test["ExternalRiskEstimate"] != -9]
Y_test = Y_test.loc[X_test.index]



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion

do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])

pipeline = Pipeline([("expand features", feature_expansion),
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])

arr_X_train_t = pipeline.fit_transform(X_train)

minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_train)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_train)


col_names_7 = X_train.columns.values[minus_7_indicator_transformer.features_].tolist()
col_names_7 = list(map(lambda s:str(s)+'=-7',col_names_7))
col_names_8 = X_train.columns.values[minus_8_indicator_transformer.features_].tolist()
col_names_8 = list(map(lambda s:str(s)+'=-8',col_names_8))
column_names = X_train.columns.values.tolist() + col_names_7 + col_names_8


print(column_names)
X_train_t = pd.DataFrame(arr_X_train_t, columns=column_names)


X_test_t = pipeline.transform(X_test)
X_test_t = pd.DataFrame(X_test_t, columns=column_names)
Y_train_t = Y_train

X_train_t_tr, X_train_t_val, Y_train_t_tr, Y_train_t_val = train_test_split(X_train_t, Y_train_t, test_size=0.25, random_state=1234)

Y_train = Y_train.astype(int)
Y_train_t_tr = Y_train_t_tr.astype(int)
X_train_t_tr = X_train_t_tr.astype(float)
X_train_t_val = X_train_t_val.astype(float)
Y_train_t_val = Y_train_t_val.astype(int)


from sklearn import tree
from sklearn.model_selection import cross_validate

clf_tree = tree.DecisionTreeClassifier().fit(X_train_t_tr, Y_train_t_tr)
cv_results_tree = cross_validate(tree.DecisionTreeClassifier(), X_train_t, Y_train, cv=5, return_estimator=True)
#cv_results_tree


#Hyperparameter tuning - most parameters
# '''
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, recall_score

# # Define the model
# clf_tree = tree.DecisionTreeClassifier()

# # Define the hyperparameters to tune
# param_grid = {
#     'max_depth': [1, 2, 3, 4, 5],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 5],
#     'criterion': ['gini', 'entropy'],
#     'class_weight': [None, 'balanced']
# }

# # Use negative pas the scoring metric to minimize false negatives
# scorer = make_scorer(recall_score, greater_is_better=True)

# # Perform GridSearchCV
# grid_search = GridSearchCV(clf_tree, param_grid, scoring=scorer, cv=5)
# grid_search.fit(X_train_t, Y_train)  # Replace X_train, y_train with your dataset

# # Print best parameters and best recall score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Recall Score:", grid_search.best_score_)

# # Use the best model
# best_model = grid_search.best_estimator_
# clf_tree = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=1, criterion='entropy', class_weight='balanced').fit(X_train_t_tr, Y_train_t_tr)
# '''



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score
import numpy as np


def negative_predictive_value(y_true, y_pred):
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 1) & (y_true == 0))
    npv = tn / (tn + fn)
    return npv


def custom_score(y_true, y_pred):
    # Compute Accuracy
    acc = accuracy_score(y_true, y_pred)
    return acc + negative_predictive_value(y_true, y_pred)



# Define the model

clf_tree = tree.DecisionTreeClassifier()

# Define the hyperparameters to tune

param_grid = {
    'max_depth': [3,4,5],
    'min_samples_split': [1,2],
    'min_samples_leaf': [1,2],
    'criterion': ['gini', 'entropy'],
    'class_weight': [{0: 1, 1: 1.9}, {0: 1, 1: 2}, {0: 1, 1: 1.85}, {0: 1, 1 : 1.95} ]
}


#Use recall as the scoring metric to minimize false negatives
scorer = make_scorer(custom_score, greater_is_better=True)

#Perform GridSearchCV
grid_search = GridSearchCV(clf_tree, param_grid, scoring=scorer, cv=5)
grid_search.fit(X_train_t, Y_train)  # Replace X_train, y_train with your dataset

#Print best parameters and best recall score
print("Best Parameters:", grid_search.best_params_)
print("Best Acc + Recall Score:", grid_search.best_score_)

#Use the best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
#clf_tree = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=1, min_samples_split=2, criterion='gini', class_weight={0: 1, 1: 2}).fit(X_train_t_tr, Y_train_t_tr)



from sklearn.metrics import confusion_matrix

#clf_tree = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=1, min_samples_split=2, criterion='gini', class_weight={0: 1, 1: 2}).fit(X_train_t_tr, Y_train_t_tr)
clf_tree = tree.DecisionTreeClassifier(**grid_search.best_params_).fit(X_train_t_tr, Y_train_t_tr)

y_pred_tree = best_model.predict(X_train_t_val)
conf_matrix = confusion_matrix(Y_train_t_val, y_pred_tree)
print(conf_matrix)

tp, fp, fn, tn = conf_matrix.ravel()

total = tp + fp + fn + tn
accuracy = (tp + tn) / (tp + tn + fp + fn)
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
precision = tp / (tp + fp)
appr_rate = (tp + fp) / (tp + tn + fp + fn)
npv = tn / (tn + fn)
print("accuracy: " + str(accuracy))
print("tnr: " + str(npv))
print("approval rating: " + str(appr_rate))
print("test size: " + str(total))


import matplotlib.pyplot as plt
tree.plot_tree(clf_tree, feature_names=column_names, filled=True)
plt.savefig("tree_3.png", dpi=1000)




# Streamlit App
st.title("📊 HELOC Risk Classification using Decision Tree")

# Load dataset
st.subheader("Dataset Preview")
st.write(df.head())


# Data Preprocessing
st.subheader("Data Cleaning and Preprocessing")
st.write("Dataset after removing missing values:", df_without_missing_rows)

# Decision Tree
st.subheader("Decision Tree Visualization")


fig, ax = plt.subplots(figsize=(20, 15))
plot_tree(clf_tree, feature_names=column_names, class_names=['Bad', 'Good'], filled=True, fontsize=8, ax=ax)
st.pyplot(fig)

# Display Metrics
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"NPV (Negativity Predictive Value): {npv:.2f}")
st.write(f"Approval Rating: {appr_rate:.2f}")
st.write(f"Test Size: {total}")

# Collect user input - using selected columns
st.subheader("🔍 Predict HELOC Risk for a New Applicant")

selected_columns = [
    "ExternalRiskEstimate", 
    "AverageMInFile",
    "NumSatisfactoryTrades", 
    "NetFractionRevolvingBurden"
]  

formal_names = {
    "ExternalRiskEstimate": "External Risk Estimate: Consolidated version of risk markers",
    "AverageMInFile": "Average Months in File",
    "NumSatisfactoryTrades": "Number Satisfactory Trades",
    "NetFractionRevolvingBurden": "Net Fraction Revolving Burden"}

full_input = {}
for col in X_train.columns:
    # Use mean as default value
    full_input[col] = X_train[col].mean()

st.write("Please enter values for the following fields:")
for col in selected_columns:
    display_name = formal_names.get(col, col) 
    full_input[col] = st.number_input(f"Enter {display_name}", value=float(X_train[col].mean()))

user_df = pd.DataFrame([full_input], columns=X_train.columns)

user_df_t = pipeline.transform(user_df)
user_df_t = pd.DataFrame(user_df_t, columns=column_names)
node_indicator = clf_tree.decision_path(user_df_t)
leaf_index = node_indicator.indices[-1] if len(node_indicator.indices) > 0 else None
terminal_node = clf_tree.apply(user_df_t)


Risk_Reason_A = "Need a longer credit history"
Risk_Reason_B = "Need a longer history of successful payments"
Risk_Reason_C = "Lower spending relative to credit limit"

terminal_node = clf_tree.apply(user_df_t)

# Predict based on user input
if st.button("🔮 Predict Risk"):
    prediction = best_model.predict(user_df_t)
    if prediction[0] == 1:
      prediction_label = "Good Loan" 
    else:
      prediction_label = "Bad Loan"
      if prediction_label == "Bad Loan":
        if terminal_node[0] == 4:
          st.write(Risk_Reason_A)
          st.write(Risk_Reason_B)
        elif terminal_node[0] == 5:
          st.write(Risk_Reason_A)
        elif terminal_node[0] == 8:
          st.write(Risk_Reason_C)
        elif terminal_node[0] == 11:
          st.write(Risk_Reason_B)
        elif terminal_node[0] == 12:
          st.write(Risk_Reason_B)
        elif terminal_node[0] == 14:
          st.write(Risk_Reason_A)
      
    # Display Prediction
    st.write(f"### 🏦 Predicted HELOC Risk: **{prediction_label}**")


