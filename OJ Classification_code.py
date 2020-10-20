#!/usr/bin/env python
# coding: utf-8

# In[1]:

# ## IMPORT LIBRARIES AND SET SYSTEM OPTIONS

# In[75]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.display import DisplayObject, display


# In[401]:


import pandas as pd
pd.set_option('display.max_rows', 270000)
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:.5f}'.format

import os

import pandas_profiling
from pandas_profiling import ProfileReport

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc

import time


# ## Read the data

# In[78]:


# Read in data from personal Github
df_org = pd.read_csv("https://raw.githubusercontent.com/LDang47/Orange-juice_Supermarket-data/master/OJ.csv")

# print info
df_org.shape
df_org.columns
df_org.head()
df_org.dtypes


# ## Preprocessing Data

# In[109]:


# Make a copy of df_org
df = df_org.copy()


# In[110]:


# Check for Nulls and missing value
df.isnull().sum()

# The data has no missing value


# In[111]:


# Imbalance dataset check: Number of instances in each class
df["Purchase"].value_counts()

# The dataset is relatively balance, no need treatment for imbalance dataset


# In[107]:


# Have a look at summary statistics of the entire dataset 
profile = pandas_profiling.ProfileReport(df, html={'style':{'full_width':True}}, minimal=True)
profile.to_notebook_iframe()


# In[113]:


# Change format of target column 'Purchase' to CH = 1 and MM= 0
df['Purchase'] = df['Purchase'].apply(lambda x:1 if x == 'CH' else 0)


# In[119]:


# OHE all categorical columns: Store7
cat_cols = list(df.select_dtypes(include=['object']).columns) 

df = pd.get_dummies(data=df, columns=cat_cols)
df


# In[129]:


# Define column in the dataset for easy tracking later on
Id_col = 'Unnamed: 0'
target_col = 'Purchase'
df.info()
df.head()


# ### Data Exploration: Plot

# In[56]:


# Plot correlation maxtrix using sns heatmap
plt.figure(figsize=(25, 25))
df_1 = df.loc[:, df.columns != 'Unnamed: 0']

corrMatrix = df_1.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[47]:


# Create a series of pair plot between all features
plt.figure(figsize=(25, 25))
sns.pairplot(df, vars = ['Purchase', 'WeekofPurchase', 'StoreID', 'PriceCH',
       'PriceMM', 'DiscCH', 'DiscMM', 'SpecialCH', 'SpecialMM', 'LoyalCH',
       'SalePriceMM', 'SalePriceCH', 'PriceDiff', 'PctDiscMM',
       'PctDiscCH', 'ListPriceDiff', 'STORE'])
plt.show()


# ## MODELLING

# In[137]:


# Split the data in to train and test with ratio 80:20
X = df.drop([Id_col, target_col], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print('Y (train) counts:')
print(y_train.value_counts())
print('Y (test) counts:')
print(y_test.value_counts())


# In[138]:


# Looing at the structure and content of the training file
X.info()
X.shape
X.head()

X_train.info()
X_train.shape
X_train.head()


# ## MODEL 1: LOGISTIC REGRESSION

# In[163]:


# Grid search with 10 cross validation for Logisitic regression model

# List of hyperparameters for tuning
grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}# l1 lasso l2 ridge

# Run the GridSearch on the X_train and y_train data
LR = LogisticRegression(solver='liblinear')
LR_cv = GridSearchCV(LR, grid, cv = 10) # 10 cross validation
LR_cv.fit(X_train, y_train)

# Print the model hyperparameter set with the best accuracy
print("tuned hpyerparameters :(best parameters) ",LR_cv.best_params_)
print("accuracy :",LR_cv.best_score_)


# In[172]:


# Run the best Logisitc regression with {'C': 1000, 'penalty': 'l1'}
LRM = LogisticRegression(solver='liblinear', random_state=0, C = 1000, penalty = 'l1')


# In[173]:


# Fit the train data into the LRM model
LRM.fit(X_train, y_train)


# In[175]:


# The intercept and coefficients of the LRM model
LRM.intercept_

LRM.coef_


# ### MODEL EVALUATION: USING THRESHOLD 0.5

# In[362]:


# Model evaluation on the TEST test: default threshold = 0.5
confusion_matrix(y_test, LRM.predict(X_test))

print(classification_report(y_test, LRM.predict(X_test)))

# True positive is 63
# True negative is 119
# False positive is 19
# False negative is 13

# AUC score
auc_score = roc_auc_score(y_test,LRM.predict(X_test))
round(float(auc_score ), 2 )


# In[365]:


# Other performance metrics
print("Accuracy = {:.2f}".format(accuracy_score(y_test, LRM.predict(X_test))))
print("F1 Score = {:.2f}".format(f1_score(y_test, LRM.predict(X_test))))


# In[344]:


# Plot the ROC Curve for AUC score = 0.835
fpr, tpr, threshold = roc_curve(y_test,LRM.predict(X_test),drop_intermediate=False)
roc_auc = metrics.auc(fpr, tpr)
 
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label='ROC curve (area = %0.2f)' % auc_score)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ### CLASSIFICATION USING DIFFERENT THRESHOLD: 0.6

# In[326]:


# Classification using different threshold: 0.6 (ratio of 1:0 in the training dataset)
predict_proba = pd.DataFrame(LRM.predict_proba(X_test))

predictions = predict_proba.loc[: , 1].map( lambda x: 1 if x >= 0.6 else 0 )
predictions.head()


# In[363]:


# Model evaluation on the TEST test using new threshold: 0.6
confusion_matrix(y_test, predictions)

print(classification_report(y_test, predictions))

# True positive is 69
# True negative is 113
# False positive is 13
# False negative is 19

# AUC score
auc_score1 = roc_auc_score(y_test,predictions)
round(float( auc_score1 ), 2 )


# In[364]:


# Other performance metrics
print("Accuracy = {:.2f}".format(accuracy_score(y_test, predictions)))
print("F1 Score = {:.2f}".format(f1_score(y_test, predictions)))


# In[343]:


# Plot the ROC Curve for AUC score = 0.85
fpr, tpr, threshold = roc_curve(y_test, predictions, drop_intermediate=False)
roc_auc = metrics.auc(fpr, tpr)
 
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label='ROC curve (area = %0.2f)' % auc_score1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ## MODEL 2: DECISION TREE

# In[353]:


# Grid search for Decision tree model

# List of hyperparameters for tuning
grid = {'criterion':('gini', 'entropy'), 'max_depth':[2, 4, 6, 8, 10], 'min_samples_split':[2, 10, 50], 'min_samples_leaf':[1, 5, 10],
             'max_features':[None, 'auto'], 'max_leaf_nodes':[None, 5, 10, 50], 'min_impurity_decrease':[0, 0.1, 0.2]}

# Run the GridSearch on the X_train and y_train data
DT = DecisionTreeClassifier(random_state=42)
DT_cv = GridSearchCV(DT, grid, cv = 10, scoring='roc_auc') # 10 cross validation
DT_cv.fit(X_train, y_train)

# Print the summary stats on best model hyperparameter set
print("tuned hpyerparameters :(best parameters) ",DT_cv.best_params_)
print("roc_auc :",DT_cv.best_score_)


# In[355]:


# Run Decision tree algorithm with the best set of hyperparameters: 
# {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2}
DTM = DecisionTreeClassifier(criterion = 'gini', max_depth = 6, max_features = None, max_leaf_nodes = 10, min_impurity_decrease = 0,
                             min_samples_leaf = 1, min_samples_split = 50, random_state = 42)


# In[356]:


# Fit the train data into the DT model
DTM.fit(X_train, y_train)


# ### MODEL EVALUATION

# In[361]:


# Model evaluation on the TEST test
confusion_matrix(y_test, DTM.predict(X_test))

print(classification_report(y_test, DTM.predict(X_test)))

# True positive is 68
# True negative is 102
# False positive is 14
# False negative is 30

# AUC score
auc_score_dt = roc_auc_score(y_test, DTM.predict(X_test))
round(float( auc_score_dt ), 2 )


# In[359]:


# Other performance metrics
y_pred_dt = DTM.predict(X_test)
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_dt)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_dt)))


# In[366]:


# Plot the ROC Curve for AUC score = 0.849
fpr, tpr, threshold = roc_curve(y_test, y_pred_dt, drop_intermediate=False)
roc_auc = metrics.auc(fpr, tpr)
 
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label='ROC curve (area = %0.2f)' % auc_score_dt)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[396]:


# Plot the acual Decision Tree
plt.figure(figsize=(25, 25))

feature_names = X_train.columns
class_names = [str(x) for x in DTM.classes_]

plt.figure(figsize=(12, 7));
plot_tree(DTM, filled=True, feature_names=feature_names, class_names=class_names, proportion=False, fontsize=8);


# ## MODEL 3: XGBOOST

# In[407]:


# Grid search for XGBoost model

# List of hyperparameters for tuning
grid = {'max_depth': range (2, 10, 1),
    'n_estimators': range(50, 500, 50),
    'learning_rate': [0.1, 0.01, 0.05], 
    'colsample_bytree': [0.7, 0.8],
    'subsample': [0.7, 0.8]}

# Run the GridSearch on the X_train and y_train data
XGB = XGBClassifier(objective= 'binary:logistic', nthread=4, seed=42)
XGB_cv = GridSearchCV(estimator=XGB,
                     param_grid=grid,
                     scoring = 'roc_auc',
                     n_jobs = 10,
                     cv = 10, # 10 cross validation
                     verbose=True) 
XGB_cv.fit(X_train, y_train)

# Print the summary stats on best model hyperparameter set
print("tuned hpyerparameters :(best parameters) ",XGB_cv.best_params_)
print("roc_auc :",XGB_cv.best_score_)


# In[410]:


# Run XGBoost algorithm with the best set of hyperparameters: 
# {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 60}
XGB = XGBClassifier(objective= 'binary:logistic', learning_rate = 0.1, max_depth = 2, n_estimators= 100,
                    colsample_bytree = 0.8, subsample = 0.7,nthread=4, seed=42, n_jobs = 10, cv = 10, verbose=True)


# In[412]:


# Fit the train data into the DT model
XGB.fit(X_train, y_train)


# ### MODEL EVALUATION

# In[413]:


# Model evaluation on the TEST test
confusion_matrix(y_test, XGB.predict(X_test))

print(classification_report(y_test, XGB.predict(X_test)))

# True positive is 63
# True negative is 120
# False positive is 19
# False negative is 12

# AUC score
auc_score_xgb = roc_auc_score(y_test, XGB.predict(X_test))
round(float( auc_score_xgb ), 2 )


# In[414]:


# Other performance metrics
y_pred_xgb = XGB.predict(X_test)
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_xgb)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_xgb)))


# In[415]:


# Plot the ROC Curve for AUC score = 0.84
fpr, tpr, threshold = roc_curve(y_test, y_pred_xgb, drop_intermediate=False)
roc_auc = metrics.auc(fpr, tpr)
 
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label='ROC curve (area = %0.2f)' % auc_score_xgb)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[416]:


# Plot Validation curve (based on auc) for XGBoost for different learning rate
viz = ValidationCurve(XGBClassifier(n_estimators=100, max_depth=3, random_state=0), param_name="learning_rate", param_range=np.linspace(0.001,2,30), cv=5, scoring="roc_auc")
get_ipython().run_line_magic('time', 'viz.fit(X, y)')
viz.poof()


# In[418]:


# Plot Validation curve (based on auc) for XGBoost for different n_estimators
viz = ValidationCurve(XGBClassifier(learning_rate=1.0, max_depth=3, random_state=0), param_name="n_estimators", param_range=np.arange(1, 100), cv=5, scoring="roc_auc")
get_ipython().run_line_magic('time', 'viz.fit(X, y)')
viz.poof()


# In[ ]:




