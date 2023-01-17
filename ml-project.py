# %% [code] {"execution":{"iopub.status.busy":"2023-01-16T15:01:22.938175Z","iopub.execute_input":"2023-01-16T15:01:22.939465Z","iopub.status.idle":"2023-01-16T15:01:22.951883Z","shell.execute_reply.started":"2023-01-16T15:01:22.939415Z","shell.execute_reply":"2023-01-16T15:01:22.950483Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-01-16T15:01:22.956667Z","iopub.execute_input":"2023-01-16T15:01:22.957038Z"}}
train_dataset_ = pd.read_feather('../input/amexfeather/train_data.ftr')
# Keep the latest statement features for each customer
training_dataset = train_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()

test_dataset_ = pd.read_feather('/kaggle/input/amexfeather/test_data.ftr')
# Keep the latest statement features for each customer
test_dataset = test_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()

# %% [code]
import gc
del train_dataset_
gc.collect()

# %% [code]
training_dataset.head()

# %% [code]

training_dataset_cp = training_dataset.copy()
test_dataset_cp = test_dataset.copy()

# %% [code]
# # Lable encoding for categoricals
# object_cols = []
# for colname in cat_cols.columns:
#     object_cols.append(colname)
#     training_dataset_cp[colname], _ = training_dataset_cp[colname].factorize()

# %% [code]
# Remove columns if there are > 80% of missing values
training_dataset_cp = training_dataset_cp.drop(['S_2','D_66','D_42','D_49','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)
# Remove columns if there are > 80% of missing values
test_dataset_cp = test_dataset_cp.drop(['S_2','D_66','D_42','D_49','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)

# %% [code]
# fill with median coulmns that has null values
selected_col = np.array(['P_2','S_3','B_2','D_41','D_43','B_3','D_44','D_45','D_46','D_48','D_50','D_53','S_7','D_56','S_9','B_6','B_8','D_52','P_3','D_54','D_55','B_13','D_59','D_61','B_15','D_62','B_16','B_17','D_77','B_19','B_20','D_69','B_22','D_70','D_72','D_74','R_7','B_25','B_26','D_78','D_79','D_80','B_27','D_81','R_12','D_82','D_105','S_27','D_83','R_14','D_84','D_86','R_20','B_33','D_89','D_91','S_22','S_23','S_24','S_25','S_26','D_102','D_103','D_104','D_107','B_37','R_27','D_109','D_112','B_40','D_113','D_115','D_118','D_119','D_121','D_122','D_123','D_124','D_125','D_128','D_129','B_41','D_130','D_131','D_133','D_139','D_140','D_141','D_143','D_144','D_145'])

# fill with median coulmns that has null values
for col in selected_col:
    test_dataset_cp[col] = test_dataset_cp[col].fillna(test_dataset_cp[col].median())
for col in selected_col:
    training_dataset_cp[col] = training_dataset_cp[col].fillna(training_dataset_cp[col].median())

# %% [code]
# drop unusable columns
selcted_col2 = np.array(['D_68','B_30','B_38','D_64','D_114','D_116','D_117','D_120','D_126'])

# drop unusable columns
for col2 in selcted_col2:
    test_dataset_cp[col2] =  test_dataset_cp[col2].fillna(test_dataset_cp[col2].mode()[0])
for col2 in selcted_col2:
    training_dataset_cp[col2] =  training_dataset_cp[col2].fillna(training_dataset_cp[col2].mode()[0])


# %% [code]
training_dataset_cp.head()

# %% [code]
train_target_cp = training_dataset_cp.pop("target")

# %% [code]
# Find the columns with categorical data
cat_cols = training_dataset_cp.select_dtypes(include=["category"])

# Print the names of the categorical columns
print(cat_cols.columns)

for colname in cat_cols.columns:
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(test_dataset_cp[colname], prefix="ohp")
    # Drop column B as it is now encoded
    test_dataset_cp = test_dataset_cp.drop(colname,axis = 1)
    # Join the encoded df
    test_dataset_cp = test_dataset_cp.join(one_hot)
    
for colname in cat_cols.columns:
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(training_dataset_cp[colname], prefix=colname)
    # Drop column B as it is now encoded
    training_dataset_cp = training_dataset_cp.drop(colname,axis = 1)
    # Join the encoded df
    training_dataset_cp = training_dataset_cp.join(one_hot)

# %% [code]
# Lable encoding for categoricals
object_cols = []
for colname in cat_cols.columns:
    object_cols.append(colname)
    training_dataset_cp[colname], _ = training_dataset_cp[colname].factorize()
for colname in cat_cols.columns:
    object_cols.append(colname)
    test_dataset_cp[colname], _ = test_dataset_cp[colname].factorize()

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(training_dataset_cp, train_target_cp, test_size=0.3,
    shuffle=True,stratify=train_target_cp,random_state=6)

scaler = StandardScaler()

scaler.fit(X_train)
scaler.transform(X_train)

# scaler.fit(y_train)
# scaler.transform(y_train)

scaler.fit(X_test)
scaler.transform(X_test)
X_train.head()

# %% [code]
# ensembling - high training time low perfromance
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and pre-process data
# X, y = load_and_preprocess_data()

# Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create individual models
log_reg = LogisticRegression()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Create ensemble model using majority voting
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('dt', decision_tree), ('rf', random_forest)], voting='soft')

# Fit ensemble model on training data
ensemble.fit(X_train, y_train)

# Evaluate ensemble model on test data
ensemble_accuracy = ensemble.score(X_test, y_test)
print("Ensemble model accuracy: ", ensemble_accuracy)

# %% [code]

lrclf = LogisticRegression(C=4,class_weight="balanced",dual=False,fit_intercept=True,
                           intercept_scaling=100,l1_ratio=None,max_iter=5,multi_class='auto',
                           n_jobs=None,penalty='l2',random_state=None,solver='lbfgs',tol=0.0001,
                           verbose=0,warm_start=False)

pred = lrclf.fit(X_train, y_train).predict_proba(X_test)

print(roc_auc_score(y_test, pred[:, 1])) # new

# %% [code] {"scrolled":true}
import lightgbm as lgb
import optuna

def hyperparameter_tuning(trial):
    # Define the hyperparameters to be tuned
    params = {
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0)
    }

    # Create the LightGBM model
    model = lgb.LGBMClassifier(**params)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the target values for the validation data
    y_pred = model.predict_proba(X_test)

    # Calculate the evaluation metric
    accuracy = roc_auc_score(y_test, y_pred[:, 1])

    # Report the evaluation metric to Optuna
    return -accuracy


# # Create the Optuna study
# study = optuna.create_study()

# # Run the hyperparameter tuning using Optuna's optimize function
# study.optimize(hyperparameter_tuning, n_trials=100)

# # Print the best hyperparameter values and evaluation metric
# print('Best hyperparameters:', study.best_params)
# print('Best accuracy:', study.best_value)

# %% [code]
# tuned hyperparameters
params = {
'learning_rate': 0.08395719054553268, 'max_depth': 9, 'num_leaves': 62, 'feature_fraction': 0.5354486758502452
}

# Create the LightGBM model
model_lgbm = lgb.LGBMClassifier(**params)

# Train the model on the training data
model_lgbm.fit(X_train, y_train)

# Predict the target values for the validation data
y_pred = model_lgbm.predict_proba(X_test)

print(roc_auc_score(y_test, y_pred[:, 1])) # new

# %% [code]
# SVM - not enough perfromance and high training time
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def objective(trial):
    # extract the hyperparameters
    C = trial.suggest_float('C', 0.1, 10)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
    gamma = trial.suggest_float('gamma', 0.1, 10)
    # create the SVM model
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    # use cross-validation to evaluate the model
    return -1.0 * cross_val_score(svm, X_train, y_train, cv=5).mean()

study = optuna.create_study()
study.optimize(objective, n_trials=100)

# print the best parameters
print("Best params: ", study.best_params)

# %% [code]
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {'objective': 'binary','n_estimators': 1200,'metric': 'binary_logloss','boosting': 'gbdt','num_leaves': 90,'reg_lambda' : 50,'colsample_bytree': 0.19,'learning_rate': 0.03,'min_child_samples': 2400,'max_bins': 511,'seed': 42,'verbose': -1}

# trained model with 100 iterations
model_lgb = lgb.train(params, d_train, 100)

# %% [code] {"scrolled":true}
from catboost import CatBoostClassifier


model_cboost = CatBoostClassifier(iterations=1000,
                           task_type="CPU",
                           devices='0:1')
model_cboost.fit(X_train,
          y_train,
          verbose=True)


# %% [code]
y_pred = model_cboost.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1])) # new

# %% [code]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

KNN = KNeighborsClassifier()
clf = KNN.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
# not enough performance

# %% [code]
# filling null values with mode in test_dataset
columnsWithNa = test_dataset_cp.columns[test_dataset_cp.isnull().any()].tolist()
for column in columnsWithNa :
    test_dataset_cp[column].fillna(test_dataset_cp[column].mode()[0], inplace = True)
    

# %% [code]
# pred = pd.DataFrame({"prediction": lrclf.predict_proba(test_dataset_cp)[:, 1]}, index=test_dataset_cp.index)
# # del y_test['customer_ID']
# pred.to_csv("submission.csv")
# pred.head()

# %% [code]
# predictions = ensemble.predict_proba(test_dataset_cp)

# %% [code]
# get presictions for submission
predictions = model_cboost.predict_proba(test_dataset_cp)
print(predictions)

# %% [code]
output = pd.DataFrame({'customer_ID': test_dataset_cp.index, 'prediction': predictions[:,1]})
output.to_csv('submission.csv', index=False)