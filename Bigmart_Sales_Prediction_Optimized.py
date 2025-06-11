'''Author:Souptik Sarkar'''

import pandas as pd
import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Get current working directory (where your script is run)
cwd = os.getcwd()

# Join it with the folder name
folder_path = os.path.join(cwd, 'Data Sources')
#Load Data
train = pd.read_csv(os.path.join(folder_path,'train_v9rqX0R.csv'))
test = pd.read_csv(os.path.join(folder_path,"test_AbJTz2l.csv"))
# Submission
submission = pd.read_csv(os.path.join(folder_path,'sample_submission_8RXa3c6.csv'))
test['Item_Outlet_Sales'] = np.nan
data = pd.concat([train, test], axis=0)

# Impute missing values
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

# Normalize Item_Fat_Content
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})

# Feature Engineering
data['Outlet_Age'] = 2025 - data['Outlet_Establishment_Year']
data['Item_Visibility'] = data['Item_Visibility'].replace(0, data['Item_Visibility'].mean())
data['Item_MRP_bins'] = pd.cut(data['Item_MRP'], bins=[0, 70, 140, 200, 270], labels=[0, 1, 2, 3])
data['Item_MRP_bins'] = data['Item_MRP_bins'].astype(int)
# Target encoding
for col in ['Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']:
    mapping = train.groupby(col)['Item_Outlet_Sales'].mean()
    data[col + '_enc'] = data[col].map(mapping)

# Label Encoding
le = LabelEncoder()
for col in ['Item_Fat_Content', 'Item_Identifier']:
    data[col] = le.fit_transform(data[col])

# Final feature set
features = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Item_MRP_bins',
    'Item_Type_enc', 'Outlet_Identifier_enc', 'Outlet_Location_Type_enc',
    'Outlet_Type_enc', 'Outlet_Size_enc', 'Item_Fat_Content', 'Item_Identifier'
]

train_data = data[~data['Item_Outlet_Sales'].isnull()]
test_data = data[data['Item_Outlet_Sales'].isnull()]
X = train_data[features]
y = train_data['Item_Outlet_Sales']
X_test = test_data[features]

# Train LightGBM
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'verbose': -1
}

model_lgb = lgb.train(
    lgb_params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(200)
    ]
)

# Predict with LGB
preds_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

# Train XGBoost

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse'
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=200
)

preds_xgb = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration))

# Final prediction: average ensemble
final_preds = 0.5 * preds_lgb + 0.5 * preds_xgb

# Prepare submission
submission = test_data[['Item_Identifier', 'Outlet_Identifier']]
submission['Item_Outlet_Sales'] = final_preds
submission.to_csv("submission_optimized.csv", index=False)
print("Submission file saved as submission_optimized.csv")
