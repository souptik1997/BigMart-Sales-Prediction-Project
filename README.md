## BigMart-Sales-Prediction-Project
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. The aim is to build a predictive model and predict the sales of each product at a particular outlet.  Using this model, BigMart will try to understand the properties of products and outlets that play a key role in increasing sales.

# Objective
To build a highly accurate regression model that predicts the Item_Outlet_Sales for the BigMart dataset using advanced ensemble techniques, particularly stacking multiple base models and a meta-model.
Problem Understanding
The dataset comprises both categorical and numerical variables related to retail sales across various outlets and items. Challenges include:
•	Missing data
•	Inconsistent categorical encoding
•	High variance in item pricing
•	Non-linear relationships in sales behavior

# Data Preprocessing
To ensure high-quality model input:
•	Missing Values:
o	Item_Weight: Filled using item-wise mean.
o	Item_Visibility: Treated 0 as missing, imputed using median visibility per Item_Type.
o	Outlet_Size: Filled using mode per Outlet_Type.
•	Label Fixing: Standardized Item_Fat_Content values for consistency.
•	Feature Engineering:
o	Years_Since_Establishment = 2025 - Outlet_Establishment_Year
o	Item_Category from first 2 letters of Item_Identifier
o	MRP_Cluster: Binned using quartiles
o	Price_per_Weight = Item_MRP / Item_Weight

# Encoding and Final Features
•	Label encoding applied on categorical variables using LabelEncoder
•	Removed Item_Identifier and Outlet_Establishment_Year as they don't contribute to learning
•	Final dataset used for modeling consists of transformed and meaningful features

# Modeling Strategy: Stacking Ensemble
We applied 5-fold stacking using the following base learners:
1.	Random Forest Regressor
2.	XGBoost Regressor – with early_stopping_rounds=50 and validation monitoring
3.	LightGBM Regressor – similar setup to XGBoost
4.	CatBoost Regressor – trained with use_best_model=True and silent mode
Each base model was trained across 5 folds and generated Out-of-Fold (OOF) predictions on training and test sets. These predictions formed the input features for the meta-model.

# Meta-Model: Ridge Regressor
We used Ridge regression to learn from the base models' predictions:
•	Trained on OOF predictions
•	Regularization helps prevent overfitting

# Evaluation
•	Used RMSE as the metric via early stopping during base model training
•	Consistent folds ensured fair evaluation and generalization

# Submission
•	Final predictions from Ridge meta-model were written to stacked_submission.csv
•	Format preserved as per sample submission structure

