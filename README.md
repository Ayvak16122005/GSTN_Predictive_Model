# GSTN_Predictive_Model
Random Forest Classifier for GST Classification


# Title: Random Forest Classifier for GST Data Classification
# Author: KAVYA T
# Team ID: GSTN_623
# Generated Key: 579b464db66ec23bdd000001b1b0ef1507914d9e40bf5212c6b640a3
```words
Description:
This code trains and evaluates a Random Forest classifier on GST data 
with preprocessed features. The model predicts whether a transaction 
belongs to class 0 or 1 based on 21 features. The accuracy and 
classification metrics are displayed along with the feature importance.

Steps:
1. Data loading and merging (features and target).
2. Data preprocessing (missing values handling and scaling).
3. Model training using Random Forest.
4. Model evaluation and feature importance analysis.
```

# CODE:
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
X_train = pd.read_csv('/content/X_Train_Data_Input.csv')
Y_train = pd.read_csv('/content/Y_Train_Data_Target.csv')
X_test = pd.read_csv('/content/X_Test_Data_Input.csv')
Y_test = pd.read_csv('/content/Y_Test_Data_Target.csv')
import pandas as pd

# Load your feature and target CSV files (update the paths accordingly)
df = pd.read_csv('/content/X_Train_Data_Input.csv')  # This should be your feature dataset
target_df = pd.read_csv('/content/Y_Train_Data_Target.csv')  # This should be your target dataset

# Check the contents of the loaded datasets to ensure they've been loaded correctly
print(df.head())
print(target_df.head())

# Merging features and target based on the 'ID' column
merged_df = pd.merge(df, target_df, on='ID')

# Check the merged result
print(merged_df.head())
```
# output:
![image](https://github.com/user-attachments/assets/4e31740b-5aca-4157-8389-294ae191abf4)
![image](https://github.com/user-attachments/assets/4b9e1ef8-917c-499c-9f9e-7d240b6e7e9f)

```python
# Features (X) and target (y)
X = merged_df.drop(columns=['ID', 'target'])  # Dropping ID and target for the features
y = merged_df['target']  # The target variable

# Preprocessing pipeline (Imputation and Scaling)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median
    ('scaler', StandardScaler())  # Scale features for better performance
])
```
```python
# Apply the preprocessing
X_processed = numeric_transformer.fit_transform(X)
# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
# Model training (Random Forest Classifier as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
# output:
![image](https://github.com/user-attachments/assets/329e1eaa-4885-4e84-acca-d0077491dd56)

```python
# Predictions on the test set
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
# Evaluation of the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance (Optional: to check which features are most important)
feature_importances = model.feature_importances_
for idx, importance in enumerate(feature_importances):
    print(f"Feature {idx}: {importance:.4f}")
```
# Output:
![image](https://github.com/user-attachments/assets/6d1c81b4-b91d-48ff-a87e-43e61ef1ad1b)
