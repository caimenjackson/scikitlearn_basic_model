import pandas as pd
import datetime
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# Connect to your MySQL database
engine = create_engine('mysql+pymysql://app_connect:password@localhost/carapp')

# SQL Query
query = """
SELECT
    v.make,
    v.model,
    YEAR(v.first_use_date) AS year,
    YEAR(m.test_date) AS test_year,
    m.test_mileage,  # Include this line to fetch mileage
    m.test_result
FROM
    vehicles v
JOIN vehicle_mot vm ON v.vehicle_id = vm.vehicle_id
JOIN mot_tests m ON vm.test_id = m.test_id

LIMIT 100000;  # Reduced for quicker processing and simplicity
"""

# Execute the query and load into a pandas DataFrame
data = pd.read_sql_query(query, engine)

# Calculate vehicle age
data['vehicle_age'] = data['test_year'] - data['year']

# Define categorical and numeric features
categorical_features = ['make', 'model']

numeric_features = ['year', 'test_year', 'vehicle_age', 'test_mileage']

numeric_features = ['year', 'test_year', 'vehicle_age']


# Transformers setup
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Encode target variable
encoder = LabelEncoder()
data['test_result_encoded'] = encoder.fit_transform(data['test_result'])

# Data split
X = data.drop(['test_result', 'test_result_encoded'], axis=1)
y = data['test_result_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training
pipeline.fit(X_train, y_train)

# Model evaluation
y_pred = pipeline.predict(X_test)


import os
import joblib

# Define the path where the model should be saved
model_path = 'C:/xampp/htdocs/carapp/machineLearningModule/mysql_queries/'
model_filename = 'model_V2.pkl'
model_filename_encoder = 'encoder_V2.pkl'
full_model_path = os.path.join(model_path, model_filename)
full_model_path_encoder = os.path.join(model_path, model_filename_encoder)

model_filename = 'model_V1.pkl'
full_model_path = os.path.join(model_path, model_filename)


# Ensure the directory exists
os.makedirs(model_path, exist_ok=True)

# Print the path for verification
print(f"Saving model to: {full_model_path}")

# Try to save the model, with exception handling
try:
    joblib.dump(pipeline, full_model_path)
    print("Model saved successfully.")
except Exception as e:
    print(f"Failed to save the model: {e}")
    
    
# Save the encoder for later use
joblib.dump(encoder, full_model_path_encoder)



print(classification_report(y_test, y_pred, target_names=encoder.classes_))