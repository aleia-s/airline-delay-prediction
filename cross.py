import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data
file_path = 'flights-wx.csv'
flights_data = pd.read_csv(file_path)

# Remove canceled flights
df_cleaned = flights_data[flights_data['Cancelled'] == 0].copy()

# Fill missing values in relevant columns with median (safe for numeric data)
df_cleaned['DepTime'].fillna(df_cleaned['DepTime'].median(), inplace=True)
df_cleaned['DepDelay'].fillna(df_cleaned['DepDelay'].median(), inplace=True)
df_cleaned['ArrDelay'].fillna(df_cleaned['ArrDelay'].median(), inplace=True)
df_cleaned['ORIGIN_ceiling'].fillna(df_cleaned['ORIGIN_ceiling'].median(), inplace=True)
df_cleaned['ORIGIN_wind'].fillna(df_cleaned['ORIGIN_wind'].median(), inplace=True)
df_cleaned['ORIGIN_visibility'].fillna(df_cleaned['ORIGIN_visibility'].median(), inplace=True)
df_cleaned['DEST_ceiling'].fillna(df_cleaned['DEST_ceiling'].median(), inplace=True)
df_cleaned['DEST_wind'].fillna(df_cleaned['DEST_wind'].median(), inplace=True)
df_cleaned['DEST_visibility'].fillna(df_cleaned['DEST_visibility'].median(), inplace=True)

# Create target column, 1 is delayed, 0 is not
df_cleaned['IsDelayed'] = (df_cleaned['ArrDelay'] > 15).astype(int)

features = ['DepTime', 'DepDelay', 'Reporting_Airline', 'Origin', 'Dest', 
            'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility', 
            'DEST_ceiling', 'DEST_wind', 'DEST_visibility']
target = 'IsDelayed'

categorical_columns = ['Reporting_Airline', 'Origin', 'Dest']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le 

# Extract X and Y
X = df_cleaned[features]
y = df_cleaned[target]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold Cross-Validation with KNN
knn = KNeighborsClassifier(n_neighbors=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = []
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train and evaluate
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cv_scores.append(accuracy_score(y_test, y_pred))

# Results
print(f"Cross-Validation Accuracies: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
