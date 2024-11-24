from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

data = pd.read_csv("C:/Users/river/Desktop/School/COSC4368/Group Project/flights-wx.csv")

data['IsDelayed'] = (data['DepDelay'] > 0).astype(int)
y = data['IsDelayed']
X =data.drop(columns=['Origin', 'Flight_Number_Reporting_Airline', 'CancellationCode', 'DepDelay', 'IsDelayed'])

print(data.head())
print(X.columns)

numeric_columns = X.select_dtypes(include=["float64", "int64"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

numeric_imputer = SimpleImputer(strategy="mean")
X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
X[categorical_columns] = X[categorical_columns].fillna("Unknown")

label_encoder = LabelEncoder()
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

