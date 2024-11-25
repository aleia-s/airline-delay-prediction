import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
try:
    data = pd.read_csv('flights-wx.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Please check the file path.")
    exit()

# Preprocessing
print("Preprocessing the data...")
# Extract day of the week from FlightDate
data['FlightDate'] = pd.to_datetime(data['FlightDate'])
data['DayOfWeek'] = data['FlightDate'].dt.dayofweek + 1  # Monday=1, Sunday=7

# Select relevant columns
data = data[['DepTime', 'DepDelay', 'DayOfWeek', 'Reporting_Airline', 'ArrDelay', 
             'Cancelled', 'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility', 
             'DEST_ceiling', 'DEST_wind', 'DEST_visibility']]
data = data.dropna()  # Drop rows with missing values

# Encode categorical variables (e.g., Reporting_Airline)
label_encoder = LabelEncoder()
data['Reporting_Airline'] = label_encoder.fit_transform(data['Reporting_Airline'])

# Define features (X) and target (y)
X = data[['DepTime', 'DepDelay', 'DayOfWeek', 'Reporting_Airline', 'Cancelled', 
          'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility', 
          'DEST_ceiling', 'DEST_wind', 'DEST_visibility']]
y = (data['ArrDelay'] > 15).astype(int)  # Target: 1 = Delayed, 0 = On Time

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data preprocessing completed.")

# Train Model
print("Training the KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Model training completed.")

# Evaluate Model
print("Evaluating the model...")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

k_values = range(1, 21)  # Testing k from 1 to 20
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='s')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy vs. Number of Neighbors (k)')
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['On Time', 'Delayed'], yticklabels=['On Time', 'Delayed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
