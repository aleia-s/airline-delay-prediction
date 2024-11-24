import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Dataset
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

'''
# Visualize the Data
print("Visualizing the data...")
# Distribution of delays
sns.countplot(x=y, palette='coolwarm')
plt.title('Distribution of Flight Delays')
plt.xlabel('Delayed (0 = No Delay, 1 = Delayed)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
'''