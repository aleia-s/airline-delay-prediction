from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("flights-wx.csv")
data['FlightDate'] = pd.to_datetime(data['FlightDate'])
data['DayOfWeek'] = data['FlightDate'].dt.dayofweek + 1

data = data[['DepTime', 'DepDelay', 'DayOfWeek', 'Reporting_Airline', 'ArrDelay',
             'Cancelled', 'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility',
             'DEST_ceiling', 'DEST_wind', 'DEST_visibility']]
data = data.dropna()

label_encoder = LabelEncoder()
data['Reporting_Airline'] = label_encoder.fit_transform(data['Reporting_Airline'])

X = data[['DepTime', 'DepDelay', 'DayOfWeek', 'Reporting_Airline', 'Cancelled',
          'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility',
          'DEST_ceiling', 'DEST_wind', 'DEST_visibility']]
y = (data['ArrDelay'] > 15).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# SVM
svm = SVC(kernel='linear', random_state=42)
svm_accuracies = cross_val_score(svm, X, y, cv=skf, scoring='accuracy')
results['SVM'] = svm_accuracies

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn_accuracies = cross_val_score(knn, X, y, cv=skf, scoring='accuracy')
results['KNN'] = knn_accuracies

# MLP
def create_mlp_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

mlp_accuracies = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train MLP
    model = create_mlp_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    
    # Evaluate MLP
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    mlp_accuracies.append(accuracy)

results['MLP'] = mlp_accuracies

# Results
plt.figure(figsize=(10, 6))
for model, scores in results.items():
    plt.plot(range(1, len(scores) + 1), scores, marker='o', label=f'{model} (Mean: {np.mean(scores):.2f})')

plt.title('Cross-Validation Accuracy Comparison')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

for model, scores in results.items():
    print(f"{model} Mean Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
