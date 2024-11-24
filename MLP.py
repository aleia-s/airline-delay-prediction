#mlp implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

data = pd.read_csv("flights-wx.csv")

data['FlightDate']=pd.to_datetime(data['FlightDate'])
data['DayOfWeek']= data['FlightDate'].dt.dayofweek+1

#relevant columns going off the knn.py file
data = data[['DepTime', 'DepDelay', 'DayOfWeek', 'Reporting_Airline', 'ArrDelay', 
             'Cancelled', 'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility', 
             'DEST_ceiling', 'DEST_wind', 'DEST_visibility']]

data = data.dropna() #drops rows w missing values

label_encoder = LabelEncoder()

data['Reporting_Airline'] = label_encoder.fit_transform(data['Reporting_Airline'])

#define X are features, Y is target
X = data[['DepTime', 'DepDelay', 'DayOfWeek', 'Reporting_Airline', 'Cancelled', 
          'ORIGIN_ceiling', 'ORIGIN_wind', 'ORIGIN_visibility', 
          'DEST_ceiling', 'DEST_wind', 'DEST_visibility']]
y = (data['ArrDelay'] > 15).astype(int) # True if value in ArrDelay > 15, else False --> done to create binary target variable, y, where delay is Arrival Delay > 15 minutes

#scale numerical features accordingly
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') #sigmoid for binary classification
])

#compile model
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

#train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

#evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

#generate predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#plot training and validation accuracy over all epochs
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#plot training and validation loss over all epochs
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()