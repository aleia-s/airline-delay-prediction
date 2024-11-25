import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional

# Constants
SEQUENCE_LENGTH = 15
BATCH_SIZE = 32
EPOCHS = 50
THRESHOLD = 15  # Minutes to classify as delayed or not delayed

def preprocess_data(file_path, threshold=15):
    """
    Preprocess the data for binary classification.

    Args:
        file_path (str): Path to the CSV file.
        threshold (int): Threshold to classify delays as 0 or 1.

    Returns:
        Tuple: Processed features (X), labels (y).
    """
    # Load data
    data = pd.read_csv(file_path)

    # Parse datetime columns with error handling
    data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
    data['DEP_TS'] = pd.to_datetime(data['DEP_TS'], errors='coerce')
    data['ARR_TS'] = pd.to_datetime(data['ARR_TS'], errors='coerce')

    # Drop rows with invalid datetime entries
    data = data.dropna(subset=['DATE', 'DEP_TS', 'ARR_TS'])
    print("Warning: Some datetime conversions failed. Dropping problematic rows.")

    # Feature engineering
    print("Engineering features...")
    data['DayOfWeek'] = data['DATE'].dt.dayofweek
    data['DepHour'] = data['DEP_TS'].dt.hour
    data['ArrHour'] = data['ARR_TS'].dt.hour

    # Ensure ARR_DELAY is numeric
    data['ARR_DELAY'] = pd.to_numeric(data['ARR_DELAY'], errors='coerce')

    # Drop rows with missing or invalid ARR_DELAY
    data = data.dropna(subset=['ARR_DELAY'])

    # Convert all feature columns to numeric
    features = data.drop(columns=['ARR_DELAY', 'DATE', 'DEP_TS', 'ARR_TS']).apply(pd.to_numeric, errors='coerce')
    features = features.fillna(0)  # Replace any remaining NaN values with 0

    # Convert features and target to numpy arrays
    X = features.to_numpy(dtype=np.float32)
    y = (data['ARR_DELAY'] >= threshold).astype(int).to_numpy()  # Binary labels: 0 or 1

    # Reshape for RNN input
    num_samples = len(X) // SEQUENCE_LENGTH * SEQUENCE_LENGTH
    X = X[:num_samples].reshape(-1, SEQUENCE_LENGTH, X.shape[1])
    y = y[:num_samples].reshape(-1, SEQUENCE_LENGTH)[:, -1]  # Use last time step's label

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    return X, y

def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return alpha * tf.pow(1 - p_t, gamma) * bce


# Step 2: Build Classification Model
def build_classification_model(input_shape, dropout_rate=0.2):
    """
    Build a binary classification model with attention.

    Args:
        input_shape (tuple): Shape of the input data.
        dropout_rate (float): Dropout rate.

    Returns:
        keras.Model: Compiled classification model.
    """
    inputs = layers.Input(shape=input_shape)

    # Convolutional layer
    x = layers.Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # LSTM layer
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # Attention mechanism
    attention = layers.Attention()([x, x])
    x = layers.Add()([x, attention])

    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy', 'Precision', 'Recall'])
    return model

# Step 3: Calculate Statistics
def calculate_statistics(y_true, y_pred):
    """
    Calculate and print classification metrics.

    Args:
        y_true (numpy array): True binary labels.
        y_pred (numpy array): Predicted binary labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Main Function
def main():
    # Load and preprocess data
    file_path = 'flights-wx.csv'
    X, y = preprocess_data(file_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = build_classification_model(X_train.shape[1:], dropout_rate=0.2)
    model.summary()

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob >= 0.3).astype(int).flatten()

    print(f"y_test shape: {y_test.shape}, y_pred_class shape: {y_pred_class.shape}")

    # Calculate statistics
    calculate_statistics(y_test, y_pred_class)

    # Save the model
    model.save('flight_delay_classifier.h5')
    print("Model saved as flight_delay_classifier.h5")

if __name__ == "__main__":
    main()