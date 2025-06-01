import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# Paths and config
data_path = os.path.join("data", "iris.csv")
weights_path = "model.weights.h5"
log_path = "metrics.csv"
epochs = 30
batch_size = 8

# Load CSV data
df = pd.read_csv(data_path)

# Features and labels
X = df.iloc[:, :-1].values  # First 4 columns
y = df.iloc[:, -1].values   # Last column (species)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_val, y_val),
          verbose=0,
          callbacks=[TqdmCallback(), CSVLogger(log_path)])

# Save weights
model.save_weights(weights_path)

