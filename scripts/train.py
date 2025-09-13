import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    GlobalAveragePooling2D, Dense, Input, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMAGE_SIZE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# Data generators with augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

# Training Data Generator
train_generator = datagen.flow_from_directory(
    "dataset/",
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

# Validation Data Generator
val_generator = datagen.flow_from_directory(
    "dataset/",
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,  # Important for evaluation
)

# Deep Dual-Channel Neural Network (DCNN) Architecture
def build_dcnn(input_shape):
    input_layer = Input(shape=input_shape)

    # First Channel (Texture Extraction)
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = Conv2D(128, (3, 3), activation="relu", padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = Conv2D(256, (3, 3), activation="relu", padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = GlobalAveragePooling2D()(x1)

    # Second Channel (Contour Extraction)
    x2 = Conv2D(64, (5, 5), activation="relu", padding="same")(input_layer)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    x2 = Conv2D(128, (5, 5), activation="relu", padding="same")(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    x2 = Conv2D(256, (5, 5), activation="relu", padding="same")(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    x2 = GlobalAveragePooling2D()(x2)

    # Feature Fusion
    merged = concatenate([x1, x2])

    # Fully Connected Layers
    dense_layer = Dense(256, activation="relu")(merged)
    dense_layer = Dropout(DROPOUT_RATE)(dense_layer)
    output_layer = Dense(1, activation="sigmoid")(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Compile Model
model = build_dcnn(IMAGE_SIZE)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
)

# Save the Model
model.save("models/dcnn_fire_detection.h5")

# Plot Training History
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate Model and Show Classification Report
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
y_pred = np.round(predictions).astype(int).flatten()
y_true = val_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
