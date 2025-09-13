import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === Directories ===
MODEL_PATH = "models/dcnn_fire_detection.h5"
TEST_DIR = "dataset/test/"
SAVE_DIR = "models"

# === Load Trained Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Test Data Generator ===
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128, 128),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# === Predict ===
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=class_labels)
print("\nClassification Report:")
print(report)

# Save classification report
report_path = os.path.join(SAVE_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
conf_matrix_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (AUC = {:.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
roc_curve_path = os.path.join(SAVE_DIR, "roc_curve.png")
plt.savefig(roc_curve_path)
plt.close()
