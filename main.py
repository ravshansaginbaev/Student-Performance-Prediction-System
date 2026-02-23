import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    RocCurveDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("student-mat.csv", sep=';')

# Create target column
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Select features
X = df[['absences', 'G1', 'G2']]
y = df['pass']

# Split (stratify keeps pass/fail ratio similar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# Build DNN
# ----------------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Train (capture history)
# ----------------------------
EPOCHS = 50
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=0
)

# ----------------------------
# Plots: Train vs Val Accuracy, Train vs Val Loss
# ----------------------------
epochs_range = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Evaluate
# ----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")

# ----------------------------
# Predictions on test set
# ----------------------------
# Probabilities (for ROC)
y_prob = model.predict(X_test, verbose=0).ravel()

# Labels (for confusion matrix / precision / recall)
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAIL", "PASS"])
disp.plot(values_format='d')
plt.title("Confusion Matrix (threshold=0.5)")
plt.grid(False)
plt.show()

# ----------------------------
# Precision, Recall, F1 + full report
# ----------------------------
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["FAIL", "PASS"], zero_division=0))

# ----------------------------
# ROC Curve + AUC
# ----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', label="Random classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# NEW: Sample input prediction + probability bar
# ----------------------------
new_student = np.array([[2, 12, 11]], dtype=float)  # [absences, G1, G2]
new_student_scaled = scaler.transform(new_student)

prob = float(model.predict(new_student_scaled, verbose=0)[0][0])
label = "PASS" if prob >= 0.5 else "FAIL"

print(f"New student predicted probability (PASS): {prob:.2f}")
print(f"New student prediction: {label}")

plt.figure()
plt.bar(['FAIL', 'PASS'], [1.0 - prob, prob])
plt.title('New Student Predicted Probability')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
