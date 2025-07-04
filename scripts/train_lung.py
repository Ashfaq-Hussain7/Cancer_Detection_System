# =============================
# train_lung.py (Updated Version)
# =============================

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter

# Dataset setup
lung_dir = "C:/Users/ashfa/OneDrive/Desktop/Cancer_Detection_System/datasets/lung/lung_dataset"
classes = ["benign cases", "malignant cases", "normal cases"]
img_size = 224

X, y = [], []

# Load and label data
for idx, cls in enumerate(classes):
    cls_dir = os.path.join(lung_dir, cls)
    for img_file in os.listdir(cls_dir):
        path = os.path.join(cls_dir, img_file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)

X = np.array(X) / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Class distribution
print("Class distribution:", Counter(np.argmax(y, axis=1)))

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Class weights
y_train_integers = np.argmax(y_train, axis=1)
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_integers), y=y_train_integers)
class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Load VGG16 base model
base_model = VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Build custom model
inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# Train with frozen base
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Fine-tune top layers of VGG16
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/lung_vgg16_finetuned_model.h5")
print("✅ Model saved to models/lung_vgg16_finetuned_model.h5")

# ======================
# Evaluation on Validation Set
# ======================
print("\n🧪 Evaluating on validation set...")

y_val_pred = model.predict(X_val)
y_val_true = np.argmax(y_val, axis=1)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)

print("\n📊 Classification Report on Validation Set:")
print(classification_report(y_val_true, y_val_pred_labels, target_names=classes, digits=4))

print("\n🔍 Confusion Matrix on Validation Set:")
print(confusion_matrix(y_val_true, y_val_pred_labels))

val_accuracy = accuracy_score(y_val_true, y_val_pred_labels)
print(f"\n🎯 Validation Accuracy: {val_accuracy:.4f}")
