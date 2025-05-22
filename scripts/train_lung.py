# =============================
# train_lung_vgg16.py with test split, class weights, fine-tuning, and evaluation
# =============================
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

# Check class balance
print("Class distribution:", Counter(np.argmax(y, axis=1)))

# Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

# Compute class weights
y_train_integers = np.argmax(y_train, axis=1)
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_integers), y=y_train_integers)
class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Load VGG16 base
base_model = VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Build model
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

# Train initial model with frozen base
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Fine-tune top VGG16 layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Fine-tune training
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/lung_vgg16_finetuned_model.h5")
print("✅ Fine-tuned lung cancer detection model (VGG16) saved.")

# ======================
# Evaluation on Test Set
# ======================
print("\n🧪 Evaluating on test set...")
y_test_pred = model.predict(X_test)
y_test_true_labels = np.argmax(y_test, axis=1)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(
    y_test_true_labels, 
    y_test_pred_labels, 
    target_names=classes,
    digits=4
))

# Confusion Matrix
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test_true_labels, y_test_pred_labels))
