# =============================
# train_lung.py
# =============================
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

lung_dir = "C:/Users/ashfa/OneDrive/Desktop/Cancer_Detection_System/datasets/lung/lung_dataset"
classes = ["benign cases", "malignant cases", "normal cases"]

X, y = [], []
img_size = 128

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

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, horizontal_flip=True)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))
model.save("models/lung_model.h5")
print("Lung cancer model saved.")