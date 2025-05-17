# =============================
# train_skin.py
# =============================
import pandas as pd
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Load metadata
meta = pd.read_csv("datasets/skin/HAM10000_metadata.csv")
img_dir = "datasets/skin/images"

label_map = {label: idx for idx, label in enumerate(meta["dx"].unique())}
meta["label"] = meta["dx"].map(label_map)

X, y = [], []
img_size = 128

for i, row in meta.iterrows():
    img_id = row["image_id"]
    label = row["label"]
    path = os.path.join(img_dir, img_id + ".jpg")
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(label)

X = np.array(X) / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(label_map))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Augmentation
augment = ImageDataGenerator(rotation_range=15, zoom_range=0.2, horizontal_flip=True)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(augment.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))
model.save("models/skin_model.h5")
print("Skin cancer model saved.")