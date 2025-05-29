# =============================
# predict_lung_test_cases.py
# =============================

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Paths
test_dir = "C:/Users/ashfa/OneDrive/Desktop/Cancer_Detection_System/datasets/lung/test_cases"
model_path = "models/lung_vgg16_finetuned_model.h5"
img_size = 224
classes = ["benign cases", "malignant cases", "normal cases"]

# Load model
model = load_model(model_path)
print("✅ Model loaded.")

# Predict
filenames = []
predictions = []

for img_file in os.listdir(test_dir):
    path = os.path.join(test_dir, img_file)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (img_size, img_size))
        img = np.expand_dims(img / 255.0, axis=0)
        pred = model.predict(img)
        label = classes[np.argmax(pred)]
        filenames.append(img_file)
        predictions.append(label)

# Save to CSV
df = pd.DataFrame({'filename': filenames, 'predicted_label': predictions})
df.to_csv("lung_test_predictions.csv", index=False)
print("📁 Predictions saved to lung_test_predictions.csv")
