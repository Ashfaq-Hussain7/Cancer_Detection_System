import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Paths
IMG_DIR = "C:/Users/ashfa/OneDrive/Desktop/Cancer_Detection_System/datasets/skin/images"
META_CSV = "C:/Users/ashfa/OneDrive/Desktop/Cancer_Detection_System/datasets/skin/HAM10000_metadata.csv"
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Load metadata and clean labels
df = pd.read_csv(META_CSV)
df['filename'] = df['image_id'] + ".jpg"
df['label'] = df['dx'].astype(str).str.strip()

# Reconstruct label list
all_classes = sorted(df['label'].unique())
print("✅ Expected class labels:", all_classes)

# Map to class indices
label_map = {label: idx for idx, label in enumerate(all_classes)}
reverse_label_map = {v: k for k, v in label_map.items()}
df['class'] = df['label'].map(label_map)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=SEED)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['class'], random_state=SEED)

# Oversample train_df
df_list = []
max_samples = train_df['label'].value_counts().max()
for label in all_classes:
    class_df = train_df[train_df['label'] == label]
    class_upsampled = resample(class_df, replace=True, n_samples=max_samples, random_state=SEED)
    df_list.append(class_upsampled)

train_df_balanced = pd.concat(df_list).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\n✅ Class distribution in oversampled train_df:")
print(train_df_balanced['label'].value_counts())

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(
    train_df_balanced,
    directory=IMG_DIR,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    classes=all_classes,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_data = val_test_gen.flow_from_dataframe(
    val_df,
    directory=IMG_DIR,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    classes=all_classes,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = val_test_gen.flow_from_dataframe(
    test_df,
    directory=IMG_DIR,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    classes=all_classes,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ✅ Debug: Verify class mappings
print("\n💡 train_data.class_indices:", train_data.class_indices)
print("💡 val_data.class_indices:  ", val_data.class_indices)
print("💡 test_data.class_indices: ", test_data.class_indices)

# Check model output size
num_classes = len(all_classes)
print(f"\n✅ Model will output {num_classes} classes")

# Build model
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train (initial freeze)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)

# Fine-tune last 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=['accuracy'])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/skin_model_debug_safe.h5")
print("✅ Model saved: skin_model_debug_safe.h5")

# Evaluation
print("\n🧪 Evaluating on test set...")

y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=all_classes,
    digits=4
))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=all_classes, yticklabels=all_classes)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Save predictions
pred_df = pd.DataFrame({
    'filename': test_data.filenames,
    'true_label': [all_classes[i] for i in y_true],
    'pred_label': [all_classes[i] for i in y_pred]
})
pred_df.to_csv("skin_test_predictions.csv", index=False)
print("📄 Predictions saved to skin_test_predictions.csv")
