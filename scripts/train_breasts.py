import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ======================
# 1. Dataset Preparation
# ======================
def split_dataset(source_dir='datasets/breast/breast_data', target_dir='breast_split_dataset'):
    classes = ['benign', 'malignant', 'normal']
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    for class_name in classes:
        img_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

        for split, split_images in zip(['train', 'val', 'test'], [train, val, test]):
            split_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(img_dir, img)
                dst = os.path.join(split_dir, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

if not os.path.exists('breast_split_dataset/train'):
    print("⏳ Splitting dataset...")
    split_dataset()
    print("✅ Dataset split complete.")

# ======================
# 2. Data Loading & Augmentation
# ======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
train_ds = image_dataset_from_directory(
    'breast_split_dataset/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)
val_ds = image_dataset_from_directory(
    'breast_split_dataset/val',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)
test_ds = image_dataset_from_directory(
    'breast_split_dataset/test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Get class names
class_names = test_ds.class_names

# Class weights for imbalanced data
class_counts = []
for class_name in class_names:
    class_dir = os.path.join('breast_split_dataset/train', class_name)
    class_counts.append(len(os.listdir(class_dir)))

class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(len(class_names)),
    y=np.repeat(np.arange(len(class_names)), class_counts)
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Class weights:", class_weights)

# Enhanced Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.GaussianNoise(0.1),
])

# ======================
# 3. Model Architecture
# ======================
def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    
    # Load VGG16 with pretrained weights
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=x)
    
    # Freeze early layers, train later ones
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True
    
    # Enhanced top layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

model = build_model()

# ======================
# 4. Training Configuration
# ======================
# Custom optimizer
optimizer = optimizers.Adam(
    learning_rate=1e-5,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/best_breast_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ======================
# 5. Model Training
# ======================
print("\n🚀 Training model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    class_weight=class_weights
)

# ======================
# 6. Evaluation & Testing
# ======================
# Load best model
model = tf.keras.models.load_model('models/best_breast_model.keras')

# Test Time Augmentation (TTA)
def predict_with_tta(model, dataset, n_aug=5):
    all_preds = []
    for images, _ in dataset:
        batch_preds = []
        for _ in range(n_aug):
            augmented_images = data_augmentation(images)
            batch_preds.append(model.predict(augmented_images))
        avg_preds = np.mean(batch_preds, axis=0)
        all_preds.append(avg_preds)
    return np.concatenate(all_preds, axis=0)

print("\n🧪 Evaluating with TTA...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = predict_with_tta(model, test_ds, n_aug=5)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

# Classification Report
print("\n📊 Final Classification Report:")
print(classification_report(
    y_true_labels, 
    y_pred_labels, 
    target_names=class_names,
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
print("\n🔍 Confusion Matrix:")
print(cm)

# ======================
# 7. Save Final Model
# ======================
os.makedirs("models", exist_ok=True)
model.save("models/final_breast_vgg16_model.keras")
print("\n✅ Final model saved to models/final_breast_vgg16_model.keras")

# ======================
# 8. Plot Training History
# ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('training_history.png')
plt.show()