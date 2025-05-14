import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 1. Split Dataset Function
def split_dataset(source_dir='C:/Users/ashfa/OneDrive/Desktop/Cancer_Detection_System/datasets/breast/breast_data',
                  target_dir='breast_split_dataset'):
    classes = ['benign', 'malignant', 'normal']
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    for class_name in classes:
        img_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        for split, split_images in zip(['train', 'val', 'test'], [train, val, test]):
            split_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(img_dir, img)
                dst = os.path.join(split_dir, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

# 2. Split Dataset if Not Already Done
if not os.path.exists('breast_split_dataset/train'):
    print("⏳ Splitting dataset...")
    split_dataset()
    print("✅ Dataset split complete.")

# 3. Load Dataset
img_size = (224, 224)
batch_size = 32

train_ds = image_dataset_from_directory(
    'breast_split_dataset/train',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)
val_ds = image_dataset_from_directory(
    'breast_split_dataset/val',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)
test_ds = image_dataset_from_directory(
    'breast_split_dataset/test',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False
)

# ✅ Get class names BEFORE prefetching
class_names = test_ds.class_names

# 4. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# 5. Load and Fine-Tune VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

# Freeze all layers except last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

# 6. Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# 7. Compile Model (lower LR for fine-tuning)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 8. Learning Rate Scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 9. Train Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[lr_scheduler]
)

# 10. Save Model
os.makedirs("models", exist_ok=True)
model.save("models/breast_vgg16_finetuned_model.keras")
print("✅ Model saved to models/breast_vgg16_finetuned_model.keras")

# 11. Evaluate on Test Set
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

# 12. Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))