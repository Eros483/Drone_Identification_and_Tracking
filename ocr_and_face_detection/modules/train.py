import tensorflow as tf
import os
import sys
import json

# Config
IMG_SIZE=224
BATCH_SIZE=16
EPOCHS=1
SPLIT_RATIO=0.8

# Accept single input folder path
DATA_DIR =sys.argv[1] if len(sys.argv) > 1 else "data"

# Verify data directory exists
if not os.path.exists(DATA_DIR):
    print(f"[ERROR] Data directory '{DATA_DIR}' not found!")
    sys.exit(1)

print(f"[INFO] Using data directory: {DATA_DIR}")

try: 
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    
    class_names = full_dataset.class_names
    total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
    total_samples = total_batches * BATCH_SIZE
    
    print(f"[INFO] Found {total_samples} total samples")
    print(f"[INFO] Classes: {class_names}")
    
    # Split into train and validation
    train_size = int(total_batches * SPLIT_RATIO)
    val_size = total_batches - train_size
    
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    print(f"[INFO] Training batches: {train_size}")
    print(f"[INFO] Validation batches: {val_size}")
    
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    print("Make sure your data directory structure is:")
    print("data/")
    print("  ├── class1/")
    print("  │   ├── image1.jpg")
    print("  │   └── image2.jpg")
    print("  └── class2/")
    print("      ├── image3.jpg")
    print("      └── image4.jpg")
    sys.exit(1)

# Data preprocessing and augmentation
def preprocess_data(image, label):
    # Convert to float32 and apply MobileNetV2 preprocessing
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def augment_data(image, label):
    # Apply data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    # Random rotation (requires TensorFlow Addons or manual implementation)
    return image, label

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_data)
val_dataset = val_dataset.map(preprocess_data)

# Apply augmentation only to training data
train_dataset = train_dataset.map(augment_data)

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
print("[INFO] Creating model...")

# MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Use sparse since we have integer labels
    metrics=['accuracy']
)

print("[INFO] Model summary:")
model.summary()

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]

print("[INFO] Starting training...")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Create models directory
os.makedirs("models", exist_ok=True)

# Save the model
print("[INFO] Saving model...")
model.save("models/classifier_model.h5")

# Save class names
with open("models/labels.txt", "w") as f:
    for label in class_names:
        f.write(f"{label}\n")

# Save class mapping as JSON
class_mapping = {i: name for i, name in enumerate(class_names)}
with open("models/class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)

# Convert to TensorFlow Lite
print("[INFO] Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Add optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save TFLite model
with open("models/classifier_model.tflite", "wb") as f:
    f.write(tflite_model)

print("[INFO] Training and TFLite export complete!")
print(f"[INFO] Model saved to: models/classifier_model.h5")
print(f"[INFO] TFLite model saved to: models/classifier_model.tflite")
print(f"[INFO] Labels saved to: models/labels.txt")
print(f"[INFO] Class mapping saved to: models/class_mapping.json")

# Print final training results
if history.history:
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"[INFO] Final training accuracy: {final_acc:.4f}")
    print(f"[INFO] Final validation accuracy: {final_val_acc:.4f}")