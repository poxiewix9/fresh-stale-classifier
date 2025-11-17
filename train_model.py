"""
Train a fresh model without batch_shape issues
Run this locally or on a machine with GPU access
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from pathlib import Path

# Set to avoid batch_shape in saved models
tf.keras.backend.set_image_data_format('channels_last')
# Use input_shape instead of batch_shape for compatibility
os.environ['TF_KERAS_SAVE_FORMAT'] = 'h5'

def get_datasets(data_dir, img_size=(224, 224), batch_size=32):
    """Load and prepare datasets"""
    AUTOTUNE = tf.data.AUTOTUNE
    
    def map_labels(x, y):
        class_name = tf.gather(class_names, y)
        is_fresh = tf.strings.regex_full_match(class_name, r"^fresh.*")
        label_binary = tf.where(is_fresh, 0, 1)
        return x, tf.cast(label_binary, tf.int32)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        label_mode="int",
        image_size=img_size,
        batch_size=None,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        label_mode="int",
        image_size=img_size,
        batch_size=None,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        label_mode="int",
        image_size=img_size,
        batch_size=None,
        shuffle=False,
    )

    global class_names
    class_names = val_ds.class_names
    print(f"Found {len(class_names)} classes. Mapping to binary (0=fresh, 1=stale)...")

    train_ds = train_ds.map(map_labels, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(map_labels, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(map_labels, num_parallel_calls=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()

    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(img_size=(224, 224, 3), dropout_rate=0.3):
    """Build MobileNetV2 model"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=img_size)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/dataset_split", help="Path to dataset split directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--output", default="best_model.h5", help="Output model filename")
    args = parser.parse_args()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Training with data from: {args.data_dir}")

    # Load datasets
    train_ds, val_ds, test_ds = get_datasets(
        args.data_dir, 
        img_size=(args.img_size, args.img_size), 
        batch_size=args.batch_size
    )

    # Build model
    model = build_model(img_size=(args.img_size, args.img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=3, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output, 
            save_best_only=True, 
            monitor="val_loss"
        ),
    ]

    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Evaluate
    print("Evaluating on test set...")
    test_metrics = model.evaluate(test_ds)
    print(f"Test Loss: {test_metrics[0]:.4f}, Test Accuracy: {test_metrics[1]:.4f}")

    print(f"\n✅ Model saved to {args.output}")
    print("You can now use this model in your API!")


if __name__ == "__main__":
    class_names = []
    main()

