import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

class_names = []
def get_datasets(data_dir, img_size=(224, 224), batch_size=32):
    def map_labels(x, y):
        class_name = tf.gather(class_names, y)

        is_fresh = tf.strings.regex_full_match(class_name, r"^fresh.*")

        label_binary = tf.where(is_fresh, 0, 1)

        return x, tf.cast(label_binary, tf.int32)


    AUTOTUNE = tf.data.AUTOTUNE

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

    # Batch all datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


    return train_ds, val_ds, test_ds


def build_model(img_size=(224, 224, 3), dropout_rate=0.3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze

    inputs = layers.Input(shape=img_size)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    return model


def plot_history(history, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='train acc')
    plt.plot(epochs, val_acc, 'ro-', label='val acc')
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='train loss')
    plt.plot(epochs, val_loss, 'ro-', label='val loss')
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "training_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/dataset_split")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_out", default="models/fresh_stale_model")
    args = parser.parse_args()

    train_ds, val_ds, test_ds = get_datasets(
        args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size
    )

    # Build model
    model = build_model(img_size=(args.img_size, args.img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    plot_history(history, out_dir="logs")

    print("Evaluating on test set...")
    test_metrics = model.evaluate(test_ds)
    print("Test metrics (loss, acc):", test_metrics)

    Path(args.model_out).mkdir(parents=True, exist_ok=True)

    print(f"Exporting final model to {args.model_out}...")
    model.export(args.model_out)
    print(f"Model successfully exported to {args.model_out}")


if __name__ == "__main__":
    main()