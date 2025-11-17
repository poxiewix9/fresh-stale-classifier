import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path


def load_model(model_dir):
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_1")

    sm_layer = tf.keras.layers.TFSMLayer(
        model_dir,
        call_endpoint='serving_default'
    )

    outputs = sm_layer(inputs)

    inference_model = tf.keras.models.Model(inputs, outputs)
    return inference_model


def get_labels_and_preds(model, dataset, class_names):
    y_true = []
    y_pred = []

    def map_labels_to_binary(labels_int):
        binary_labels = []
        for label_int in labels_int.numpy():
            class_name = class_names[label_int]
            binary_labels.append(0 if class_name.startswith("fresh") else 1)
        return np.array(binary_labels, dtype=int)

    for batch_images, batch_labels_int in dataset.unbatch().batch(32):
        preds = model.predict(batch_images)

        if isinstance(preds, dict):
            preds = list(preds.values())[0]

        preds_binary = (preds > 0.5).astype(int).reshape(-1)

        y_true_binary = map_labels_to_binary(batch_labels_int)

        y_pred.extend(preds_binary.tolist())
        y_true.extend(y_true_binary.tolist())

    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/fresh_stale_model")
    parser.add_argument("--data_dir", default="data/raw/dataset_split")
    args = parser.parse_args()

    test_ds = image_dataset_from_directory(
        os.path.join(args.data_dir, "test"),
        label_mode="int",  # <-- CRITICAL
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )

    class_names = test_ds.class_names
    print(f"Found {len(class_names)} classes, mapping to binary...")

    test_ds_prefetched = test_ds.prefetch(tf.data.AUTOTUNE)

    print("Loading model...")
    model = load_model(args.model_dir)

    model.summary()
    print("Model loaded. Starting evaluation...")

    y_true, y_pred = get_labels_and_preds(model, test_ds_prefetched, class_names)

    print("\n--- Evaluation Report ---")
    print("Classification report:")

    print(classification_report(y_true, y_pred, target_names=['fresh', 'stale']))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("       [fresh] [stale]")  # Added labels for clarity
    print(cm)


if __name__ == "__main__":
    main()