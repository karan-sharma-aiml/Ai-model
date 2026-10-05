"""Predict a TrashNet class from one image using the trained Keras model."""

from __future__ import annotations

import os
from os import PathLike

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from waste_pipeline import CLASS_NAMES, IMAGE_SIZE, decode_and_preprocess


MODEL_PATH = os.path.join("saved_model", "best_model.keras")


def load_classifier(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Load the model and validate its output dimensions against class labels."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Classifier model does not exist: {model_path}")
    model = load_model(model_path)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "Loaded model output count does not match the class mapping: "
            f"{model.output_shape[-1]} outputs for {len(CLASS_NAMES)} classes."
        )
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
        raise ValueError(
            f"Loaded model input shape {model.input_shape} is incompatible with "
            f"the expected image shape {(*IMAGE_SIZE, 3)}."
        )
    return model


model = load_classifier()


def predict_image(image_path: str | PathLike[str]) -> tuple[str, float]:
    """Return the predicted class name and calibrated top-class confidence."""
    image_path = os.fspath(image_path)
    if not tf.io.gfile.exists(image_path):
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image_bytes = tf.io.read_file(image_path)
    image = decode_and_preprocess(image_bytes)
    batch = tf.expand_dims(image, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output probabilities, "
            f"received shape {probabilities.shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite class probabilities.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError("Model output is not a normalized probability distribution.")

    predicted_index = int(np.argmax(probabilities))
    return CLASS_NAMES[predicted_index], float(probabilities[predicted_index] * 100.0)


if __name__ == "__main__":
    image_path = input("Enter Image Path: ").strip()
    try:
        predicted_class, confidence = predict_image(image_path)
    except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
        print(f"\nPrediction failed: {error}")
    else:
        print("=" * 60)
        print("Prediction Result")
        print("=" * 60)
        print(f"Waste Category : {predicted_class.upper()}")
        print(f"Confidence     : {confidence:.2f}%")
        print("=" * 60)
