"""Evaluate the trained waste classifier on the untouched test split."""

from __future__ import annotations

import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from waste_pipeline import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    TEST_FRACTION,
    VALIDATION_FRACTION,
    discover_dataset_splits,
    make_dataset,
    save_evaluation_outputs,
)


MODEL_PATH = os.path.join("saved_model", "best_model.keras")
DATASET_PATH = os.path.join("dataset", "trashnet")
RESULTS_DIRECTORY = "results"
SEED = 42


def main() -> None:
    """Load, validate, and report performance on the fixed held-out test split."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Classifier model does not exist: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
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
    required_layers = {"augment_flip", "logits", "temperature_scale"}
    model_layer_names = {layer.name for layer in model.layers}
    if not required_layers.issubset(model_layer_names):
        raise ValueError(
            "This checkpoint predates the leakage-safe training pipeline and "
            "cannot be evaluated as an untouched-test result. Retrain with the "
            "current train.py before running evaluate.py."
        )

    splits = discover_dataset_splits(
        DATASET_PATH,
        seed=SEED,
        test_fraction=TEST_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
    )
    if splits.class_names != CLASS_NAMES:
        raise ValueError(
            f"Class order differs from the model contract: {splits.class_names}"
        )

    test_dataset = make_dataset(
        splits.test_paths,
        splits.test_labels,
        training=False,
        seed=SEED,
    )
    print(f"TensorFlow version           : {tf.__version__}")
    print(f"Image size / batch size      : {IMAGE_SIZE} / {BATCH_SIZE}")
    print(f"Class order                  : {list(CLASS_NAMES)}")
    print(f"Untouched test samples       : {len(splits.test_paths)}")
    print(
        "Exact duplicate groups/files: "
        f"{splits.duplicate_groups_removed}/{splits.duplicate_files_removed} removed"
    )
    probabilities = model.predict(test_dataset, verbose=1)
    metrics = save_evaluation_outputs(
        splits.test_labels,
        probabilities,
        CLASS_NAMES,
        splits.test_paths,
        RESULTS_DIRECTORY,
    )
    print(f"Evaluation accuracy          : {metrics['accuracy'] * 100:.2f}%")
    print(f"Results directory            : {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
