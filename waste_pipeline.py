"""Shared dataset, preprocessing, and evaluation utilities for TrashNet."""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


CLASS_NAMES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
CALIBRATION_BINS = 15


@dataclass(frozen=True)
class DatasetSplits:
    """Stable, stratified paths and integer labels for all three data splits."""

    class_names: tuple[str, ...]
    train_paths: tuple[str, ...]
    train_labels: np.ndarray
    validation_paths: tuple[str, ...]
    validation_labels: np.ndarray
    test_paths: tuple[str, ...]
    test_labels: np.ndarray
    duplicate_groups_removed: int
    duplicate_files_removed: int


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_image(path: str) -> bool:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Unreadable or corrupted image: {path}: {error}") from error
    return True


def discover_dataset_splits(
    dataset_path: str,
    seed: int,
    test_fraction: float = TEST_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> DatasetSplits:
    """Validate classes/images, remove exact duplicates, and make fixed splits.

    Exact duplicates assigned different labels are excluded as ambiguous
    annotations. Same-label exact duplicates are represented once, preventing
    copies of an image from crossing the train/validation/test boundaries.
    """
    from sklearn.model_selection import train_test_split

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    class_names = tuple(
        sorted(
            entry.name
            for entry in os.scandir(dataset_path)
            if entry.is_dir()
        )
    )
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Expected class folders {list(CLASS_NAMES)}, found {list(class_names)}"
        )

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0 - test_fraction:
        raise ValueError(
            "validation_fraction must be positive and smaller than "
            "1 - test_fraction."
        )

    files_by_hash: dict[str, list[tuple[str, int]]] = {}
    for class_index, class_name in enumerate(class_names):
        class_directory = os.path.join(dataset_path, class_name)
        class_paths = sorted(
            os.path.join(class_directory, filename)
            for filename in os.listdir(class_directory)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(f"No supported images found in {class_directory}")

        for path in class_paths:
            _valid_image(path)
            files_by_hash.setdefault(_sha256(path), []).append((path, class_index))

    retained_paths: list[str] = []
    retained_labels: list[int] = []
    duplicate_groups_removed = 0
    duplicate_files_removed = 0
    for duplicate_group in files_by_hash.values():
        if len(duplicate_group) > 1:
            labels_in_group = {label for _, label in duplicate_group}
            if len(labels_in_group) > 1:
                duplicate_files_removed += len(duplicate_group)
                duplicate_groups_removed += 1
                duplicate_names = [
                    f"{class_names[label]}/{os.path.basename(path)}"
                    for path, label in duplicate_group
                ]
                warnings.warn(
                    "Excluding exact duplicate images with conflicting labels: "
                    + ", ".join(duplicate_names),
                    stacklevel=2,
                )
                continue

            duplicate_groups_removed += 1
            duplicate_files_removed += len(duplicate_group) - 1
            warnings.warn(
                "Keeping one copy of duplicate images: "
                + ", ".join(os.path.basename(path) for path, _ in duplicate_group),
                stacklevel=2,
            )

        path, label = duplicate_group[0]
        retained_paths.append(path)
        retained_labels.append(label)

    labels = np.asarray(retained_labels, dtype=np.int64)
    if any(np.count_nonzero(labels == index) < 3 for index in range(len(class_names))):
        raise ValueError(
            "Each class needs at least three unique, readable images for "
            "stratified train/validation/test splits."
        )

    development_paths, test_paths, development_labels, test_labels = train_test_split(
        retained_paths,
        labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    validation_relative_fraction = validation_fraction / (1.0 - test_fraction)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        development_paths,
        development_labels,
        test_size=validation_relative_fraction,
        random_state=seed + 1,
        stratify=development_labels,
    )

    return DatasetSplits(
        class_names=class_names,
        train_paths=tuple(train_paths),
        train_labels=np.asarray(train_labels, dtype=np.int64),
        validation_paths=tuple(validation_paths),
        validation_labels=np.asarray(validation_labels, dtype=np.int64),
        test_paths=tuple(test_paths),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        duplicate_groups_removed=duplicate_groups_removed,
        duplicate_files_removed=duplicate_files_removed,
    )


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """Resize RGB pixels consistently and normalize for MobileNetV2."""
    image = tf.image.resize(image, IMAGE_SIZE, method="nearest")
    image = tf.cast(image, tf.float32)
    return preprocess_input(image)


def decode_and_preprocess(
    encoded_image: tf.Tensor,
) -> tf.Tensor:
    """Decode image bytes and apply the shared prediction/training transform."""
    image = tf.io.decode_image(
        encoded_image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape((None, None, 3))
    return preprocess_image(image)


def _load_path_and_label(
    path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = decode_and_preprocess(tf.io.read_file(path))
    return image, tf.one_hot(label, len(CLASS_NAMES))


def make_dataset(
    paths: Sequence[str],
    labels: np.ndarray,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build a lazy, parallel data pipeline without caching all images in RAM."""
    if len(paths) != len(labels) or not paths:
        raise ValueError("Image paths and labels must be non-empty and equal in size.")

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return (
        dataset.map(
            _load_path_and_label,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one scalar temperature using the validation split only."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) == 0:
        raise ValueError("Calibration requires aligned, non-empty logits and labels.")

    row_indices = np.arange(len(labels))

    def negative_log_likelihood(temperature: float) -> float:
        scaled_logits = logits / temperature
        return float(
            np.mean(
                logsumexp(scaled_logits, axis=1)
                - scaled_logits[row_indices, labels]
            )
        )

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(0.05, 10.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"Temperature calibration failed: {result.message}")
    return float(result.x)


def expected_calibration_error(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-label ECE and equal-width reliability-bin data."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(true_labels) != len(probabilities):
        raise ValueError("ECE requires aligned class labels and probability rows.")
    if bins < 1:
        raise ValueError("bins must be a positive integer.")

    confidence = probabilities.max(axis=1)
    correctness = probabilities.argmax(axis=1) == true_labels
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_accuracy = np.full(bins, np.nan, dtype=np.float64)
    bin_confidence = np.full(bins, np.nan, dtype=np.float64)
    bin_counts = np.zeros(bins, dtype=np.int64)
    ece = 0.0

    for bin_index in range(bins):
        lower, upper = bin_edges[bin_index : bin_index + 2]
        if bin_index == bins - 1:
            members = (confidence >= lower) & (confidence <= upper)
        else:
            members = (confidence >= lower) & (confidence < upper)
        count = int(np.count_nonzero(members))
        bin_counts[bin_index] = count
        if count:
            bin_accuracy[bin_index] = float(np.mean(correctness[members]))
            bin_confidence[bin_index] = float(np.mean(confidence[members]))
            ece += count / len(true_labels) * abs(
                bin_accuracy[bin_index] - bin_confidence[bin_index]
            )

    return ece, bin_accuracy, bin_confidence, bin_counts


def evaluate_predictions(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Validate probabilities and calculate aggregate plus top-k metrics."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    true_labels = np.asarray(true_labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_count = len(class_names)
    if len(true_labels) == 0 or probabilities.shape != (len(true_labels), class_count):
        raise ValueError(
            "Model output shape does not match the sample count and class list: "
            f"{probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite prediction probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Model outputs are not normalized class probabilities.")
    if np.any(true_labels < 0) or np.any(true_labels >= class_count):
        raise ValueError("Ground-truth class index is outside the class list.")

    predicted_labels = probabilities.argmax(axis=1)
    top_indices = np.argsort(probabilities, axis=1)
    ece, _, _, _ = expected_calibration_error(true_labels, probabilities)
    top_two = np.any(top_indices[:, -2:] == true_labels[:, None], axis=1)
    top_three = np.any(top_indices[:, -3:] == true_labels[:, None], axis=1)
    return {
        "accuracy": float(np.mean(predicted_labels == true_labels)),
        "precision_macro": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "top_2_accuracy": float(np.mean(top_two)),
        "top_3_accuracy": float(np.mean(top_three)),
        "ece": ece,
    }


def save_evaluation_outputs(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    file_paths: Sequence[str],
    results_directory: str,
) -> dict[str, float]:
    """Print and persist held-out metrics, reports, and diagnostic plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.integrate import trapezoid
    from sklearn.metrics import (
        auc,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    os.makedirs(results_directory, exist_ok=True)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted_labels = probabilities.argmax(axis=1)
    metrics = evaluate_predictions(true_labels, probabilities, class_names)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    print("=" * 72)
    print("HELD-OUT TEST PERFORMANCE")
    print("=" * 72)
    for metric_name, metric_value in metrics.items():
        display_name = metric_name.replace("_", " ").title()
        print(f"{display_name:<24}: {metric_value * 100:6.2f}%")
    print("\nClassification report")
    print(report)
    with open(
        os.path.join(results_directory, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as report_file:
        report_file.write(report)

    confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
    )
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Held-Out Test Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, "confusion_matrix.png"),
        dpi=300,
    )
    plt.close()

    one_hot_labels = np.eye(len(class_names))[true_labels]
    plt.figure(figsize=(10, 8))
    for class_index, class_name in enumerate(class_names):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            one_hot_labels[:, class_index],
            probabilities[:, class_index],
        )
        class_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{class_name} (AUC = {class_auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.title("One-vs-Rest ROC Curves (Held-Out Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "roc_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    for class_index, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            one_hot_labels[:, class_index],
            probabilities[:, class_index],
        )
        class_auc = trapezoid(precision[::-1], recall[::-1])
        plt.plot(
            recall,
            precision,
            label=f"{class_name} (AUC = {class_auc:.3f})",
        )
    plt.title("One-vs-Rest Precision-Recall Curves (Held-Out Test)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, "precision_recall_curves.png"),
        dpi=300,
    )
    plt.close()

    ece, bin_accuracy, bin_confidence, bin_counts = expected_calibration_error(
        true_labels,
        probabilities,
    )
    non_empty = bin_counts > 0
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(
        bin_confidence[non_empty],
        bin_accuracy[non_empty],
        marker="o",
        label="Model",
    )
    plt.title(f"Confidence Calibration (ECE = {ece:.4f})")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, "calibration_curve.png"),
        dpi=300,
    )
    plt.close()

    wrong_indices = np.flatnonzero(predicted_labels != true_labels)
    with open(
        os.path.join(results_directory, "wrong_predictions.txt"),
        "w",
        encoding="utf-8",
    ) as wrong_file:
        wrong_file.write("Held-out test misclassifications\n")
        wrong_file.write("=" * 60 + "\n")
        for index in wrong_indices:
            line = (
                f"{os.path.relpath(file_paths[index])} | "
                f"Actual: {class_names[true_labels[index]]} | "
                f"Predicted: {class_names[predicted_labels[index]]} | "
                f"Confidence: {probabilities[index].max() * 100:.2f}%"
            )
            wrong_file.write(line + "\n")

    return metrics


def save_training_curves(
    history: dict[str, list[float]],
    output_directory: str,
) -> None:
    """Save accuracy and loss curves for both training phases."""
    import matplotlib.pyplot as plt

    os.makedirs(output_directory, exist_ok=True)
    for metric, title, filename in (
        ("accuracy", "Training and Validation Accuracy", "accuracy.png"),
        ("loss", "Training and Validation Loss", "loss.png"),
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(history.get(metric, []), label=f"Training {metric}")
        plt.plot(history.get(f"val_{metric}", []), label=f"Validation {metric}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, filename), dpi=300)
        plt.close()
