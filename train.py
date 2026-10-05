"""Reproducible three-way TrashNet training with MobileNetV2 transfer learning."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.initializers import Constant, HeNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Softmax,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

from waste_pipeline import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    TEST_FRACTION,
    VALIDATION_FRACTION,
    discover_dataset_splits,
    fit_temperature,
    make_dataset,
    save_evaluation_outputs,
    save_training_curves,
)


SEED = 42
DATASET_PATH = os.path.join("dataset", "trashnet")
MODEL_DIRECTORY = "saved_model"
BEST_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "best_model.keras")
LOG_DIRECTORY = "logs"
GRAPH_DIRECTORY = "graphs"
RESULTS_DIRECTORY = "results"

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 25
UNFREEZE_LAST_LAYERS = 50
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
L2_REGULARIZATION = 1e-4
HEAD_DROPOUT = 0.40
SECOND_DROPOUT = 0.30


def configure_runtime() -> None:
    """Set deterministic seeds and a device-appropriate compute policy."""
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(
            f"GPU detected ({len(gpu_devices)} device(s)); "
            "memory growth and mixed precision enabled."
        )
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("No GPU detected; using float32 CPU training.")


def build_model(num_classes: int) -> tuple[Model, tf.keras.Model]:
    """Create the inference-compatible model and its MobileNetV2 backbone."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = RandomFlip("horizontal", name="augment_flip")(inputs)
    x = RandomRotation(0.04, fill_mode="reflect", name="augment_rotation")(x)
    x = RandomZoom(0.08, fill_mode="reflect", name="augment_zoom")(x)
    x = RandomContrast(0.10, name="augment_contrast")(x)
    x = RandomBrightness(
        0.08,
        value_range=(-1.0, 1.0),
        name="augment_brightness",
    )(x)
    x = RandomTranslation(
        height_factor=0.05,
        width_factor=0.05,
        fill_mode="reflect",
        name="augment_translation",
    )(x)

    backbone = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        name="mobilenetv2",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = BatchNormalization(name="head_batch_normalization")(x)
    x = Dense(
        512,
        activation="relu",
        kernel_initializer=HeNormal(seed=SEED),
        kernel_regularizer=l2(L2_REGULARIZATION),
        name="dense_512",
    )(x)
    x = Dropout(HEAD_DROPOUT, seed=SEED, name="dropout_512")(x)
    x = Dense(
        256,
        activation="relu",
        kernel_initializer=HeNormal(seed=SEED + 1),
        kernel_regularizer=l2(L2_REGULARIZATION),
        name="dense_256",
    )(x)
    x = Dropout(SECOND_DROPOUT, seed=SEED + 1, name="dropout_256")(x)

    logits = Dense(
        num_classes,
        kernel_initializer=HeNormal(seed=SEED + 2),
        kernel_regularizer=l2(L2_REGULARIZATION),
        dtype="float32",
        name="logits",
    )(x)
    calibrated_logits = Dense(
        num_classes,
        use_bias=False,
        trainable=False,
        kernel_initializer=Constant(np.eye(num_classes, dtype=np.float32)),
        dtype="float32",
        name="temperature_scale",
    )(logits)
    outputs = Softmax(dtype="float32", name="output")(calibrated_logits)
    return Model(inputs, outputs, name="waste_classifier"), backbone


def compile_model(model: Model, learning_rate: float) -> None:
    """Compile with decoupled weight decay and stable multiclass loss."""
    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        ),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def training_callbacks(stage: str, checkpoint: ModelCheckpoint) -> list:
    """Return callbacks shared by both stages, preserving the global best."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=4 if stage == "head" else 6,
            restore_best_weights=True,
            start_from_epoch=2,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        checkpoint,
        TensorBoard(
            log_dir=os.path.join(LOG_DIRECTORY, stage),
            update_freq="epoch",
        ),
    ]


def main() -> None:
    configure_runtime()
    for directory in (
        MODEL_DIRECTORY,
        LOG_DIRECTORY,
        GRAPH_DIRECTORY,
        RESULTS_DIRECTORY,
    ):
        os.makedirs(directory, exist_ok=True)

    splits = discover_dataset_splits(
        DATASET_PATH,
        seed=SEED,
        test_fraction=TEST_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
    )
    if splits.class_names != CLASS_NAMES:
        raise ValueError(
            f"Class order differs from the inference contract: {splits.class_names}"
        )

    train_dataset = make_dataset(
        splits.train_paths,
        splits.train_labels,
        training=True,
        seed=SEED,
    )
    validation_dataset = make_dataset(
        splits.validation_paths,
        splits.validation_labels,
        training=False,
        seed=SEED,
    )
    test_dataset = make_dataset(
        splits.test_paths,
        splits.test_labels,
        training=False,
        seed=SEED,
    )

    counts = np.bincount(splits.train_labels, minlength=len(CLASS_NAMES))
    weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=splits.train_labels,
    )
    class_weights = {
        index: float(weight) for index, weight in enumerate(weight_values)
    }

    print("=" * 72)
    print(f"TensorFlow version             : {tf.__version__}")
    print(f"Training images                : {len(splits.train_paths)}")
    print(f"Validation images              : {len(splits.validation_paths)}")
    print(f"Held-out test images            : {len(splits.test_paths)}")
    print(
        f"Duplicate groups/files removed : "
        f"{splits.duplicate_groups_removed}/{splits.duplicate_files_removed}"
    )
    print(f"Class order                     : {list(splits.class_names)}")
    print("Training class counts and balanced weights:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name:<12}: {counts[index]:>4} "
            f"| weight {class_weights[index]:.3f}"
        )
    print("=" * 72)

    model, backbone = build_model(len(CLASS_NAMES))
    model.summary()
    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    compile_model(model, HEAD_LEARNING_RATE)
    print("\nStage 1/2: training the new classifier head.")
    try:
        head_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=HEAD_EPOCHS,
            class_weight=class_weights,
            callbacks=training_callbacks("head", checkpoint),
            shuffle=False,
            verbose=1,
        )
    except tf.errors.ResourceExhaustedError as error:
        raise RuntimeError(
            "TensorFlow ran out of memory during head training. "
            "Reduce BATCH_SIZE or close other GPU/CPU workloads."
        ) from error

    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-UNFREEZE_LAST_LAYERS:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    compile_model(model, FINE_TUNE_LEARNING_RATE)
    trainable_backbone_layers = sum(
        layer.trainable for layer in backbone.layers
    )
    print(
        "\nStage 2/2: fine-tuning "
        f"{trainable_backbone_layers} of the final "
        f"{UNFREEZE_LAST_LAYERS} backbone layers; BatchNormalization frozen."
    )
    try:
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=training_callbacks("fine_tune", checkpoint),
            shuffle=False,
            verbose=1,
        )
    except tf.errors.ResourceExhaustedError as error:
        raise RuntimeError(
            "TensorFlow ran out of memory during fine tuning. "
            "Reduce BATCH_SIZE or close other GPU/CPU workloads."
        ) from error

    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Training did not produce the best checkpoint: {BEST_MODEL_PATH}"
        )
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    logits_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer("logits").output,
        name="validation_logits",
    )
    validation_logits = logits_model.predict(validation_dataset, verbose=1)
    temperature = fit_temperature(
        validation_logits,
        splits.validation_labels,
    )
    model.get_layer("temperature_scale").set_weights(
        [np.eye(len(CLASS_NAMES), dtype=np.float32) / temperature]
    )
    model.save(BEST_MODEL_PATH)
    print(f"Validation-fitted temperature     : {temperature:.4f}")
    print(f"Best calibrated model saved       : {BEST_MODEL_PATH}")

    history = {}
    for metric in ("accuracy", "val_accuracy", "loss", "val_loss"):
        history[metric] = (
            head_history.history.get(metric, [])
            + fine_tune_history.history.get(metric, [])
        )
    save_training_curves(history, GRAPH_DIRECTORY)

    test_probabilities = model.predict(test_dataset, verbose=1)
    test_metrics = save_evaluation_outputs(
        splits.test_labels,
        test_probabilities,
        CLASS_NAMES,
        splits.test_paths,
        RESULTS_DIRECTORY,
    )
    print(f"\nTest accuracy                     : {test_metrics['accuracy'] * 100:.2f}%")
    print("Training, calibration, and held-out test evaluation completed.")


if __name__ == "__main__":
    main()
