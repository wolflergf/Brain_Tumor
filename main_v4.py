import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class BrainTumorClassifier:
    def __init__(
        self,
        img_size=(160, 160),  # Reduced image size for CPU
        batch_size=16,  # Reduced batch size for CPU
        num_epochs=20,
        learning_rate=1e-4,
        dataset_dir="./brain_tumor_classification/Training",
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset_dir = Path(dataset_dir)

        # Disable mixed precision for CPU
        tf.keras.mixed_precision.set_global_policy("float32")

    def setup_data_generators(self):
        """Setup data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.2,
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )

        self.val_generator = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

    def build_model(self):
        """Build a lighter model suitable for CPU training"""
        model = keras.Sequential(
            [
                # First Convolution Block
                keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=(self.img_size[0], self.img_size[1], 3),
                ),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                # Second Convolution Block
                keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                # Third Convolution Block
                keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                # Dense Layers
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(
                    len(self.train_generator.class_indices), activation="softmax"
                ),
            ]
        )

        # Learning rate schedule
        initial_learning_rate = self.learning_rate
        decay_steps = 1000
        decay_rate = 0.9
        learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model
        return model

    def train(self):
        """Train the model with callbacks"""
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                "best_model.keras",  # Changed from .h5 to .keras
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
            ),
        ]

        steps_per_epoch = len(self.train_generator)
        validation_steps = len(self.val_generator)

        self.history = self.model.fit(
            self.train_generator,
            epochs=self.num_epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,
        )

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(self.history.history["accuracy"], label="Training Accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(self.history.history["loss"], label="Training Loss")
        ax2.plot(self.history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        predictions = self.model.predict(self.val_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.val_generator.classes

        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    def generate_classification_report(self):
        """Generate and print classification report"""
        predictions = self.model.predict(self.val_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.val_generator.classes
        class_labels = list(self.val_generator.class_indices.keys())

        report = classification_report(
            true_classes, predicted_classes, target_names=class_labels, zero_division=0
        )
        logger.info("\nClassification Report:\n%s", report)
        return report

    def save_model(self, filepath="brain_tumor_model.keras"):
        """Save the trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")


def main():
    # Initialize classifier
    classifier = BrainTumorClassifier()

    # Setup and train
    classifier.setup_data_generators()
    classifier.build_model()
    classifier.train()

    # Evaluate and visualize results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix()
    classifier.generate_classification_report()

    # Save the model
    classifier.save_model()


if __name__ == "__main__":
    main()
