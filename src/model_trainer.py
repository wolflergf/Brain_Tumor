import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .model_factory import build_model
from .preprocessing import apply_clahe

class BrainTumorTrainer:
    """
    Handles the training pipeline, evaluation, and reporting.
    """
    def __init__(self, dataset_dir: str, img_size: tuple = (224, 224), batch_size: int = 32):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = sorted(os.listdir(os.path.join(dataset_dir)))
        self.num_classes = len(self.classes)
        self.model = None
        self.history = None

    def clahe_preprocessing(self, img):
        """Wrapper to use CLAHE with ImageDataGenerator."""
        img_uint8 = (img.astype(np.uint8))
        processed = apply_clahe(img_uint8)
        return processed.astype(np.float32)

    def get_data_generators(self):
        """Prepare train and validation generators with augmentation."""
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.clahe_preprocessing,
            rescale=1.0/255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_gen = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset='training'
        )

        val_gen = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False
        )

        return train_gen, val_gen

    def train(self, epochs: int = 20):
        """Trains the model with early stopping."""
        train_gen, val_gen = self.get_data_generators()
        self.model = build_model(self.num_classes)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[early_stopping]
        )
        return self.history

    def evaluate(self, val_gen):
        """Generates detailed metrics."""
        y_pred_probs = self.model.predict(val_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = val_gen.classes

        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=self.classes))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'.")

    def save_model(self, path: str = "brain_tumor_model_v2.keras"):
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
