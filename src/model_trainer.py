import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        self.classes = sorted(os.listdir(dataset_dir))
        self.num_classes = len(self.classes)
        self.model = None
        self.history = None
        self.val_gen = None

    def clahe_preprocessing(self, img):
        """Wrapper to use CLAHE with ImageDataGenerator."""
        img_uint8 = img.astype(np.uint8)
        processed = apply_clahe(img_uint8)
        return processed.astype(np.float32)

    def get_data_generators(self):
        """Prepare train and validation generators with augmentation."""
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.clahe_preprocessing,
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

    def train(self, epochs: int = 30):
        """Trains the model with early stopping."""
        train_gen, val_gen = self.get_data_generators()
        self.val_gen = val_gen

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(self.num_classes),
            y=train_gen.classes
        )
        class_weight_dict = dict(enumerate(class_weights))

        self.model = build_model(self.num_classes)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e6, verbose=1
        )

        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict
        )
        return self.history

    def evaluate(self, val_gen=None):
        """Generates detailed metrics."""
        val_gen = val_gen or self.val_gen
        val_gen.reset()
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