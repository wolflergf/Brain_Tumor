import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Define constants
num_epochs = 20
batch_size = 32
img_size = (224, 224)
dataset_dir = "./brain_tumor_classification/Training"

# Check dataset existence
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

# Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2,  # Automatically split into train/validation
)

# Train and validation generators
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    subset="validation",
)


# Improved CNN model architecture
def create_model():
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(224, 224, 3)
            ),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(256, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),  # Prevent overfitting
            keras.layers.Dense(
                len(train_generator.class_indices), activation="softmax"
            ),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Train the model
def train_model(model):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=val_generator,
        callbacks=[early_stopping],
        verbose=1,
    )
    return history


# Evaluate the model
def evaluate_model(model):
    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    return loss, accuracy


# Plot training history
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    ax[0].plot(history.history["accuracy"], label="Train Accuracy")
    ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss
    ax[1].plot(history.history["loss"], label="Train Loss")
    ax[1].plot(history.history["val_loss"], label="Validation Loss")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.show()


# Model evaluation and classification report
def generate_classification_report(model):
    val_steps = val_generator.samples // val_generator.batch_size
    y_pred = model.predict(val_generator, steps=val_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes[: len(y_pred_classes)]  # Ground truth

    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred_classes,
            target_names=list(train_generator.class_indices.keys()),
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))


# Main workflow
model = create_model()
history = train_model(model)
evaluate_model(model)
plot_history(history)
generate_classification_report(model)

# Save the trained model
model.save("brain_tumor_model.keras")
