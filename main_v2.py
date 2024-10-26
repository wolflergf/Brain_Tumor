import os

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
    """
    Create a convolutional neural network (CNN) model.

    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential(
        [
            # Convolutional layer with ReLU activation
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(224, 224, 3)
            ),
            keras.layers.MaxPooling2D((2, 2)),
            # Convolutional layer with ReLU activation
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            # Convolutional layer with ReLU activation
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            # Convolutional layer with ReLU activation
            keras.layers.Conv2D(256, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            # Flatten the output
            keras.layers.Flatten(),
            # Dense layer with ReLU activation
            keras.layers.Dense(256, activation="relu"),
            # Dropout layer to prevent overfitting
            keras.layers.Dropout(0.5),
            # Output layer with softmax activation for multi-class classification
            keras.layers.Dense(
                len(train_generator.class_indices), activation="softmax"
            ),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam",  # Adam optimizer
        loss="sparse_categorical_crossentropy",  # Sparse categorical cross-entropy loss
        metrics=["accuracy"],  # Accuracy metric
    )

    return model


# Train the model
def train_model(model):
    """
    Train a Keras model using the training generator.

    Args:
        model (keras.Model): The model to be trained.

    Returns:
        A history object containing the training and validation losses and accuracies.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=5,  # Wait for 5 epochs without improvement
        restore_best_weights=True,  # Restore the best weights
    )

    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=val_generator,
        callbacks=[early_stopping],
        verbose=1,  # Display training and validation progress
    )

    return history


# Evaluate the model
def evaluate_model(model):
    """
    Evaluate a Keras model using the validation generator.

    Args:
        model (keras.Model): The model to be evaluated.

    Returns:
        A tuple containing the validation loss and accuracy.
    """
    loss, accuracy = model.evaluate(val_generator)

    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

    return loss, accuracy


# Plot training history
def plot_history(history):
    """
    Plot the training and validation losses and accuracies over epochs.

    Args:
        history (keras.callbacks.History): A history object containing the training and validation losses and accuracies.
    """
    import matplotlib.pyplot as plt

    # Plot training and validation losses
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    # Plot training and validation accuracies
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.title("Training and Validation Losses and Accuracies")
    plt.show()


# Generate classification report
def generate_classification_report(model, val_generator):
    """
    Generate a classification report using the validation generator.

    Args:
        model (keras.Model): The model to be evaluated.
        val_generator (keras.utils.TensorArray): The validation generator.

    Returns:
        A dictionary containing the classification metrics.
    """
    predictions = model.predict(val_generator)
    _, predicted = np.argmax(predictions, axis=1), np.argmax(
        val_generator.labels, axis=1
    )

    report = classification_report(np.array(predicted), val_generator.classes)

    return report


# Main function
def main():
    # Create a CNN model
    model = create_model()

    # Train the model
    history = train_model(model)

    # Evaluate the model
    loss, accuracy = evaluate_model(model)

    # Plot training history
    plot_history(history)

    # Generate classification report
    report = generate_classification_report(model, val_generator)

    print(report)


if __name__ == "__main__":
    main()
