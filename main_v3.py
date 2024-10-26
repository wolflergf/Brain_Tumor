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
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=30,
    zoom_range=[0.5, 1],
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    dataset_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,  # Shuffle should be False to match predictions with labels
)

# Create a CNN model
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 3)
        ),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(train_generator.class_indices), activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_generator, epochs=num_epochs, validation_data=val_generator, verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Loss: {loss:.3f}")
print(f"Accuracy: {accuracy:.3f}")

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history.history["loss"], label="Training Loss")
plt.legend()
plt.show()


# Generate classification report
def generate_classification_report(model, val_generator):
    # Predict class probabilities for validation set
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true classes from the validation generator
    true_classes = val_generator.classes

    # Get class labels (reverse the class indices map)
    class_labels = list(val_generator.class_indices.keys())

    # Generate the classification report
    report = classification_report(
        true_classes, predicted_classes, target_names=class_labels, zero_division=0
    )

    return report


report = generate_classification_report(model, val_generator)
print(report)

# Save the trained model
model.save("brain_tumor_model.keras")
