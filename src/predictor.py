import numpy as np
import tensorflow as tf
from PIL import Image
from .preprocessing import preprocess_image

class TumorPredictor:
    """
    Handles model loading and inference for single images.
    """
    def __init__(self, model_path: str = "brain_tumor_model.keras"):
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def predict(self, img: Image.Image) -> dict:
        """
        Preprocesses and predicts the class of an input image.
        
        Args:
            img: PIL Image object.
            
        Returns:
            Dictionary with the predicted class and probabilities.
        """
        try:
            # Preprocess the image
            processed_img = preprocess_image(img)
            img_array = np.expand_dims(processed_img, axis=0)

            # Perform prediction
            prediction = self.model.predict(img_array, verbose=0)
            pred_class_idx = np.argmax(prediction[0])
            
            # Map probabilities to class names
            probs = {self.classes[i]: float(prediction[0][i]) for i in range(len(self.classes))}
            
            return {
                "class": self.classes[pred_class_idx],
                "probabilities": probs,
                "processed_img": processed_img  # For real-time feedback
            }
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
