"""Prediction utilities for Fashion MNIST classifier."""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from src.config import MODEL_PATH, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH


class FashionClassifier:
    """Fashion MNIST image classifier."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to trained model. Uses default if None.
        """
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.class_names = CLASS_NAMES
        
    def load_model(self):
        """Load the trained model."""
        if self.model is None:
            self.model = keras.models.load_model(self.model_path)
        return self.model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = np.mean(image, axis=-1)
        
        # Resize to 28x28
        if image.shape != (IMG_HEIGHT, IMG_WIDTH):
            pil_image = Image.fromarray(image.astype('uint8'))
            pil_image = pil_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            image = np.array(pil_image)
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        image = image.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        
        return image
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Predict the class of a clothing image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with prediction results
        """
        self.load_model()
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
        # Get results
        predicted_class_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_idx])
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            {
                'class_name': self.class_names[idx],
                'class_index': int(idx),
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_index': predicted_class_idx,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(self.class_names, predictions)
            }
        }
    
    def predict_batch(self, images: list) -> list:
        """
        Predict classes for multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in images]


# Global classifier instance for API use
_classifier = None

def get_classifier() -> FashionClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = FashionClassifier()
    return _classifier
