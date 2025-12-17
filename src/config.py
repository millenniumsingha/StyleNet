"""Configuration settings for the Fashion MNIST classifier."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model settings
MODEL_PATH = MODELS_DIR / "fashion_mnist_cnn.keras"
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10

# Training settings
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# Class names
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser", 
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
