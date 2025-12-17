"""CNN Model architecture for Fashion MNIST classification."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, LEARNING_RATE


def create_cnn_model() -> keras.Model:
    """
    Create an improved CNN model for Fashion MNIST classification.
    
    Architecture:
    - 3 Convolutional blocks with BatchNorm and MaxPooling
    - Dropout for regularization
    - Dense layers for classification
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_simple_model() -> keras.Model:
    """
    Create the original simple model for comparison.
    This is the legacy 3-layer model from the original project.
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH)),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """Get model summary as string."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)
