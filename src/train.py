"""Training script for Fashion MNIST classifier."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    BATCH_SIZE, EPOCHS, MODEL_PATH, MODELS_DIR,
    VALIDATION_SPLIT, CLASS_NAMES
)
from src.model import create_cnn_model, create_simple_model


def load_data():
    """Load and preprocess Fashion MNIST dataset."""
    (train_images, train_labels), (test_images, test_labels) = \
        keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Add channel dimension for CNN
    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]
    
    return (train_images, train_labels), (test_images, test_labels)


def train_model(model_type: str = 'cnn', epochs: int = EPOCHS) -> dict:
    """
    Train the Fashion MNIST classifier.
    
    Args:
        model_type: 'cnn' for improved model, 'simple' for legacy model
        epochs: Number of training epochs
        
    Returns:
        Dictionary with training history and evaluation metrics
    """
    print(f"Loading Fashion MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    print(f"Training set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    
    # Create model
    if model_type == 'cnn':
        print("\nCreating CNN model...")
        model = create_cnn_model()
    else:
        print("\nCreating simple model...")
        model = create_simple_model()
        # Reshape for simple model (no channel dimension needed)
        train_images = train_images.squeeze()
        test_images = test_images.squeeze()
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model
    model_save_path = MODEL_PATH if model_type == 'cnn' else \
        MODELS_DIR / "fashion_mnist_simple.keras"
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Save training metadata
    metadata = {
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'epochs_trained': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'class_names': CLASS_NAMES
    }
    
    metadata_path = model_save_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return {
        'history': history.history,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_path': str(model_save_path)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fashion MNIST classifier')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'simple'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    train_model(args.model, args.epochs)
