# Fashion MNIST Project Upgrade Guide
## Complete Instructions for AI Agent Implementation

---

## PROJECT OVERVIEW

**Repository:** https://github.com/millenniumsingha/StyleNet

**Objective:** Transform a basic Fashion MNIST classification project into a production-ready ML application with:
1. Improved CNN model architecture (target: 92%+ accuracy)
2. FastAPI REST API backend
3. Streamlit interactive frontend
4. Docker containerization
5. Professional project structure
6. Updated dependencies and documentation

---

## STEP 1: CREATE NEW PROJECT STRUCTURE

Create the following directory structure:

```
StyleNet/
├── README.md                    # New professional README
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker compose for easy deployment
├── .gitignore                   # Git ignore file
├── .dockerignore                # Docker ignore file
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── model.py                 # CNN model definition
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction utilities
│   └── config.py                # Configuration settings
│
├── api/                         # FastAPI backend
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   └── schemas.py               # Pydantic schemas
│
├── app/                         # Streamlit frontend
│   └── streamlit_app.py         # Streamlit application
│
├── models/                      # Saved models directory
│   └── .gitkeep
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploration.ipynb     # Data exploration
│   └── 02_model_comparison.ipynb # Model comparison (original vs CNN)
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_model.py
│   └── test_api.py
│
├── images/                      # Keep existing images
│   └── (existing prediction images)
│
└── legacy/                      # Archive original work
    └── StyleNet.ipynb  # Original notebook
```

---

## STEP 2: FILE CONTENTS

### FILE: requirements.txt

```
# Core ML
tensorflow>=2.15.0
numpy>=1.24.0
scikit-learn>=1.3.0

# API
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6

# Frontend
streamlit>=1.31.0
pillow>=10.2.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Testing
pytest>=7.4.0
httpx>=0.26.0

# Utilities
pydantic>=2.5.0
python-dotenv>=1.0.0
```

---

### FILE: src/__init__.py

```python
"""Fashion MNIST Classification Package."""
__version__ = "2.0.0"
```

---

### FILE: src/config.py

```python
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
```

---

### FILE: src/model.py

```python
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
```

---

### FILE: src/train.py

```python
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
```

---

### FILE: src/predict.py

```python
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
```

---

### FILE: api/__init__.py

```python
"""FastAPI backend for Fashion MNIST classifier."""
```

---

### FILE: api/schemas.py

```python
"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Dict


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., description="Predicted class name")
    class_index: int = Field(..., description="Predicted class index (0-9)")
    confidence: float = Field(..., description="Confidence score (0-1)")


class PredictionResponse(BaseModel):
    """Full prediction response."""
    success: bool = Field(default=True)
    predicted_class: str = Field(..., description="Top predicted class")
    predicted_index: int = Field(..., description="Top predicted class index")
    confidence: float = Field(..., description="Confidence of top prediction")
    top_predictions: List[PredictionResult] = Field(..., description="Top 3 predictions")
    all_probabilities: Dict[str, float] = Field(..., description="All class probabilities")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    model_loaded: bool
    version: str


class ClassNamesResponse(BaseModel):
    """Class names response."""
    class_names: List[str]
    num_classes: int
```

---

### FILE: api/main.py

```python
"""FastAPI application for Fashion MNIST classifier."""

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.config import CLASS_NAMES, MODEL_PATH
from src.predict import get_classifier
from api.schemas import (
    PredictionResponse, 
    HealthResponse, 
    ClassNamesResponse,
    PredictionResult
)


# Initialize FastAPI app
app = FastAPI(
    title="Fashion MNIST Classifier API",
    description="""
    A production-ready API for classifying fashion items using a CNN model.
    
    ## Features
    - Upload images of clothing items
    - Get predictions with confidence scores
    - View all class probabilities
    
    ## Supported Classes
    T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        classifier = get_classifier()
        classifier.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Fashion MNIST Classifier API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health status."""
    classifier = get_classifier()
    model_loaded = classifier.model is not None
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version=__version__
    )


@app.get("/classes", response_model=ClassNamesResponse)
async def get_classes():
    """Get list of all class names."""
    return ClassNamesResponse(
        class_names=CLASS_NAMES,
        num_classes=len(CLASS_NAMES)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Image file to classify")):
    """
    Classify a clothing item image.
    
    Upload an image of a clothing item and get predictions for what type it is.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns prediction with confidence scores for all classes.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Get prediction
        classifier = get_classifier()
        result = classifier.predict(image_array)
        
        # Format response
        return PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            predicted_index=result['predicted_index'],
            confidence=result['confidence'],
            top_predictions=[
                PredictionResult(**pred) for pred in result['top_predictions']
            ],
            all_probabilities=result['all_probabilities']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Classify multiple clothing item images.
    
    Upload multiple images and get predictions for each.
    Maximum 10 images per request.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per request"
        )
    
    results = []
    classifier = get_classifier()
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            result = classifier.predict(image_array)
            results.append({
                "filename": file.filename,
                "success": True,
                **result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### FILE: app/streamlit_app.py

```python
"""Streamlit frontend for Fashion MNIST classifier."""

import streamlit as st
import numpy as np
from PIL import Image
import requests
import io
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES
from src.predict import FashionClassifier

# Page config
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Load the classifier model (cached)."""
    classifier = FashionClassifier()
    classifier.load_model()
    return classifier


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    return "confidence-low"


def main():
    # Header
    st.markdown('<p class="main-header">👗 Fashion MNIST Classifier</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a Convolutional Neural Network (CNN) to classify 
        fashion items into 10 categories.
        
        **Supported Classes:**
        """)
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{i}. {name}")
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.write("""
        - **Architecture**: CNN with 3 conv blocks
        - **Input Size**: 28x28 grayscale
        - **Accuracy**: ~92% on test set
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image of a clothing item",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Upload a grayscale or color image of a clothing item"
        )
        
        # Option to use sample images
        st.markdown("---")
        st.subheader("Or try a sample image:")
        
        use_sample = st.checkbox("Use Fashion MNIST sample")
        sample_index = st.slider("Sample index", 0, 999, 0) if use_sample else 0
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None or use_sample:
            try:
                # Load classifier
                classifier = load_classifier()
                
                if use_sample:
                    # Load sample from Fashion MNIST
                    from tensorflow import keras
                    (_, _), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
                    image_array = test_images[sample_index]
                    true_label = CLASS_NAMES[test_labels[sample_index]]
                    image = Image.fromarray(image_array)
                else:
                    # Load uploaded image
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    true_label = None
                
                # Display image
                st.image(image, caption="Input Image", width=200)
                
                # Get prediction
                with st.spinner("Classifying..."):
                    result = classifier.predict(image_array)
                
                # Display results
                st.markdown("### Prediction")
                confidence = result['confidence']
                conf_class = get_confidence_class(confidence)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{result['predicted_class']}</h2>
                    <p>Confidence: <span class="{conf_class}">{confidence:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                if true_label:
                    if result['predicted_class'] == true_label:
                        st.success(f"✅ Correct! True label: {true_label}")
                    else:
                        st.error(f"❌ Incorrect. True label: {true_label}")
                
                # Top predictions
                st.markdown("### Top 3 Predictions")
                for pred in result['top_predictions']:
                    progress = pred['confidence']
                    st.write(f"**{pred['class_name']}**")
                    st.progress(progress)
                    st.caption(f"{progress:.1%}")
                
                # All probabilities
                with st.expander("📊 All Class Probabilities"):
                    import pandas as pd
                    probs_df = pd.DataFrame([
                        {"Class": name, "Probability": prob}
                        for name, prob in result['all_probabilities'].items()
                    ]).sort_values("Probability", ascending=False)
                    st.bar_chart(probs_df.set_index("Class"))
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the model is trained. Run: `python -m src.train`")
        else:
            st.info("👈 Upload an image or select a sample to get started!")


if __name__ == "__main__":
    main()
```

---

### FILE: tests/__init__.py

```python
"""Test suite for Fashion MNIST classifier."""
```

---

### FILE: tests/test_model.py

```python
"""Tests for model architecture and training."""

import pytest
import numpy as np
from src.model import create_cnn_model, create_simple_model
from src.config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES


class TestModelArchitecture:
    """Test model creation and architecture."""
    
    def test_cnn_model_creation(self):
        """Test CNN model can be created."""
        model = create_cnn_model()
        assert model is not None
        
    def test_cnn_model_input_shape(self):
        """Test CNN model has correct input shape."""
        model = create_cnn_model()
        expected_shape = (None, IMG_HEIGHT, IMG_WIDTH, 1)
        assert model.input_shape == expected_shape
        
    def test_cnn_model_output_shape(self):
        """Test CNN model has correct output shape."""
        model = create_cnn_model()
        expected_shape = (None, NUM_CLASSES)
        assert model.output_shape == expected_shape
        
    def test_cnn_model_prediction(self):
        """Test CNN model can make predictions."""
        model = create_cnn_model()
        dummy_input = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 1)
        prediction = model.predict(dummy_input, verbose=0)
        
        assert prediction.shape == (1, NUM_CLASSES)
        assert np.isclose(prediction.sum(), 1.0, atol=1e-5)
        
    def test_simple_model_creation(self):
        """Test simple model can be created."""
        model = create_simple_model()
        assert model is not None
        
    def test_simple_model_output_shape(self):
        """Test simple model has correct output shape."""
        model = create_simple_model()
        expected_shape = (None, NUM_CLASSES)
        assert model.output_shape == expected_shape


class TestModelPredictions:
    """Test model prediction functionality."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        return np.random.rand(5, IMG_HEIGHT, IMG_WIDTH, 1).astype('float32')
    
    def test_batch_prediction(self, sample_images):
        """Test model can predict on batches."""
        model = create_cnn_model()
        predictions = model.predict(sample_images, verbose=0)
        
        assert predictions.shape == (5, NUM_CLASSES)
        
    def test_predictions_sum_to_one(self, sample_images):
        """Test softmax outputs sum to 1."""
        model = create_cnn_model()
        predictions = model.predict(sample_images, verbose=0)
        
        for pred in predictions:
            assert np.isclose(pred.sum(), 1.0, atol=1e-5)
```

---

### FILE: tests/test_api.py

```python
"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import io
from PIL import Image

from api.main import app
from src.config import CLASS_NAMES


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_response_format(self, client):
        """Test health response has required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestClassesEndpoint:
    """Test classes endpoint."""
    
    def test_classes_endpoint(self, client):
        """Test classes endpoint returns 200."""
        response = client.get("/classes")
        assert response.status_code == 200
        
    def test_classes_response_content(self, client):
        """Test classes response contains all classes."""
        response = client.get("/classes")
        data = response.json()
        
        assert data["class_names"] == CLASS_NAMES
        assert data["num_classes"] == len(CLASS_NAMES)


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_requires_file(self, client):
        """Test predict endpoint requires file upload."""
        response = client.post("/predict")
        assert response.status_code == 422
        
    def test_predict_rejects_non_image(self, client):
        """Test predict endpoint rejects non-image files."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
```

---

### FILE: Dockerfile

```dockerfile
# Fashion MNIST Classifier - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose ports
EXPOSE 8000 8501

# Default command (API server)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### FILE: docker-compose.yml

```yaml
version: '3.8'

services:
  # FastAPI Backend
  api:
    build: .
    container_name: fashion-mnist-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Streamlit Frontend
  streamlit:
    build: .
    container_name: fashion-mnist-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    command: streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      - api

  # Training service (run once)
  train:
    build: .
    container_name: fashion-mnist-train
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    command: python -m src.train --model cnn --epochs 15
    profiles:
      - training
```

---

### FILE: .gitignore

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Model files (large)
models/*.keras
models/*.h5
models/*.pkl
!models/.gitkeep

# TensorFlow
*.pb
checkpoint

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temp files
*.tmp
*.temp
```

---

### FILE: .dockerignore

```dockerignore
# Git
.git
.gitignore

# Documentation
*.md
!README.md

# IDE
.idea/
.vscode/

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.coverage

# Virtual environments
venv/
env/

# Jupyter
.ipynb_checkpoints/
notebooks/

# Large model files (rebuild in container)
models/*.keras
models/*.h5

# Tests (not needed in production)
tests/

# Legacy
legacy/
```

---

### FILE: README.md

```markdown
# Fashion MNIST Classifier 👗

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

A production-ready image classification system for the Fashion MNIST dataset, featuring a CNN model with ~92% accuracy, REST API, and interactive web interface.

![Demo](images/prediction_5.png)

## 🎯 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/millenniumsingha/StyleNet.git
cd StyleNet

# Train model and start services
docker-compose --profile training up train
docker-compose up -d api streamlit

# Access:
# - API Docs: http://localhost:8000/docs
# - Web App:  http://localhost:8501
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python -m src.train --model cnn --epochs 15

# Start API
uvicorn api.main:app --reload

# Start Web App (new terminal)
streamlit run app/streamlit_app.py
```

## 📊 Model Performance

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| Simple (Original) | ~88% | ~101K |
| **CNN (Current)** | **~92%** | **~400K** |

## 🏗️ Architecture

```
Input (28x28x1)
    │
    ▼
┌─────────────────┐
│ Conv Block 1    │  32 filters, BatchNorm, MaxPool, Dropout
└────────┬────────┘
         │
┌────────▼────────┐
│ Conv Block 2    │  64 filters, BatchNorm, MaxPool, Dropout
└────────┬────────┘
         │
┌────────▼────────┐
│ Conv Block 3    │  128 filters, BatchNorm, MaxPool, Dropout
└────────┬────────┘
         │
┌────────▼────────┐
│ Dense (256)     │  BatchNorm, Dropout
└────────┬────────┘
         │
┌────────▼────────┐
│ Output (10)     │  Softmax
└─────────────────┘
```

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/classes` | List class names |
| POST | `/predict` | Classify single image |
| POST | `/predict/batch` | Classify multiple images |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.png"
```

### Example Response

```json
{
  "success": true,
  "predicted_class": "Ankle boot",
  "predicted_index": 9,
  "confidence": 0.97,
  "top_predictions": [
    {"class_name": "Ankle boot", "class_index": 9, "confidence": 0.97},
    {"class_name": "Sneaker", "class_index": 7, "confidence": 0.02},
    {"class_name": "Sandal", "class_index": 5, "confidence": 0.01}
  ]
}
```

## 📁 Project Structure

```
├── src/                 # Core ML code
│   ├── model.py         # CNN architecture
│   ├── train.py         # Training script
│   └── predict.py       # Inference utilities
├── api/                 # FastAPI backend
│   └── main.py          # REST API endpoints
├── app/                 # Streamlit frontend
│   └── streamlit_app.py # Web interface
├── tests/               # Unit tests
├── models/              # Saved models
├── notebooks/           # Jupyter notebooks
└── legacy/              # Original project files
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=api
```

## 📈 Training Your Own Model

```bash
# CNN model (recommended)
python -m src.train --model cnn --epochs 15

# Simple model (for comparison)
python -m src.train --model simple --epochs 10
```

## 🎨 Supported Classes

| Index | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 🚀 Future Improvements

- [ ] Model versioning with MLflow
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Kubernetes deployment configs
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework

## 📜 License

MIT License

## 🙏 Acknowledgments

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research
- TensorFlow and Keras teams
- FastAPI and Streamlit communities

---

*Originally created as a learning project, upgraded to production-ready status.*
```

---

## STEP 3: IMPLEMENTATION INSTRUCTIONS FOR AI AGENT

1. **Clone the existing repository**
2. **Create the new directory structure** as shown in Step 1
3. **Move the original notebook** to `legacy/StyleNet.ipynb`
4. **Create all files** with the exact contents provided in Step 2
5. **Keep existing images** in the `images/` folder
6. **Create empty `models/.gitkeep`** file
7. **Test the setup:**
   ```bash
   pip install -r requirements.txt
   python -m src.train --model cnn --epochs 15
   pytest tests/ -v
   uvicorn api.main:app --reload
   # In new terminal:
   streamlit run app/streamlit_app.py
   ```
8. **Commit and push** all changes

---

## STEP 4: DEPLOYMENT OPTIONS

### Option A: Streamlit Cloud (Free, Easiest)
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo and deploy `app/streamlit_app.py`

### Option B: Hugging Face Spaces (Free)
1. Create new Space with Streamlit SDK
2. Upload project files
3. Auto-deploys

### Option C: Railway/Render (Free tier)
1. Connect GitHub repo
2. Deploy API service
3. Deploy Streamlit service

---

## VERIFICATION CHECKLIST

- [ ] All files created in correct locations
- [ ] requirements.txt has all dependencies
- [ ] Model trains successfully (~92% accuracy)
- [ ] All tests pass
- [ ] API starts and /docs works
- [ ] Streamlit app loads and classifies images
- [ ] Docker build succeeds
- [ ] README displays correctly on GitHub
