"""FastAPI application for Fashion MNIST classifier."""

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.config import CLASS_NAMES, MODEL_PATH, MODELS_DIR
from src.predict import get_classifier
from api.schemas import (
    PredictionResponse, 
    HealthResponse, 
    ClassNamesResponse,
    PredictionResult
)

# Monitoring setup
import csv
from pathlib import Path
from datetime import datetime

MONITORING_DIR = Path("monitoring")
MONITORING_DIR.mkdir(exist_ok=True)
CURRENT_DATA_PATH = MONITORING_DIR / "current_data.csv"

def log_prediction(confidence: float, predicted_index: int):
    """Log prediction metadata for monitoring."""
    file_exists = CURRENT_DATA_PATH.exists()
    
    with open(CURRENT_DATA_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'confidence', 'predicted_class_index'])
        
        writer.writerow([
            datetime.now().isoformat(),
            confidence,
            predicted_index
        ])

import os
import random

# A/B Testing Configuration
AB_TEST_RATIO = float(os.getenv("AB_TEST_RATIO", "0.0"))
CANARY_MODEL_PATH = MODELS_DIR / "canary_model.keras"

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
    """Load models on startup."""
    try:
        # Load stable model
        classifier = get_classifier()
        classifier.load_model()
        print("Stable model loaded successfully!")
        
        # Load canary model if ratio > 0
        if AB_TEST_RATIO > 0:
            if CANARY_MODEL_PATH.exists():
                # We need a way to store the canary classifier. 
                # For simplicity, we'll attach it to the app state or a global dict
                # But since get_classifier is a singleton pattern, we might need to adjust it
                # For this implementation, we'll instantiate a second classifier
                from src.predict import FashionClassifier
                global canary_classifier
                canary_classifier = FashionClassifier(model_path=CANARY_MODEL_PATH)
                canary_classifier.load_model()
                print(f"Canary model loaded! Traffic ratio: {AB_TEST_RATIO}")
            else:
                print(f"Warning: A/B ratio is {AB_TEST_RATIO} but canary model not found at {CANARY_MODEL_PATH}")
                
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
        
        # A/B Routing Logic
        model_version = "stable"
        if AB_TEST_RATIO > 0 and 'canary_classifier' in globals() and canary_classifier.model:
            if random.random() < AB_TEST_RATIO:
                classifier = canary_classifier
                model_version = "canary"
        
        result = classifier.predict(image_array)
        
        # Log for monitoring
        log_prediction(result['confidence'], result['predicted_index'])

        # Format response
        # Log for monitoring
        log_prediction(result['confidence'], result['predicted_index'])

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
