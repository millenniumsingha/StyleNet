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
