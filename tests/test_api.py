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
