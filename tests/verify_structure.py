import sys
from unittest.mock import MagicMock

# MOCK TENSORFLOW AND OTHER ML LIBS
# This allows us to import and test the API structure without installing heavy ML deps on ARM64
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()
sys.modules["tensorflow.keras.layers"] = MagicMock()

# Mock numpy if it wasn't installed (though it likely was, checking anyway)
try:
    import numpy
except ImportError:
    sys.modules["numpy"] = MagicMock()

# Now we can import the app
from api.main import app
from fastapi.testclient import TestClient
from api.schemas import HealthResponse

client = TestClient(app)

def test_health_check_mocked():
    """Test that the API health endpoint works even if model fails to load."""
    print("Testing /health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print(f"Health Response: {data}")
    assert data["status"] == "healthy"
    # Model won't be loaded because we mocked TF, so verification passed if this runs

def test_classes_endpoint():
    """Test that the classes endpoint returns correct data."""
    print("Testing /classes endpoint...")
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    print(f"Classes Response: {data}")
    assert len(data["class_names"]) == 10

if __name__ == "__main__":
    print("Running Compatible Verification...")
    try:
        test_health_check_mocked()
        test_classes_endpoint()
        print("\nSUCCESS: API Structure and Logic verified!")
    except Exception as e:
        print(f"\nFAILURE: Verification failed with error: {e}")
        sys.exit(1)
