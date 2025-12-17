# Fashion MNIST Classifier 👗

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

A production-ready image classification system for the Fashion MNIST dataset, featuring a CNN model with ~92% accuracy, REST API, and interactive web interface.

## 🔴 Live Demo

You can deploy this application directly to Streamlit Cloud to see it in action:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

*Note: After deploying, update the link above to your specific app URL.*

![Demo](images/prediction_5.png)

## 🎯 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/millenniumsingha/ImageClassification_FashionMNIST.git
cd ImageClassification_FashionMNIST

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
