# AI AGENT TASK: Upgrade Fashion MNIST Repository

## TASK SUMMARY
Transform https://github.com/millenniumsingha/StyleNet from a basic ML notebook into a production-ready application with CNN model, REST API, and web interface.

## EXECUTION STEPS

### PHASE 1: Setup Structure
```bash
# Create directories
mkdir -p src api app tests models notebooks legacy

# Move original notebook
mv StyleNet.ipynb legacy/

# Create placeholder files
touch src/__init__.py api/__init__.py tests/__init__.py models/.gitkeep
```

### PHASE 2: Create Core Files
Create these files with contents from the main guide (FashionMNIST_Upgrade_Guide.md):

**Priority 1 (Required for functionality):**
1. `requirements.txt`
2. `src/config.py`
3. `src/model.py`
4. `src/train.py`
5. `src/predict.py`
6. `api/main.py`
7. `api/schemas.py`
8. `app/streamlit_app.py`

**Priority 2 (Required for deployment):**
9. `Dockerfile`
10. `docker-compose.yml`
11. `.gitignore`
12. `.dockerignore`
13. `README.md`

**Priority 3 (Quality assurance):**
14. `tests/test_model.py`
15. `tests/test_api.py`

### PHASE 3: Validate
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (should achieve ~92% accuracy)
python -m src.train --model cnn --epochs 15

# Run tests
pytest tests/ -v

# Test API
uvicorn api.main:app --port 8000 &
curl http://localhost:8000/health

# Test Streamlit
streamlit run app/streamlit_app.py --server.port 8501
```

### PHASE 4: Commit
```bash
git add .
git commit -m "Upgrade to production-ready ML application

- Added CNN model architecture (92%+ accuracy)
- Added FastAPI REST API with /predict endpoint
- Added Streamlit web interface
- Added Docker containerization
- Added comprehensive tests
- Restructured project for production use
- Archived original notebook in legacy/"

git push origin master
```

## EXPECTED FINAL STRUCTURE
```
StyleNet/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .dockerignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_api.py
├── models/
│   └── .gitkeep
├── notebooks/
├── images/
│   └── (keep existing)
└── legacy/
    └── StyleNet.ipynb
```

## SUCCESS CRITERIA
- [ ] Model trains to 92%+ test accuracy
- [ ] API responds at http://localhost:8000/docs
- [ ] Streamlit loads at http://localhost:8501
- [ ] All tests pass
- [ ] Docker builds successfully
- [ ] README renders correctly on GitHub

## REFERENCE
See `FashionMNIST_Upgrade_Guide.md` for complete file contents.
