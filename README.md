# 🩺 Diabetes Prediction API

A local FastAPI-based REST API for diabetes classification using a trained machine learning model. This project takes your existing `.py` model training script and wraps it into a production-ready API for local testing and deployment.

---

## 📁 Project Structure

```
.
├── diabetes_unclean.csv          # Your raw dataset
├── train_and_save.py             # Training script — cleans data, trains model, saves artifacts
├── api.py                        # FastAPI server — loads artifacts and serves predictions
├── test_api.py                   # Simple Python client to test the API
├── requirements.txt              # Python dependencies
├── diabetes_model.pkl            # Trained model (generated)
├── scaler.pkl                    # Fitted StandardScaler (generated)
├── label_encoder.pkl             # Class label encoder (generated)
└── feature_columns.pkl           # Ordered feature names (generated)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install fastapi uvicorn numpy pandas scikit-learn joblib
```

### 2. Train and Save the Model

Place your `diabetes_unclean.csv` in the project root, then run:

```bash
python train_and_save.py
```

This will:
- Clean and preprocess the data
- Handle missing values and outliers
- Encode categorical variables
- Train a `RandomForestClassifier`
- Save 4 artifacts (`*.pkl` files) for the API to use

### 3. Start the API Server

```bash
python api.py
```

The server starts at: **`http://localhost:8000`**

### 4. Test the API

**Option A — Using the test script:**
```bash
python test_api.py
```

**Option B — Using `curl`:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [45.0, 4.7, 75.0, 6.5, 4.2, 1.8, 1.2, 2.5, 0.8, 28.5, 1.0]}'
```

**Option C — Interactive Swagger UI:**
Open your browser and go to: **`http://localhost:8000/docs`**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Health check — confirms model is loaded |
| `GET` | `/feature-info` | Returns expected feature names and order |
| `POST` | `/predict` | Submit features, get prediction + probabilities |

### POST `/predict` — Request Body

```json
{
  "features": [45.0, 4.7, 75.0, 6.5, 4.2, 1.8, 1.2, 2.5, 0.8, 28.5, 1.0]
}
```

> ⚠️ **Important:** Features must be provided in the exact order returned by `/feature-info`. The API validates the count and returns a clear error if they don't match.

### POST `/predict` — Response

```json
{
  "prediction": 1,
  "prediction_label": "Y",
  "probability": [0.12, 0.88],
  "input_features": [45.0, 4.7, 75.0, 6.5, 4.2, 1.8, 1.2, 2.5, 0.8, 28.5, 1.0],
  "feature_names": ["AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI", "Gender"]
}
```

---

## 🛠️ How It Works

### Training Pipeline (`train_and_save.py`)

| Step | Action |
|------|--------|
| Load Data | Reads `diabetes_unclean.csv` |
| Clean | Drops duplicates, fills missing values |
| Encode | Converts `Gender` (M/F → 1/0) |
| Split | 75% train / 25% test |
| Scale | `StandardScaler` for numerical features |
| Train | `RandomForestClassifier` (n_estimators=200) |
| Evaluate | Prints Accuracy and F1 Score |
| Save | Exports model, scaler, encoder, and column names as `.pkl` files |

### API Server (`api.py`)

| Step | Action |
|------|--------|
| Startup | Loads all 4 `.pkl` artifacts into memory |
| Request | Validates feature count and order |
| Transform | Scales input using the saved scaler |
| Predict | Runs inference and returns class + probabilities |
| Response | Returns structured JSON with labels and metadata |

---

## 🔄 Retraining the Model

If you update your dataset or want to try a different model:

1. Replace or update `diabetes_unclean.csv`
2. Run `python train_and_save.py` again
3. Restart the API server (it loads artifacts on startup)

No code changes needed in `api.py` — it dynamically reads whatever `feature_columns.pkl` contains.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: diabetes_unclean.csv` | Ensure your CSV is in the same folder as `train_and_save.py` |
| `Expected N features, got M` | Check `/feature-info` and send values in the correct order |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 8000 already in use | Change the port in `api.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)` |
| Model accuracy seems low | Tune `RandomForestClassifier` hyperparameters in `train_and_save.py` |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the API |
| `uvicorn` | ASGI server to run FastAPI |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, metrics |
| `joblib` | Efficient model serialization |

---

## 📝 Notes

- This is designed for **local testing and development**. For production, consider adding authentication, rate limiting, and HTTPS.
- The model is saved using `joblib` (more efficient than `pickle` for large numpy arrays).
- The API auto-generates interactive documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## 📄 License

This project is for educational and local testing purposes.
