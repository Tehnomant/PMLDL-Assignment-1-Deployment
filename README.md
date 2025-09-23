# ML Model Deployment Project

A complete machine learning model deployment using FastAPI and Streamlit with Docker.

## Project Structure

```
ml-deployment-project/
├── models/ # Saved ML models
├── code/
│ ├── models/ # Model training code
│ └── deployment/ # Deployment configuration
│ ├── api/ # FastAPI service
│ ├── app/ # Streamlit app
│ └── docker-compose.yml
├── data/ # Data files
└── requirements.txt # Python dependencies
```

## Quick Start

1. **Train the model:**
```bash
cd code/models
python train_model.py
```

2. **Deploy with Docker:**

```bash
cd code/deployment
docker-compose up --build
```

3. **Access the applications:**

    * Streamlit App: http://localhost:8501
    * FastAPI API: http://localhost:8000
    * API Docs: http://localhost:8000/docs

# Services

## FastAPI Service
* Port: 8000
* Endpoints:
    * ```GET /``` - Service status
    * ```GET /health``` - Health check
    * ```POST /predict``` - Make predictions
    * ```GET /model-info``` - Model information


## Streamlit Application
* Port: 8501
* Features: Interactive web interface for model predictions

# Development
## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python code/models/train_model.py

# Run API locally
cd code/deployment/api
uvicorn main:app --reload

# Run app locally
cd code/deployment/app
streamlit run app.py
```


## Step 6: Build and Run Script

**File: `deploy.sh`**

```bash
#!/bin/bash

echo "Starting ML Model Deployment..."

# Train the model first
echo "Step 1: Training the model..."
cd code/models
python train_model.py

# Build and run with docker-compose
echo "Step 2: Building and starting services..."
cd ../deployment
docker-compose down
docker-compose up --build

echo "Deployment complete!"
echo "Access the application at: http://localhost:8501"
echo "Access the API docs at: http://localhost:8000/docs"
```