import streamlit as st
import requests
import numpy as np
import json

# App configuration
st.set_page_config(
    page_title="ML Model Demo",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("Machine Learning Model Demo")
st.markdown("""
This application demonstrates a machine learning model deployment using FastAPI and Streamlit.
Enter feature values below to get predictions from the model.
""")

# API configuration
API_URL = "http://api:8000"  # Docker service name
# API_URL = "http://localhost:8000" # local name
# Sidebar for information
st.sidebar.title("Model Information")
st.sidebar.markdown("""
- **Model Type**: Random Forest Classifier
- **Input Features**: 4 numerical features
- **Output**: Binary classification (0 or 1)
""")

# Check API connection
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        st.sidebar.success("✅ API is connected")
    else:
        st.sidebar.error("❌ API connection failed")
except:
    st.sidebar.error("❌ Cannot connect to API")

# Get model info
try:
    model_info = requests.get(f"{API_URL}/model-info").json()
    feature_names = model_info.get('feature_names', ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
except:
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']

# Feature input section
st.header("Feature Input")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Feature Values")
    features = []
    for i, name in enumerate(feature_names):
        feature_value = st.number_input(
            f"{name}",
            value=0.0,
            key=f"feature_{i}",
            format="%.3f"
        )
        features.append(feature_value)

with col2:
    st.subheader("Feature Statistics")
    if features:
        features_array = np.array(features)
        st.metric("Mean", f"{features_array.mean():.3f}")
        st.metric("Standard Deviation", f"{features_array.std():.3f}")
        st.metric("Minimum", f"{features_array.min():.3f}")
        st.metric("Maximum", f"{features_array.max():.3f}")

# Prediction section
st.header("Prediction")

if st.button("Get Prediction", type="primary"):
    if len(features) != 4:
        st.error("Please enter exactly 4 feature values")
    else:
        try:
            # Prepare request
            payload = {"features": features}
            
            # Make prediction request
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("Prediction completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction", 
                        "Class 1" if result['prediction'] == 1 else "Class 0",
                        delta="Positive" if result['prediction'] == 1 else "Negative"
                    )
                
                with col2:
                    st.metric(
                        "Probability", 
                        f"{result['probability']:.4f}",
                        delta=f"{(result['probability'] - 0.5) * 100:+.1f}%"
                    )
                
                with col3:
                    confidence = result['probability'] if result['prediction'] == 1 else 1 - result['probability']
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                
                # Show probability bar
                st.subheader("Probability Distribution")
                prob_class_0 = 1 - result['probability']
                prob_class_1 = result['probability']
                
                st.write(f"**Class 0**: {prob_class_0:.4f}")
                st.progress(float(prob_class_0))
                
                st.write(f"**Class 1**: {prob_class_1:.4f}")
                st.progress(float(prob_class_1))
                
            else:
                st.error(f"Prediction failed: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            st.info("Make sure the API service is running")

# Feature importance visualization (static for demo)
st.header("Model Information")
st.markdown("""
This demo uses a Random Forest classifier trained on synthetic data with 4 features.

**How to use:**
1. Enter values for all 4 features
2. Click the 'Get Prediction' button
3. View the prediction results and probabilities

The model returns:
- **Prediction**: The predicted class (0 or 1)
- **Probability**: The probability of class 1
- **Confidence**: How confident the model is in its prediction
""")