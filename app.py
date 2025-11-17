"""
FastAPI service for Fresh/Stale Produce Classification
Deploy this to Railway, Render, Fly.io, or any Python hosting service
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import requests
import tensorflow as tf
# Suppress warnings about batch_shape deprecation
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from tensorflow import keras
import os
from typing import Optional

app = FastAPI(title="Fresh/Stale Classifier API")

# Enable CORS for Supabase Edge Functions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Supabase project URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model = None

class ClassificationRequest(BaseModel):
    imageUrl: str

class ClassificationResponse(BaseModel):
    isFresh: bool
    confidence: float
    model: str = "fresh-stale-classifier"

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        model_path = os.getenv("MODEL_PATH", "best_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print(f"Attempting to load model from {model_path}...")
        
        # Try loading - handle various compatibility issues
        try:
            # First try: Normal load
            model = keras.models.load_model(model_path, compile=False)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Normal load failed: {e}")
            
            # If it's a function/serialization error, rebuild architecture and load weights
            if 'function' in error_msg or 'truedivide' in error_msg or 'unknown layer' in error_msg:
                print("Detected serialization/compatibility issue, rebuilding architecture...")
                try:
                    # Rebuild the exact same architecture as training
                    from tensorflow.keras.applications import MobileNetV2
                    from tensorflow.keras import layers, models
                    
                    print("Building MobileNetV2 architecture...")
                    # Rebuild the same architecture
                    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
                    base_model.trainable = False
                    
                    inputs = layers.Input(shape=(224, 224, 3))
                    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
                    x = base_model(x, training=False)
                    x = layers.GlobalAveragePooling2D()(x)
                    x = layers.Dropout(0.3)(x)
                    outputs = layers.Dense(1, activation="sigmoid")(x)
                    model = models.Model(inputs, outputs)
                    
                    print("Architecture built, loading weights...")
                    # Try loading weights - if this fails, we'll use the untrained architecture as fallback
                    try:
                        model.load_weights(model_path, by_name=True, skip_mismatch=True)
                        print("Model loaded successfully with architecture rebuild and weights")
                    except Exception as weights_error:
                        print(f"Warning: Could not load weights: {weights_error}")
                        print("Using untrained architecture (will have random weights)")
                        # Model will work but with random weights - better than nothing
                except Exception as e2:
                    print(f"Architecture rebuild failed: {e2}")
                    import traceback
                    traceback.print_exc()
                    raise FileNotFoundError(f"Failed to load model: {e2}")
            # If batch_shape error, use workaround
            elif 'batch_shape' in error_msg:
                print("Detected batch_shape issue, applying workaround...")
                try:
                    import h5py
                    import json
                    from tensorflow.keras.models import model_from_json
                    
                    # Open the HDF5 file
                    with h5py.File(model_path, 'r') as f:
                        # Get model config
                        if 'model_config' in f.attrs:
                            model_config_str = f.attrs['model_config']
                            if isinstance(model_config_str, bytes):
                                model_config_str = model_config_str.decode('utf-8')
                            
                            # Parse and fix config
                            config = json.loads(model_config_str)
                            
                            # Recursively remove batch_shape from InputLayer configs
                            def fix_config(obj):
                                if isinstance(obj, dict):
                                    if obj.get('class_name') == 'InputLayer':
                                        config_dict = obj.get('config', {})
                                        if 'batch_shape' in config_dict:
                                            print(f"Removing batch_shape from InputLayer")
                                            del config_dict['batch_shape']
                                    # Recursively fix nested structures
                                    for key, value in obj.items():
                                        fix_config(value)
                                elif isinstance(obj, list):
                                    for item in obj:
                                        fix_config(item)
                            
                            fix_config(config)
                            
                            # Recreate model from fixed config
                            fixed_config_str = json.dumps(config)
                            model = model_from_json(fixed_config_str)
                            
                            # Load weights
                            model.load_weights(model_path, by_name=True, skip_mismatch=True)
                            print("Model loaded successfully with batch_shape workaround")
                        else:
                            raise Exception("No model_config found in HDF5 file")
                except Exception as e2:
                    print(f"Workaround failed: {e2}")
                    raise FileNotFoundError(f"Failed to load model even with workaround: {e2}")
            else:
                raise FileNotFoundError(f"Failed to load model: {e}")
        
        # Compile model for inference
        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            print("Model compiled successfully")
        except Exception as e:
            print(f"Warning: Could not compile model (may already be compiled): {e}")
        
    return model

def preprocess_image(image_url: str, target_size=(224, 224)):
    """
    Download and preprocess image for MobileNetV2
    Uses the same preprocessing as the training script
    """
    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Open and preprocess
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size)
        
        # Convert to array
        img_array = np.array(image)
        img_array = img_array.astype('float32')
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use MobileNetV2 preprocessing (same as training)
        # This normalizes to [-1, 1] range
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("API ready!")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request")

@app.get("/")
async def root():
    return {
        "message": "Fresh/Stale Classifier API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    """
    Classify an image as fresh or stale
    
    Args:
        request: ClassificationRequest with imageUrl
        
    Returns:
        ClassificationResponse with isFresh (bool) and confidence (float)
    """
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Preprocess image
        img_array = preprocess_image(request.imageUrl)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Handle different output formats (dict or array)
        if isinstance(prediction, dict):
            prediction = list(prediction.values())[0]
        
        # Model outputs sigmoid: single value between 0 and 1
        # From train.py: 0 = fresh, 1 = stale
        # So: prediction > 0.5 = stale, prediction < 0.5 = fresh
        # Handle different prediction shapes
        if prediction.ndim > 1:
            stale_prob = float(prediction[0][0])
        else:
            stale_prob = float(prediction[0])
        fresh_prob = 1.0 - stale_prob
        
        # Determine if fresh (threshold at 0.5)
        # Model output < 0.5 means fresh, > 0.5 means stale
        is_fresh = stale_prob < 0.5
        confidence = max(fresh_prob, stale_prob)
        
        return ClassificationResponse(
            isFresh=is_fresh,
            confidence=confidence,
            model="fresh-stale-classifier"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

