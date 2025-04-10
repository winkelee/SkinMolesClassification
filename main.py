import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

MODEL_PATH = 'D:/programming stuff/classificationBackend/savedmodel' #D:/programming stuff/classificationBackend
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["Benign", "Malignant"]

try:
    print(f"Loading model from: {MODEL_PATH}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model from {MODEL_PATH}")
    print(e)
    model = None 

app = FastAPI(title="Image Classifier API", version="1.0.0")

allowed_origins = ["http://localhost:5500", "http://127.0.0.1:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def preprocess_image(image_bytes: bytes):
    """Loads image from bytes, resizes, and normalizes it."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        #img_array = (img_array / 127.5) - 1.0
        img_batch = np.expand_dims(img_array, axis=0)
        return tf.cast(img_batch, tf.float32)

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file or format. Error: {e}")

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await file.read()

    preprocessed_image = preprocess_image(image_bytes)

        # +++ DEBUG LOGGING +++
    #print(f"Preprocessed image shape: {preprocessed_image.shape}")
    #print(f"Preprocessed image dtype: {preprocessed_image.dtype}")
    #print(f"Preprocessed image min value: {tf.reduce_min(preprocessed_image).numpy()}") 
    #print(f"Preprocessed image max value: {tf.reduce_max(preprocessed_image).numpy()}") 
    #print(f"Preprocessed image mean value: {tf.reduce_mean(preprocessed_image).numpy()}") 
    # +++ END DEBUG LOGGING +++

    try:
        predictions = model.predict(preprocessed_image)
        probability = float(predictions[0][0]) # Getting the FIRST element since we only have a batch of ONE

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed. Error: {e}")

    threshold = 0.5
    if probability >= threshold:
        predicted_class_index = 1
        confidence = probability
    else:
        predicted_class_index = 0
        confidence = 1.0 - probability


    predicted_class_name = CLASS_NAMES[predicted_class_index]

    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "prediction": {
            "predicted_class": predicted_class_name,
            "probability_raw": probability,
            "confidence": confidence,
            "class_index": predicted_class_index
        }
    })

#@app.get("/")
#async def read_root():
#    return {"message": "Welcome to the Image Classifier API!"}

if __name__ == "__main__":
    #Have set this up as 127.0.0.1 so that the server can only be accessed from the local machine.
    #This is because this is a local application, not a web page or a SaaS.
    #reload=True automatically restarts the server when code changes (for development)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)