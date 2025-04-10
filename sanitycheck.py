# THIS SCRIPT IS FOR CHECKING FOR PROBLEMS NOT RELATED TO THE API
import tensorflow as tf
from PIL import Image
import numpy as np
import io

MODEL_PATH = 'D:/programming stuff/classificationBackend/savedmodel'
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_PATH_BENIGN = 'D:/programming stuff/ml cnn/data/test/benign/5.jpg'
IMAGE_PATH_MALIGNANT = 'D:/programming stuff/ml cnn/data/test/malignant/8.jpg'

# --- Exact copy of the function from the server ---
def preprocess_image(image_bytes: bytes):
    """Loads image from bytes, resizes, and normalizes it."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        #img_array = (img_array / 127.5) - 1.0 Is the problem with this line? The rescaling layer is a part of the model.
        img_batch = np.expand_dims(img_array, axis=0)
        return tf.cast(img_batch, tf.float32)

    except Exception as e:
        print(f"Error preprocessing image: {e}")
# --- End copied function ---

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print("\nPredicting Benign Image:")
with open(IMAGE_PATH_BENIGN, 'rb') as f:
     img_bytes = f.read()
preprocessed_benign = preprocess_image(img_bytes)
pred_benign = model.predict(preprocessed_benign)
print(f"Raw probability (Benign): {float(pred_benign[0][0]):.6f}")

print("\nPredicting Malignant Image:")
with open(IMAGE_PATH_MALIGNANT, 'rb') as f:
     img_bytes = f.read()
preprocessed_malignant = preprocess_image(img_bytes)
pred_malignant = model.predict(preprocessed_malignant)
print(f"Raw probability (Malignant): {float(pred_malignant[0][0]):.6f}")