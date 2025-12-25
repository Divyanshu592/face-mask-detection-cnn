import tensorflow as tf
import os

MODEL_PATH = "models/saved_model/mask_detector"
TFLITE_DIR = "models/tflite"
TFLITE_PATH = os.path.join(TFLITE_DIR, "mask_detector.tflite")

os.makedirs(TFLITE_DIR, exist_ok=True)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("Saved TFLite model:", TFLITE_PATH)
