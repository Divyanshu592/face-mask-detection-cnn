import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/saved_model/mask_detector"
IMAGE_SIZE = 224
CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]

model = tf.keras.models.load_model(MODEL_PATH)

st.title("Face Mask Detection")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)

    box = preds["boxes"][0]
    
    probs = preds["labels"][0]
    label_idx = int(np.argmax(probs))
    confidence = float(probs[label_idx])
    label_name = CLASS_NAMES[label_idx]

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        ### 🧠 Detection Result
        **Mask Status:** `{label_name}`  
        **Confidence:** `{confidence:.2f}`
        """
    )

    if confidence < 0.5:
        st.warning("Low confidence prediction. Try a clearer image.")

