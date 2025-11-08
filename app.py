import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("fire_detection_model.keras")

st.title("🔥 Fire Detection Web App")
st.write("Upload an image and I’ll tell you if fire is detected or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.markdown(f"🔥 **Fire detected!** (Confidence: {pred*100:.2f}%)")
    else:
        st.markdown(f"🌳 **No Fire detected.** (Confidence: {(1-pred)*100:.2f}%)")
