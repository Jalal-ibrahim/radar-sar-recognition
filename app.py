import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Title
st.title("🛰️ Radar (SAR) Target Recognition")
st.write("Upload a radar image to identify the military target.")

# 1. Model Load Karein
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mstar_defense_final_pro.keras', compile=False)

model = load_my_model()
classes = ['2S1', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4', 'SLICY']

# 2. Image Upload Button
uploaded_file = st.file_uploader("Choose a radar image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Image dikhayyein
    img = Image.open(uploaded_file).convert('L') # Gray scale
    st.image(img, caption='Uploaded Radar Image', use_column_width=True)
    
    # Preprocessing
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    # Prediction
    if st.button('Identify Target'):
        prediction = model.predict(img_array)
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display Result
        st.success(f"Target Identified: **{result}**")
        st.info(f"AI Confidence Score: **{confidence:.2f}%**")
