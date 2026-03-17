import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page configuration (Ye hamesha sab se upar hoti hai)
st.set_page_config(page_title="Jalal Radar AI", page_icon="🛰️", layout="centered")

# 2. Custom Header
st.title("🛰️ Radar SAR Recognition System")
st.subheader("Developed by: Jalal Ibrahim | SAR Image Expert")
st.markdown("---")

# 3. Model Loading Function
@st.cache_resource
def load_my_model():
    try:
        # Aapki file ka sahi naam
        model = tf.keras.models.load_model('mstar_defense_final_pro.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Model load karne mein masla aaya: {e}")
        return None

model = load_my_model()

# 4. Image Processing Function
def process_image(img):
    # Agar aapka model 64x64 par train hua hai toh yahan 64,64 kar dein
    size = (64, 64) 
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image = image.convert('L') # Gray scale
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array.reshape(1, 64, 64, 1)
    return img_reshape

# 5. Main App UI
st.sidebar.title("Settings")
st.sidebar.info("MSTAR Dataset AI Model")

uploaded_file = st.file_uploader("SAR Image File Upload Karein...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Radar Image', use_container_width=True)
    
    with col2:
        st.write("### Analysis Result")
        if st.button('Identify Target'):
            if model is not None:
                # Classes Labels
                class_names = ['2S1', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']
                
                with st.spinner('Processing...'):
                    processed_img = process_image(image)
                    prediction = model.predict(processed_img)
                    result = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                
                st.success(f"**Target: {result}**")
                st.info(f"Confidence: {confidence:.2f}%")
            else:
                st.error("Model file load nahi ho saki.")

st.markdown("---")
st.caption("© 2026 Radar SAR Recognition - Jalal Ibrahim")
