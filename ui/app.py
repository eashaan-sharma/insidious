import streamlit as st
from PIL import Image

st.title("AI-Assisted Pneumonia Detection")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded X-ray", use_column_width=True)
    st.success("Prediction: Pneumonia (Demo)")
    st.info("Model: Swin Transformer (Demo Prediction)")
    st.write("Streamlit is running correctly")


