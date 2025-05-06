import streamlit as st
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="Heart Attack Risk Assessment", layout="centered")
st.title("🫀 Heart Attack Risk Assessment via Retinal Scan (Demo)")
st.write("Upload a retinal image to simulate risk level prediction.")

# --- DUMMY PREDICTION FUNCTION ---
def fake_predict():
    classes = ["Low Risk", "Medium Risk", "High Risk"]
    prediction = random.choice(classes)
    confidence = round(random.uniform(75, 99), 2)
    return prediction, confidence

# --- DUMMY GRAD-CAM ---
def generate_dummy_heatmap():
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(224, 224), cmap='jet')
    ax.set_title("Dummy Grad-CAM Heatmap")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("📁 Upload a retinal image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="👁️ Uploaded Retina Image", use_column_width=True)

    # Simulate prediction
    predicted_class, confidence = fake_predict()

    st.markdown("---")
    st.subheader(f"🧠 Predicted Risk: {predicted_class}")
    st.write(f"Confidence: **{confidence}%**")

    # Simulate Grad-CAM
    st.markdown("### 🔍 Model Explanation (Simulated Grad-CAM)")
    gradcam_img = generate_dummy_heatmap()
    st.image(gradcam_img, caption="Simulated heatmap focus", use_column_width=True)
