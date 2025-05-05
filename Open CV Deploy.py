import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧠 Face Detection App using OpenCV + Streamlit")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Option Selection
option = st.radio("Select input method:", ("📷 Upload Image", "🎥 Use Webcam"))

# --- Option 1: Image Upload ---
if option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

            st.image(img_array, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to process image: {e}")
    else:
        st.info("Please upload an image.")

# --- Option 2: Webcam ---
elif option == "🎥 Use Webcam":
    img_data = st.camera_input("Capture a photo")

    if img_data is not None:
        try:
            img = Image.open(img_data).convert("RGB")
            img_array = np.array(r"C:\Users\Samruddhi Yadav\Documents\Documents\Photo (2) - Copy.jpeg")

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

            st.image(img_array, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to process webcam image: {e}")
    else:
        st.info("Use camera to capture an image.")
