import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# Load trained model
model = keras.models.load_model("isl_model.h5")

# Load class labels
with open("class_labels.txt", "r") as f:
    class_names = f.read().splitlines()

img_size = 64

def preprocess_frame(frame):
    """Preprocesses image for model prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
    return reshaped

def predict_image(image):
    """Predicts the class of an uploaded image."""
    processed_img = preprocess_frame(image)
    prediction = model.predict(processed_img)
    label = class_names[np.argmax(prediction)]
    return label

# Streamlit UI
st.title("Indian Sign Language Detection")
st.write("Upload an image or use your webcam for real-time sign language detection.")

# File Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Error loading image. Please upload a valid file.")
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            label = predict_image(image)
            st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Webcam Controls
st.write("Press 'Start Webcam' to detect signs in real-time.")

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

start = st.button("Start Webcam")
stop = st.button("Stop Webcam")

if start:
    st.session_state.webcam_running = True

if stop:
    st.session_state.webcam_running = False

# Webcam Stream
if st.session_state.webcam_running:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    try:
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image. Please check your webcam.")
                break

            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            label = class_names[np.argmax(prediction)]

            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")
    except Exception as e:
        st.error(f"Webcam Error: {str(e)}")
    finally:
        cap.release()
