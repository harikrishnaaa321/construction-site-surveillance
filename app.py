import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# Streamlit UI setup
st.title("🥇 Safety Compliance Detection")
st.sidebar.header("Settings")
detection_mode = st.sidebar.radio("Select Detection Mode:", ("Webcam Capture", "Upload Image"))

# Download YOLO model from Hugging Face Hub
repo_id = "harikrishnaaa321/construction-site-surveillance-model"
model_filename = "best.pt"
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

# Load YOLO model
model = YOLO(model_path)
class_names = [
    "Helmet", "Mask", "NO-Helmet", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"
]

# Violation tracking
confidence_threshold = 0.45

def process_frame(frame):
    """Detect safety violations in the given frame."""
    results = model(frame)
    detected_classes = set()

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf >= confidence_threshold:
                cls = int(box.cls[0].item())
                detected_classes.add(class_names[cls])

    return results, detected_classes

def draw_detections(frame, results, detected_classes):
    """Draw bounding boxes on detected objects."""
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label_name = class_names[cls]

            # Resolve conflicting detections
            if label_name == "NO-Mask" and "Mask" in detected_classes:
                continue
            if label_name == "Mask" and "NO-Mask" in detected_classes:
                continue
            if label_name == "NO-Helmet" and "Helmet" in detected_classes:
                continue
            if label_name == "Helmet" and "NO-Helmet" in detected_classes:
                continue
            if label_name == "NO-Safety Vest" and "Safety Vest" in detected_classes:
                continue
            if label_name == "Safety Vest" and "NO-Safety Vest" in detected_classes:
                continue

            # Set bounding box color
            color = (0, 255, 0) if "NO-" not in label_name else (0, 0, 255)
            label = f"{label_name}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

if detection_mode == "Webcam Capture":
    start_webcam = st.button("Start Webcam")
    if start_webcam:
        st.write("📷 **Use your webcam to capture an image**")
        cam_image = st.camera_input("Take a picture")

        if cam_image:
            image = Image.open(cam_image)
            frame = np.array(image)
            results, detected_classes = process_frame(frame)
            frame = draw_detections(frame, results, detected_classes)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB")

elif detection_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        results, detected_classes = process_frame(frame)
        frame = draw_detections(frame, results, detected_classes)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB")
