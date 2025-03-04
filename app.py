import os
import time
import cv2
import smtplib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()
SENDER_EMAIL = "dummyyydummyyy123@gmail.com"
APP_PASSWORD = "wnwd bvzg fsal qpmx"

# Streamlit UI setup
st.title("🥇 Safety Compliance Detection")
st.sidebar.header("Settings")
recipient_email = st.sidebar.text_input("Enter recipient email address:")
detection_mode = st.sidebar.radio("Select Detection Mode:", ("Webcam", "Upload Image"))

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
violation_counts = {"NO-Helmet": 0, "NO-Mask": 0, "NO-Safety Vest": 0}
violation_times = {"NO-Helmet": [], "NO-Mask": [], "NO-Safety Vest": []}
confidence_threshold = 0.45
last_email_time = 0  # Track last email timestamp
violation_start_time = None  # Track when violations start
email_interval = 60  # Minimum interval (seconds) for continuous violations

def send_email_alert():
    """Send an email only after 1 minute of continuous violations."""
    global last_email_time, violation_start_time

    # If there are no violations, reset the timer
    if all(count == 0 for count in violation_counts.values()):
        violation_start_time = None
        return

    # If a violation is detected and timer hasn't started, start it
    if violation_start_time is None:
        violation_start_time = time.time()
        return

    # Check if violations have continued for at least 1 minute
    if time.time() - violation_start_time < email_interval:
        return

    # Prepare email content
    violation_details = "\n".join(
        f"{violation}: {count} violations detected"
        for violation, count in violation_counts.items() if count > 0
    )

    subject = "🚨 Safety Alert: Continuous Violations Detected"
    body = f"⚠️ The following violations have been continuously detected for over 1 minute:\n\n{violation_details}"
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        st.success(f"✅ Email sent to {recipient_email}!")

        # Reset email timer and violation start time
        last_email_time = time.time()
        violation_start_time = None  # Reset to detect future violations
    except Exception as e:
        st.error(f"❌ Error sending email: {str(e)}")

def process_frame(frame):
    """Detect safety violations in the given frame."""
    results = model(frame)
    detected_classes = set()
    frame_violation_counts = {"NO-Helmet": 0, "NO-Mask": 0, "NO-Safety Vest": 0}

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf >= confidence_threshold:
                cls = int(box.cls[0].item())
                detected_classes.add(class_names[cls])

                if "NO-" in class_names[cls]:
                    violation_type = class_names[cls]
                    frame_violation_counts[violation_type] += 1
                    violation_times[violation_type].append(datetime.now().strftime("%H:%M:%S"))

    for violation, count in frame_violation_counts.items():
        violation_counts[violation] += count

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

            # Set bounding box color (Red for violations, Green otherwise)
            color = (0, 255, 0) if "NO-" not in label_name else (0, 0, 255)
            label = f"{label_name}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

if detection_mode == "Webcam":
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results, detected_classes = process_frame(frame)
        frame = draw_detections(frame, results, detected_classes)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        # Send email alert only if enough time has passed with continuous violations
        send_email_alert()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

elif detection_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        results, detected_classes = process_frame(frame)
        frame = draw_detections(frame, results, detected_classes)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB")
