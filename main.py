import os
import time
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file (if any)
load_dotenv()

# Sender email and password (stored in environment variables)
sender_email = os.getenv("SENDER_EMAIL")  # Sender's fixed email (stored in .env)
app_password = os.getenv("APP_PASSWORD")  # App password (stored in .env)

# Streamlit UI setup
st.title("👷‍♂️ Safety Compliance Detection")
st.sidebar.header("Settings")
recipient_email = st.sidebar.text_input("Enter recipient email address:")

# Load YOLOv8 model
model = YOLO("best.pt")  # Update with your model path

# Class names
class_names = [
    "Helmet", "Mask", "NO-Helmet", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"
]

# Initialize event counting
event_count = 0
start_time = time.time()
last_sent_time = 0

# Dictionary to track violation counts
violation_counts = {
    "NO-Helmet": 0,
    "NO-Mask": 0,
    "NO-Safety Vest": 0
}

# List to track violation times
violation_times = {
    "NO-Helmet": [],
    "NO-Mask": [],
    "NO-Safety Vest": []
}

# OpenCV Webcam Stream
cap = cv2.VideoCapture(0)

# Streamlit live display
frame_placeholder = st.empty()

# Fixed confidence threshold
confidence_threshold = 0.45


def send_email_alert(violation_counts, violation_times, recipient_email):
    """Send an email notification with violation counts and times."""

    # Prepare violation details for email
    violation_details = ""
    for violation, count in violation_counts.items():
        if count > 0:
            # Format the violation time as a range and number of violations
            time_range = f"Between {violation_times[violation][0]} and {violation_times[violation][-1]}"
            violation_details += f"{time_range}: {count} {violation} violations detected\n"

    # Email content
    subject = "Safety Alert: Compliance Violations"
    body = f"🚨 Safety Alert: The following violations were detected:\n\n{violation_details}"

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start TLS encryption
        server.login(sender_email, app_password)  # Log in with your email and app password
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()  # Close the connection

        st.success(f"✅ Email sent to {recipient_email}!")

    except Exception as e:
        st.error(f"❌ Error sending email: {str(e)}")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)
    detected_classes = set()
    frame_violation_counts = {
        "NO-Helmet": 0,
        "NO-Mask": 0,
        "NO-Safety Vest": 0
    }

    # First loop: Collect detected classes and log violations
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf >= confidence_threshold:  # Apply fixed threshold for detection
                cls = int(box.cls[0].item())
                detected_classes.add(class_names[cls])

                # Log violation if it's a "NO-" class (violation detected)
                if "NO-" in class_names[cls]:
                    violation_type = class_names[cls]
                    frame_violation_counts[violation_type] += 1
                    violation_times[violation_type].append(str(datetime.now().strftime("%H:%M:%S")))

    # Second loop: Draw bounding boxes for detected violations
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label_name = class_names[cls]

            # Pairing logic to avoid duplicate alerts
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

            # Set color based on violation
            color = (0, 255, 0) if "NO-" not in label_name else (0, 0, 255)
            label = f"{label_name}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update global violation counts
    for violation, count in frame_violation_counts.items():
        violation_counts[violation] += count

    # Count violations per minute
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:  # Check every 60 seconds
        if any(count > 0 for count in violation_counts.values()) and (
                time.time() - last_sent_time) >= 600:  # Send alert if any violation occurred
            send_email_alert(violation_counts, violation_times, recipient_email)
            last_sent_time = time.time()
        event_count = 0
        start_time = time.time()

    # Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    # Stop when user exits
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
