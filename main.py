import os
import time
import cv2
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Sender email credentials
sender_email = os.getenv("SENDER_EMAIL")
app_password = os.getenv("APP_PASSWORD")

# Streamlit UI setup
st.title("🥷 Safety Compliance Detection")
st.sidebar.header("Settings")
recipient_email = st.sidebar.text_input("Enter recipient email address:")

# Load YOLO model with error handling
try:
    model = YOLO("best.pt")  # Update with your model path
except Exception as e:
    st.error(f"Error loading YOLO model: {str(e)}")
    st.stop()

# Class names
class_names = [
    "Helmet", "Mask", "NO-Helmet", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"
]

# Initialize event tracking
violation_counts = {"NO-Helmet": 0, "NO-Mask": 0, "NO-Safety Vest": 0}
violation_times = {"NO-Helmet": [], "NO-Mask": [], "NO-Safety Vest": []}

# Attempt to open the camera safely
cap = None
for i in range(5):  # Try multiple indexes in case of failure
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        break
    cap.release()

if not cap or not cap.isOpened():
    st.error("🚨 Camera not found. Please check your webcam and try again.")
    st.stop()

frame_placeholder = st.empty()
confidence_threshold = 0.45
start_time = time.time()
last_sent_time = 0

# Function to send email
async def send_email_alert(violation_counts, violation_times, recipient_email):
    violation_details = "\n".join(
        f"Between {times[0]} and {times[-1]}: {count} {violation} violations detected"
        for violation, (count, times) in zip(violation_counts.keys(), violation_times.items()) if count > 0
    )

    subject = "Safety Alert: Compliance Violations"
    body = f"\ud83d\udea8 Safety Alert:\n\n{violation_details}"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        st.success(f"✅ Email sent to {recipient_email}!")
    except Exception as e:
        st.error(f"❌ Error sending email: {str(e)}")

# Capture and process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video frame.")
        break

    results = model(frame)
    detected_classes = set()
    frame_violation_counts = {key: 0 for key in violation_counts}

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

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label_name = class_names[cls]
            
            if label_name.startswith("NO-") and label_name[3:] in detected_classes:
                continue

            color = (0, 255, 0) if "NO-" not in label_name else (0, 0, 255)
            label = f"{label_name}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for violation, count in frame_violation_counts.items():
        violation_counts[violation] += count

    elapsed_time = time.time() - start_time
    if elapsed_time >= 60 and any(violation_counts.values()) and (time.time() - last_sent_time) >= 600:
        asyncio.run(send_email_alert(violation_counts, violation_times, recipient_email))
        last_sent_time = time.time()
        violation_counts = {key: 0 for key in violation_counts}
        start_time = time.time()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
