import os
from dotenv import load_dotenv
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import cvzone
import requests
import smtplib
import imghdr
from email.message import EmailMessage
import time

# --- WebRTC specific imports ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # Required for handling video frames in WebRTC context
# -----------------------------

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Set this environment variable for Streamlit on Render to ensure file changes are watched
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Load environment variables from .env file (for local development)
load_dotenv()

# --- SQLAlchemy Database Setup for Streamlit (to fetch users for alerting) ---
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    st.error("🚨 DATABASE_URL environment variable is not set. Database connection for alerts will fail.")
    st.info("Please ensure DATABASE_URL is set in your Streamlit service's environment variables on Render.")
    st.stop() # Stop the app if database connection is critical

try:
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    Base = declarative_base()

    class User(Base):
        __tablename__ = 'user'
        id = Column(Integer, primary_key=True)
        name = Column(String(100), nullable=False)
        email = Column(String(100), unique=True, nullable=False)
        phone_number = Column(String(20), nullable=False)

        def __repr__(self):
            return f"<User(name='{self.name}', email='{self.email}', phone_number='{self.phone_number}')>"

    with Session() as session:
        session.query(User).limit(1).all()
        print("Streamlit app: Successfully connected to PostgreSQL database.")

except Exception as e:
    st.error(f"🚨 Error connecting to PostgreSQL database from Streamlit: {e}")
    st.info("Please check your DATABASE_URL environment variable and database connectivity.")
    st.stop() # Stop the app if database connection failed

# Initialize session state for alert throttling
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

# Load YOLO models (ensure these paths are correct in your Docker container)
# IMPORTANT: These paths are relative to the WORKDIR /app in your Docker container.
# Ensure your .pt files are placed in a 'models/' subdirectory INSIDE your streamlit_app/ folder.
violenceDetect_model = YOLO("models/best.pt")
person_model = YOLO("models/yolov8n.pt")

# Class names for YOLO model output
classNames = ['NonViolence', 'Violence']

# Confidence thresholds for detection
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.85

# Email Configuration - Get from environment variables
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# WhatsApp API Configuration - Get from environment variables
ULTRAMSG_INSTANCE_ID = os.environ.get("ULTRAMSG_INSTANCE_ID")
ULTRAMSG_API_TOKEN = os.environ.get("ULTRAMSG_API_TOKEN")

# Construct API URL using the environment variable for instance ID
API_URL = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"

# Basic validation for essential environment variables for alerting
if not EMAIL_SENDER or not EMAIL_PASSWORD:
    st.warning("Email sender or password environment variables not set. Email alerts will not work.")
if not ULTRAMSG_INSTANCE_ID or not ULTRAMSG_API_TOKEN:
    st.warning("UltraMsg instance ID or API token environment variables not set. WhatsApp alerts will not work.")

# Add a simple throttling mechanism for alerts
ALERT_COOLDOWN_SECONDS = 30 # Only send alerts every 30 seconds

# Function to send email alerts
def send_email_alert(subject, body, recipient_email, image_path):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.warning(f"Skipping email alert to {recipient_email}: Email credentials not configured.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient_email
    msg.set_content(body)

    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_type = imghdr.what(img_file.name)
            msg.add_attachment(img_data, maintype="image", subtype=img_type, filename="alert.jpg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email alert sent to {recipient_email}") # Using print as st.success might not show inside transform
    except Exception as e:
        print(f"🚨 Error sending email to {recipient_email}: {e}")

# Function to send WhatsApp alerts
def send_whatsapp_alert(phone, message):
    if not ULTRAMSG_INSTANCE_ID or not ULTRAMSG_API_TOKEN:
        st.warning(f"Skipping WhatsApp alert to {phone}: WhatsApp API credentials not configured.")
        return

    phone = str(phone).strip()
    if not phone.startswith("+"):
        print(f"Phone number {phone} for WhatsApp alert is not in international format (e.g., +91...); attempting to send anyway.")

    payload = {
        "token": ULTRAMSG_API_TOKEN,
        "to": phone,
        "body": message
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        response = requests.post(API_URL, data=payload, headers=headers)
        if response.status_code == 200:
            print(f"✅ WhatsApp alert sent to {phone}")
        else:
            print(f"❌ WhatsApp alert failed for {phone}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"🚨 Exception while sending WhatsApp alert to {phone}: {e}")

# Function to trigger alerts to all registered users (with throttling)
def send_alerts(frame):
    # Using st.session_state for persistence across Streamlit reruns
    if time.time() - st.session_state.last_alert_time < ALERT_COOLDOWN_SECONDS:
        # print(f"Alerts on cooldown. Next alert in {int(ALERT_COOLDOWN_SECONDS - (time.time() - st.session_state.last_alert_time))} seconds.")
        return

    frame_path = "violence_alert.jpg"
    cv2.imwrite(frame_path, frame)

    if 'Session' in globals() and Session: # Check if Session object was successfully created
        session = Session()
        try:
            all_users = session.query(User).all()
            if not all_users:
                print("No personnel registered in the database to send alerts to.")
                return

            for user in all_users:
                send_email_alert("Violence Detected", "Alert! Violence detected in the premises.", user.email, frame_path)
                send_whatsapp_alert(user.phone_number, "Alert! Violence detected in the premises. Check your email for details.")
            
            st.session_state.last_alert_time = time.time() # Update last alert time only if alerts were sent
        except Exception as e:
            print(f"🚨 Error fetching users or sending alerts from DB: {e}")
        finally:
            session.close()
    else:
        print("Database not connected. Cannot fetch users for alerts.")


# --- WebRTC Video Processing Class ---
# This class will receive each video frame, process it, and return the result.
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert av.VideoFrame to OpenCV numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Perform person detection
        person_results = person_model(img, verbose=False)
        persons = []
        for result in person_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0: # class 0 is 'person' in COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append((x1, y1, x2, y2))

        violence_detected_in_frame = False
        # Perform violence detection only if persons are detected
        if persons:
            violence_results = violenceDetect_model(img, verbose=False)
            for result in violence_results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    if conf > VIOLENCE_CONFIDENCE_THRESHOLD and currentClass == 'Violence':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check if violence detection box overlaps with any person box
                        for px1, py1, px2, py2 in persons:
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(img, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0,0,255))
                                violence_detected_in_frame = True
                                break # Stop checking other persons for this violence box once overlap is found

            # If violence was detected and confirmed with a person, send alerts (with throttling)
            if violence_detected_in_frame:
                send_alerts(img) # Trigger alert function
                                    
        # Convert OpenCV numpy array back to av.VideoFrame for Streamlit display
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Streamlit App User Interface ---
st.title("Violence Detection System")
st.write("Welcome! This system detects violence in video streams and sends alerts to registered personnel.")
st.write("To receive alerts, ensure your environment variables (DATABASE_URL, EMAIL_SENDER, EMAIL_PASSWORD, ULTRAMSG_INSTANCE_ID, ULTRAMSG_API_TOKEN) are correctly set in Render.")

# Radio buttons to choose input source
option = st.radio("Choose Input Source:", ("Live Camera Feed (WebRTC)", "Upload Video"))

# Handle Live Camera Feed (WebRTC) option
if option == "Live Camera Feed (WebRTC)":
    st.subheader("Live Camera Feed (via WebRTC)")
    st.info("Please grant camera permissions to start the live stream. If the stream doesn't start, try refreshing the page or checking browser permissions.")
    
    # This is where the WebRTC magic happens.
    # It sets up the video stream from the client's browser to your Streamlit app.
    webrtc_ctx = webrtc_streamer(
        key="violence-detector",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}, # Only video, no audio
        async_processing=True # Process frames asynchronously to avoid blocking UI
    )

    if webrtc_ctx.video_receiver:
        st.write("WebRTC stream active.")
    else:
        st.write("Waiting for WebRTC stream...")

# Handle video upload option (existing functionality)
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    cap = None
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("🚫 Could not open video file. Please check if it's a valid video format.")
            cap = None
    
    if cap:
        stframe = st.empty()
        st.write("Processing uploaded video stream...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                st.info("End of video stream.")
                break

            # Perform detection (similar to VideoProcessor but for uploaded file)
            person_results = person_model(frame, verbose=False)
            persons = []
            for result in person_results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        persons.append((x1, y1, x2, y2))

            violence_detected_in_frame = False
            if persons:
                violence_results = violenceDetect_model(frame, verbose=False)
                for result in violence_results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]

                        if conf > VIOLENCE_CONFIDENCE_THRESHOLD and currentClass == 'Violence':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            for px1, py1, px2, py2 in persons:
                                if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cvzone.putTextRect(frame, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0,0,255))
                                    violence_detected_in_frame = True
                                    break
                if violence_detected_in_frame:
                    send_alerts(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
    else:
        st.info("Please upload a video file.")

else:
    st.info("Select an input source from the options above.")
