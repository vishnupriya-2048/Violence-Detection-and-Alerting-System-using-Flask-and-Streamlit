import os
from dotenv import load_dotenv # Import load_dotenv for .env file support
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
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Load environment variables from .env file (for local development)
# This line should be at the very top of your script.
load_dotenv()

# --- SQLAlchemy Database Setup for Streamlit (to fetch users) ---
# Path to your Flask app's database.
# In production, this would typically be a connection string to a Cloud SQL instance or similar.
DATABASE_URL = 'sqlite:///instance/site.db'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# User Model for database (matches Flask's User model structure for relevant fields)
class User(Base):
    __tablename__ = 'user' # Table name created by Flask-SQLAlchemy
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone_number = Column(String(20), nullable=False)
    # password_hash is not needed here as we only fetch contact info for alerts

    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}', phone_number='{self.phone_number}')>"

# Load YOLO models
# Ensure these paths are correct relative to where you run streamlit_app.py
violenceDetect_model = YOLO("VoilenceDetection/best.pt") 
person_model = YOLO("yolov8n.pt")

# Class names for YOLO model output
classNames = ['NonViolence', 'Violence']

# Confidence thresholds for detection
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.85

# Email Configuration - Get from environment variables
# These must be set in your .env file locally, or in your deployment environment.
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# WhatsApp API Configuration - Get from environment variables
# These must be set in your .env file locally, or in your deployment environment.
ULTRAMSG_INSTANCE_ID = os.environ.get("ULTRAMSG_INSTANCE_ID")
ULTRAMSG_API_TOKEN = os.environ.get("ULTRAMSG_API_TOKEN")

# Construct API URL using the environment variable for instance ID
API_URL = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"

# Basic validation for essential environment variables for alerting
if not EMAIL_SENDER or not EMAIL_PASSWORD:
    st.error("Email sender or password environment variables not set. Email alerts will not work.")
if not ULTRAMSG_INSTANCE_ID or not ULTRAMSG_API_TOKEN:
    st.error("UltraMsg instance ID or API token environment variables not set. WhatsApp alerts will not work.")


# Function to send email alerts
def send_email_alert(subject, body, recipient_email, image_path):
    # Skip if email credentials are not configured
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.warning(f"Skipping email alert to {recipient_email}: Email credentials not configured.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient_email
    msg.set_content(body)

    # Attach the image of the detected violence
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_type = imghdr.what(img_file.name)
        msg.add_attachment(img_data, maintype="image", subtype=img_type, filename="alert.jpg")

    try:
        # Connect to Gmail's SMTP server securely
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD) # Log in to the email server
            server.send_message(msg) # Send the email
        st.success(f"✅ Email alert sent to {recipient_email}")
    except Exception as e:
        st.error(f"🚨 Error sending email to {recipient_email}: {e}")

# Function to send WhatsApp alerts
def send_whatsapp_alert(phone, message):
    # Skip if WhatsApp API credentials are not configured
    if not ULTRAMSG_INSTANCE_ID or not ULTRAMSG_API_TOKEN:
        st.warning(f"Skipping WhatsApp alert to {phone}: WhatsApp API credentials not configured.")
        return

    # Ensure phone number is in the correct format (e.g., +91XXXXXXXXXX)
    phone = str(phone).strip()
    # Assuming Indian numbers; adjust if your target numbers are different (e.g., remove if already prefixed)
    if not phone.startswith("+"):
        phone = "+91" + phone 

    # Payload for the UltraMSG API request
    payload = {
        "token": ULTRAMSG_API_TOKEN, # Use the API token from environment
        "to": phone, # UltraMSG expects just the phone number in 'to' field
        "body": message
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded" # Required for x-www-form-urlencoded
    }

    try:
        # Send POST request to UltraMSG API
        response = requests.post(API_URL, data=payload, headers=headers)
        if response.status_code == 200:
            st.success(f"✅ WhatsApp alert sent to {phone}")
        else:
            st.error(f"❌ WhatsApp alert failed for {phone}: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚨 Exception while sending WhatsApp alert to {phone}: {e}")

# Function to trigger alerts to all registered users
def send_alerts(frame):
    frame_path = "violence_alert.jpg"
    cv2.imwrite(frame_path, frame) # Save the frame with detected violence

    session = Session() # Create a new SQLAlchemy session
    try:
        all_users = session.query(User).all() # Fetch all registered users from the database
        if not all_users:
            st.warning("No personnel registered in the database to send alerts to.")
            return

        # Send alerts to each registered user
        for user in all_users:
            send_email_alert("Violence Detected", "Alert! Violence detected in the premises.", user.email, frame_path)
            send_whatsapp_alert(user.phone_number, "Alert! Violence detected in the premises. Check your email for details.")
    finally:
        session.close() # Close the session to release resources

# Streamlit App User Interface
st.title("Violence Detection System")
st.write("Welcome! This system detects violence in video streams and sends alerts to registered personnel.")
st.write("Ensure you are logged in via the authentication portal to receive alerts.")

# Radio buttons to choose input source
option = st.radio("Choose Input Source:", ("Upload Video", "Live Webcam"))

cap = None # VideoCapture object

# Handle video upload option
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

# Handle live webcam option
elif option == "Live Webcam":
    st.info("Attempting to open webcam...")
    # Try multiple camera indices for better compatibility
    for i in range(3): 
        temp_cap = cv2.VideoCapture(i, cv2.CAP_MSMF) # Use CAP_MSMF backend for better Windows compatibility
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Set frame width
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set frame height
        time.sleep(1) # Short warm-up time for the camera
        if temp_cap.isOpened():
            cap = temp_cap
            st.success(f"✅ Camera index {i} opened successfully.")
            break
        else:
            temp_cap.release() # Release camera if it couldn't be opened
    if not cap or not cap.isOpened():
        st.error("🚫 Could not open camera. It may be in use by another application or permissions are denied.")
        st.info("Please ensure no other application is using the webcam and check your browser/system camera permissions.")

# If a video capture object is available
if cap:
    stframe = st.empty() # Placeholder for the video frame
    fail_count = 0 # Counter for webcam read failures
    st.write("Processing video stream...")

    while cap.isOpened():
        success, frame = cap.read() # Read a frame from the video stream

        # Handle read failures (e.g., webcam disconnected)
        if not success or frame is None:
            fail_count += 1
            if fail_count > 30: # After 30 consecutive failures (approx 1 second at 30fps)
                st.error("📷 Webcam feed failed repeatedly. Please check camera permissions or restart.")
                break
            st.warning("⚠️ Could not read from webcam. Displaying black frame temporarily.")
            black_frame = np.zeros((720, 1280, 3), dtype=np.uint8) # Create a black frame
            stframe.image(black_frame, channels="RGB") # Display the black frame
            time.sleep(0.1) # Small delay to prevent tight loop on failure
            continue
        else:
            fail_count = 0 # Reset failure count on successful read

        # Perform person detection
        person_results = person_model(frame, verbose=False) # Run YOLO person detection
        persons = []
        for result in person_results:
            for box in result.boxes:
                conf = float(box.conf[0]) # Get confidence score
                # Check for 'person' class (class 0 in COCO dataset) and confidence threshold
                if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get bounding box coordinates
                    persons.append((x1, y1, x2, y2))

        violence_detected_in_frame = False
        # Perform violence detection only if persons are detected
        if persons:
            violence_results = violenceDetect_model(frame, verbose=False) # Run YOLO violence detection
            for result in violence_results:
                for box in result.boxes:
                    conf = float(box.conf[0]) # Get confidence score
                    cls = int(box.cls[0]) # Get class ID
                    currentClass = classNames[cls] # Map class ID to name

                    # Check for 'Violence' class and confidence threshold
                    if conf > VIOLENCE_CONFIDENCE_THRESHOLD and currentClass == 'Violence':
                        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get bounding box for violence
                        # Check if violence detection box overlaps with any person box
                        for px1, py1, px2, py2 in persons:
                            # Simple overlap check (bounding box intersection)
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw red rectangle
                                # Put text label with confidence
                                cvzone.putTextRect(frame, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0,0,255))
                                violence_detected_in_frame = True
                                break # Stop checking other persons for this violence box once overlap is found

                # If violence was detected and processed in this frame, send alerts
                if violence_detected_in_frame:
                    send_alerts(frame) # Trigger alert function
                    # --- Important for Production: Implement Alert Throttling Here ---
                    # To prevent spamming alerts, add a cooldown (e.g., don't send another alert for X seconds)
                    # Example:
                    # last_alert_time = st.session_state.get('last_alert_time', 0)
                    # if time.time() - last_alert_time > 30: # 30-second cooldown
                    #     send_alerts(frame)
                    #     st.session_state.last_alert_time = time.time()
                    
        # Convert BGR (OpenCV default) to RGB (Streamlit default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB") # Display the processed frame in Streamlit

    # Release resources when the loop finishes
    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Please select an input source to start the detection.")