# app.py

import streamlit as st
import cv2
import numpy as np
import time
import threading
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils import play_alert_sound, create_alert_message
from camera_simulator import CameraSimulator

# Page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
    color: #1f77b4;
}
.status-awake {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
}
.status-drowsy {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
}
.status-eyes-closed {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load Haarcascade for face and eyes
detector_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(detector_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(detector_path + 'haarcascade_eye.xml')

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'drowsy_detections' not in st.session_state:
    st.session_state.drowsy_detections = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Sidebar configuration
st.sidebar.header("⚙️ Detection Settings")
ear_threshold = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
consecutive_frames = st.sidebar.slider("Consecutive Frames", 5, 30, 15, 1)

col1, col2 = st.sidebar.columns(2)
start_button = col1.button("Start")
stop_button = col2.button("Stop")

if st.sidebar.button("Reset Statistics"):
    st.session_state.detection_history.clear()
    st.session_state.start_time = datetime.now()
    st.session_state.total_detections = 0
    st.session_state.drowsy_detections = 0
    st.rerun()

# UI layout
st.markdown('<h1 class="main-header">🚗 Driver Drowsiness Detection</h1>', unsafe_allow_html=True)
video_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
history_placeholder = st.empty()

# EAR estimation fallback (simplified)
def estimate_ear_from_eyes(eyes):
    if len(eyes) >= 2:
        distances = [abs(e1[1] - e2[1]) for i, e1 in enumerate(eyes) for j, e2 in enumerate(eyes) if i < j]
        return min(distances) / max(1, max(distances))
    return 0.0

# Main detection logic
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    status = "Awake"
    ear = 0.3
    confidence = 0.9

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        ear = estimate_ear_from_eyes(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        if ear < ear_threshold:
            status = "Drowsy"
            confidence = 0.95

    return frame, status, ear, 0.0, confidence

# Metrics and history update
def update_statistics(status, ear, confidence):
    now = datetime.now()
    st.session_state.detection_history.append({
        'timestamp': now,
        'status': status,
        'ear': ear,
        'mar': 0,
        'confidence': confidence
    })
    st.session_state.total_detections += 1
    if status != "Awake":
        st.session_state.drowsy_detections += 1

# UI helpers
def create_status_display(status, ear, confidence):
    icon = "😊" if status == "Awake" else ("😴" if status == "Drowsy" else "😑")
    class_name = "status-awake" if status == "Awake" else ("status-drowsy" if status == "Drowsy" else "status-eyes-closed")
    drowsy_pct = (st.session_state.drowsy_detections / max(1, st.session_state.total_detections)) * 100
    runtime = datetime.now() - st.session_state.start_time

    status_html = f"""
    <div class="{class_name}">
        <h2>{icon} {status}</h2>
        <p>Confidence: {confidence:.1%}</p>
    </div>
    """

    metrics_html = f"""
    <div class="metric-container">
        <p><strong>EAR:</strong> {ear:.3f} (Threshold: {ear_threshold})</p>
        <p><strong>Runtime:</strong> {str(runtime).split('.')[0]}</p>
        <p><strong>Total Detections:</strong> {st.session_state.total_detections}</p>
        <p><strong>Drowsy Events:</strong> {st.session_state.drowsy_detections}</p>
        <p><strong>Drowsiness Rate:</strong> {drowsy_pct:.1f}%</p>
    </div>
    """
    return status_html, metrics_html

def create_detection_chart():
    if not st.session_state.detection_history:
        return None
    df = pd.DataFrame(st.session_state.detection_history[-50:])
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ear'], mode='lines+markers', name='EAR'), row=1, col=1)
    fig.update_layout(title="EAR Over Time", height=300)
    return fig

def create_history_table():
    if not st.session_state.detection_history:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.detection_history[-10:])
    df['timestamp'] = df['timestamp'].dt.strftime('%H:%M:%S')
    return df[['timestamp', 'status', 'ear', 'confidence']].rename(columns={
        'timestamp': 'Time',
        'status': 'Status',
        'ear': 'EAR',
        'confidence': 'Confidence'
    })

# Start/stop handling
if start_button and not st.session_state.is_running:
    try:
        st.session_state.camera = None
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                st.session_state.camera = cap
                break
        if not st.session_state.camera:
            st.session_state.camera = CameraSimulator()
        st.session_state.is_running = True
    except:
        st.error("Camera initialization failed")

if stop_button and st.session_state.is_running:
    st.session_state.is_running = False
    if hasattr(st.session_state.camera, 'release'):
        st.session_state.camera.release()
    st.session_state.camera = None
    st.success("Detection stopped")

# Main loop
if st.session_state.is_running:
    ret, frame = st.session_state.camera.read()
    if ret:
        processed, status, ear, mar, conf = process_frame(frame)
        update_statistics(status, ear, conf)
        video_placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
        s_html, m_html = create_status_display(status, ear, conf)
        status_placeholder.markdown(s_html, unsafe_allow_html=True)
        metrics_placeholder.markdown(m_html, unsafe_allow_html=True)
        chart = create_detection_chart()
        if chart:
            chart_placeholder.plotly_chart(chart, use_container_width=True)
        hist = create_history_table()
        if not hist.empty:
            history_placeholder.dataframe(hist, use_container_width=True)
        time.sleep(0.1)
        st.rerun()
    else:
        st.error("Failed to capture frame")
        st.session_state.is_running = False

else:
    video_placeholder.info("Click Start to begin detection")
    status_placeholder.info("Detection not running")
    if st.session_state.detection_history:
        chart = create_detection_chart()
        if chart:
            chart_placeholder.plotly_chart(chart, use_container_width=True)
        hist = create_history_table()
        if not hist.empty:
            history_placeholder.dataframe(hist, use_container_width=True)
    else:
        chart_placeholder.info("No detection data yet")
        history_placeholder.info("No history available")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>🚗 Built without dlib - Stay Safe!</div>", unsafe_allow_html=True)
