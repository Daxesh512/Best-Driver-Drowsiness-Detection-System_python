import numpy as np
import time
import threading
from datetime import datetime

# Global variables for alert management
last_alert_time = 0
ALERT_COOLDOWN = 2.0  # Seconds between alerts

def generate_alert_tone(frequency=800, duration=0.5, sample_rate=22050):
    """Generate a simple alert tone"""
    frames = int(duration * sample_rate)
    arr = np.zeros(frames)
    
    for i in range(frames):
        # Generate sine wave
        arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
        # Add some amplitude envelope to avoid clicks
        envelope = min(i / (sample_rate * 0.1), 1.0, (frames - i) / (sample_rate * 0.1))
        arr[i] *= envelope
    
    # Convert to 16-bit integers
    arr = (arr * 32767).astype(np.int16)
    
    # Make stereo
    stereo_arr = np.zeros((frames, 2), dtype=np.int16)
    stereo_arr[:, 0] = arr
    stereo_arr[:, 1] = arr
    
    return stereo_arr

def play_alert_sound():
    """Visual alert instead of audio"""
    global last_alert_time
    
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return
    
    last_alert_time = current_time
    print("DROWSINESS ALERT - Driver attention required!")

def play_alert_thread():
    """Play alert sound in a separate thread to avoid blocking"""
    thread = threading.Thread(target=play_alert_sound)
    thread.daemon = True
    thread.start()

def create_alert_message(status, confidence):
    """Create an alert message based on detection status"""
    messages = {
        "Awake": {
            "message": "Driver is alert and awake",
            "color": "success",
            "icon": "😊"
        },
        "Drowsy": {
            "message": "⚠️ DROWSINESS DETECTED! Please take a break",
            "color": "warning", 
            "icon": "😴"
        },
        "Eyes Closed": {
            "message": "🚨 EYES CLOSED! Pull over safely",
            "color": "error",
            "icon": "😑"
        }
    }
    
    alert_info = messages.get(status, messages["Awake"])
    alert_info["confidence"] = confidence
    alert_info["timestamp"] = datetime.now()
    
    return alert_info

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def calculate_detection_stats(detection_history):
    """Calculate various statistics from detection history"""
    if not detection_history:
        return {
            "total_time": 0,
            "awake_time": 0,
            "drowsy_time": 0,
            "eyes_closed_time": 0,
            "awake_percentage": 0,
            "drowsy_percentage": 0,
            "eyes_closed_percentage": 0,
            "total_detections": 0,
            "average_ear": 0,
            "average_mar": 0
        }
    
    total_detections = len(detection_history)
    awake_count = sum(1 for entry in detection_history if entry['status'] == 'Awake')
    drowsy_count = sum(1 for entry in detection_history if entry['status'] == 'Drowsy')
    eyes_closed_count = sum(1 for entry in detection_history if entry['status'] == 'Eyes Closed')
    
    # Calculate percentages
    awake_percentage = (awake_count / total_detections) * 100
    drowsy_percentage = (drowsy_count / total_detections) * 100
    eyes_closed_percentage = (eyes_closed_count / total_detections) * 100
    
    # Calculate averages
    average_ear = np.mean([entry['ear'] for entry in detection_history])
    average_mar = np.mean([entry['mar'] for entry in detection_history])
    
    # Calculate time estimates (assuming ~10 FPS)
    frame_duration = 0.1  # seconds per frame
    total_time = total_detections * frame_duration
    awake_time = awake_count * frame_duration
    drowsy_time = drowsy_count * frame_duration
    eyes_closed_time = eyes_closed_count * frame_duration
    
    return {
        "total_time": total_time,
        "awake_time": awake_time,
        "drowsy_time": drowsy_time,
        "eyes_closed_time": eyes_closed_time,
        "awake_percentage": awake_percentage,
        "drowsy_percentage": drowsy_percentage,
        "eyes_closed_percentage": eyes_closed_percentage,
        "total_detections": total_detections,
        "average_ear": average_ear,
        "average_mar": average_mar
    }

def create_confidence_indicator(confidence):
    """Create a visual confidence indicator"""
    if confidence < 0.3:
        return "🟢", "Low confidence"
    elif confidence < 0.7:
        return "🟡", "Medium confidence"
    else:
        return "🔴", "High confidence"

def validate_detection_thresholds(ear_threshold, mar_threshold):
    """Validate that detection thresholds are within reasonable ranges"""
    warnings = []
    
    if ear_threshold < 0.15:
        warnings.append("EAR threshold is very low - may cause false positives")
    elif ear_threshold > 0.35:
        warnings.append("EAR threshold is very high - may miss actual drowsiness")
    
    if mar_threshold < 0.5:
        warnings.append("MAR threshold is very low - may cause false yawn detection")
    elif mar_threshold > 1.0:
        warnings.append("MAR threshold is very high - may miss actual yawns")
    
    return warnings

def export_detection_data(detection_history, filename=None):
    """Export detection history to CSV format"""
    if not detection_history:
        return None
    
    if filename is None:
        filename = f"drowsiness_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    import pandas as pd
    
    df = pd.DataFrame(detection_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    try:
        df.to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Failed to export data: {e}")
        return None

def get_system_recommendations(stats):
    """Provide recommendations based on detection statistics"""
    recommendations = []
    
    drowsy_percentage = stats['drowsy_percentage']
    eyes_closed_percentage = stats['eyes_closed_percentage']
    average_ear = stats['average_ear']
    
    if drowsy_percentage > 20:
        recommendations.append("⚠️ High drowsiness detected. Consider taking frequent breaks.")
    
    if eyes_closed_percentage > 10:
        recommendations.append("🚨 Frequent eye closure detected. Please pull over and rest.")
    
    if average_ear < 0.2:
        recommendations.append("😴 Low average EAR suggests fatigue. Consider adjusting detection sensitivity.")
    
    if stats['total_time'] > 7200:  # 2 hours
        recommendations.append("🕐 Long driving session detected. Take a break every 2 hours.")
    
    if not recommendations:
        recommendations.append("✅ Good driving alertness detected. Keep up the safe driving!")
    
    return recommendations
