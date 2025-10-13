import cv2
import mediapipe as mp
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
import os
import time

app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

current_hand_data = {"wrist": [0.5, 0.5, 0], "gesture": "none", "fingers": [0,0,0,0,0]}

class HandTracker:
    def __init__(self):
        self.cap = None
        self.hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.running = True
        
    def start_tracking(self):
        global current_hand_data
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return
            
        print("📷 Camera opened successfully")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    
                    fingers = [
                        1 if landmarks[8].y < landmarks[6].y else 0,
                        1 if landmarks[12].y < landmarks[10].y else 0, 
                        1 if landmarks[16].y < landmarks[14].y else 0,
                        1 if landmarks[20].y < landmarks[18].y else 0,
                        1 if landmarks[4].x < landmarks[3].x else 0
                    ]
                    
                    current_hand_data = {
                        'wrist': [landmarks[0].x, landmarks[0].y, landmarks[0].z],
                        'gesture': 'open' if sum(fingers) >= 3 else 'closed',
                        'fingers': fingers
                    }
            else:
                # Reset when no hand detected
                current_hand_data = {"wrist": [0.5, 0.5, 0], "gesture": "none", "fingers": [0,0,0,0,0]}

            cv2.imshow("Hand Tracking - Press ESC to quit", frame)
            
            # Check for ESC key without blocking
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\n🛑 ESC pressed - shutting down...")
                self.running = False
                break
                
    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("🎥 Camera released")

@app.route('/')
def serve_index():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        return f"Error loading index.html: {e}"

@app.route('/hand-data')
def hand_data():
    return jsonify(current_hand_data)

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

def run_flask():
    print("🌐 Starting Flask server...")
    app.run(host='localhost', port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    # Check if HTML file exists
    if not os.path.exists('index.html'):
        print("❌ ERROR: index.html not found!")
        print("📁 Current directory:", os.getcwd())
        print("📋 Files in directory:", [f for f in os.listdir('.') if f.endswith('.html')])
        exit(1)
    
    print("✅ index.html found")
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask a moment to start
    time.sleep(2)
    
    print("🚀 Hand Controller Started!")
    print("🌐 Open: http://localhost:5000/")
    print("🎮 Move your hand in front of the camera")
    print("⎋ Press ESC to quit")
    
    tracker = HandTracker()
    
    try:
        tracker.start_tracking()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        tracker.stop()
        print("👋 Application closed")