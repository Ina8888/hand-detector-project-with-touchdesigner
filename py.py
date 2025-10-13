import cv2
import mediapipe as mp
from flask import Flask, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app) 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Global variable to store hand data
current_hand_data = {"wrist": [0.5, 0.5, 0], "gesture": "none"}

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.running = True
        
    def start_tracking(self):
        global current_hand_data
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get key points
                    lm_wrist = hand_landmarks.landmark[0]
                    lm_middle = hand_landmarks.landmark[9]
                    lm_index = hand_landmarks.landmark[8]
                    
                    current_hand_data = {
                        'wrist': [lm_wrist.x, lm_wrist.y, lm_wrist.z],
                        'middle': [lm_middle.x, lm_middle.y, lm_middle.z],
                        'index': [lm_index.x, lm_index.y, lm_index.z],
                        'gesture': 'open' if lm_index.y < lm_middle.y else 'closed'
                    }

                    # Optional: draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())
            else:
                current_hand_data = {"wrist": [0.5, 0.5, 0], "gesture": "none"}
            
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
                
    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

# Flask route to get hand data
@app.route('/hand-data')
def hand_data():
    return jsonify(current_hand_data)

def run_flask():
    app.run(host='localhost', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Start hand tracking in a separate thread
    tracker = HandTracker()
    
    # Start Flask server in a thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    print("Hand tracker started!")
    print("Open your browser to: http://localhost:5000/")
    print("Press ESC in the camera window to stop")
    
    try:
        tracker.start_tracking()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        tracker.stop()