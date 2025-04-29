from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import pymongo
import threading
import time
import importlib.util

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
face_collection = db["face_vectors"]
plate_collection = db["plates and face"]
log_collection = db["logs"]

# Load checkin2.py and Checkout.py dynamically
def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

checkin_module = load_module("./checkin2.py", "checkin2")
checkout_module = load_module("./Checkout.py", "Checkout")

# Camera setup
cap_face = cv2.VideoCapture(0)
cap_plate = cv2.VideoCapture(1)

if not cap_face.isOpened() or not cap_plate.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# State variables
current_mode = None
stop_event = threading.Event()

# Recent data storage
recent_data = {"recent_face": None, "recent_plate": None}

def cleanup():
    stop_event.set()
    if cap_face.isOpened():
        cap_face.release()
    if cap_plate.isOpened():
        cap_plate.release()
    cv2.destroyAllWindows()

def process_frames():
    global recent_data
    while not stop_event.is_set():
        ret_face, frame_face = cap_face.read()
        ret_plate, frame_plate = cap_plate.read()

        if not ret_face or not ret_plate:
            print("Failed to read frames from one or both cameras.")
            break

        if current_mode == "checkin":
            frame_face, frame_plate, user_id, plate_text = checkin_module.process_checkin_frame(frame_face, frame_plate)
            if user_id:
                recent_data["recent_face"] = {"user_id": user_id, "last_access": time.time()}
            if plate_text:
                recent_data["recent_plate"] = {"plate_text": plate_text, "updated_at": time.time()}
        elif current_mode == "checkout":
            frame_face, frame_plate, user_id, plate_text = checkout_module.process_checkout_frame(frame_face, frame_plate)
            if user_id:
                recent_data["recent_face"] = {"user_id": user_id, "last_access": time.time()}
            if plate_text:
                recent_data["recent_plate"] = {"plate_text": plate_text, "updated_at": time.time()}

        time.sleep(0.03)

@app.route('/start-checkin', methods=['POST'])
def start_checkin():
    global current_mode
    stop_event.clear()
    current_mode = "checkin"
    threading.Thread(target=process_frames, daemon=True).start()
    return jsonify({"success": True})

@app.route('/start-checkout', methods=['POST'])
def start_checkout():
    global current_mode
    stop_event.clear()
    current_mode = "checkout"
    threading.Thread(target=process_frames, daemon=True).start()
    return jsonify({"success": True})

@app.route('/stop', methods=['POST'])
def stop():
    global current_mode
    stop_event.set()
    current_mode = None
    return jsonify({"success": True})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"success": True})

@app.route('/recent-data', methods=['GET'])
def recent_data_endpoint():
    return jsonify({"success": True, "data": recent_data})

@app.route('/logs', methods=['GET'])
def get_logs():
    logs = list(log_collection.find())
    for log in logs:
        log["_id"] = str(log["_id"])
    return jsonify({"success": True, "logs": logs})

def gen_face_feed():
    while not stop_event.is_set():
        ret, frame = cap_face.read()
        if not ret:
            continue
        if current_mode:
            frame, _, _, _ = (checkin_module.process_checkin_frame if current_mode == "checkin" else checkout_module.process_checkout_frame)(frame, cap_plate.read()[1])
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_plate_feed():
    while not stop_event.is_set():
        ret, frame = cap_plate.read()
        if not ret:
            continue
        if current_mode:
            _, frame, _, _ = (checkin_module.process_checkin_frame if current_mode == "checkin" else checkout_module.process_checkout_frame)(cap_face.read()[1], frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/face')
def video_feed_face():
    return Response(gen_face_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/plate')
def video_feed_plate():
    return Response(gen_plate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        cleanup()