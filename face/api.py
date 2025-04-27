from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import subprocess
import threading
import pymongo
import logging
import os
import time
import signal
import cv2
from datetime import datetime
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB connection
CLIENT = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB = CLIENT["face_db"]
LOG_COLLECTION = DB["logs"]

# Process management
current_process = None

# Global variables for video feeds
face_camera = None
plate_camera = None
face_frame = None
plate_frame = None
stop_threads = False

def run_script(script_path):
    global current_process
    try:
        # Kill any existing process
        if current_process and current_process.poll() is None:
            os.kill(current_process.pid, signal.SIGTERM)
            time.sleep(1)  # Give it time to terminate
            
        # Start new process
        current_process = subprocess.Popen(['python', script_path], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
        logger.info(f"Started script: {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error running script {script_path}: {e}")
        return False

def generate_camera_frames(camera_id):
    global face_camera, plate_camera, face_frame, plate_frame, stop_threads
    
    if camera_id == "face":
        camera = cv2.VideoCapture(0)  # First camera for face
    else:  # camera_id == "plate"
        camera = cv2.VideoCapture(1)  # Second camera for plate
    
    # Store camera objects for later release
    if camera_id == "face":
        face_camera = camera
    else:
        plate_camera = camera
    
    if not camera.isOpened():
        logger.error(f"Cannot open {camera_id} camera")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               b'Camera not available'
               b'\r\n')
        return

    while not stop_threads:
        success, frame = camera.read()
        if not success:
            logger.error(f"Cannot read frame from {camera_id} camera")
            break
        
        # Store the latest frame
        if camera_id == "face":
            face_frame = frame
        else:
            plate_frame = frame
            
        # Encode frame as jpeg
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05)  # To control frame rate
    
    camera.release()

@app.route('/status')
def status():
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/video_feed/face')
def video_feed_face():
    return Response(generate_camera_frames("face"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/plate')
def video_feed_plate():
    return Response(generate_camera_frames("plate"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-checkin', methods=['POST'])
def start_checkin():
    global stop_threads
    stop_threads = False  # Reset the stop flag
    
    # Stop any existing process first
    if current_process and current_process.poll() is None:
        try:
            os.kill(current_process.pid, signal.SIGTERM)
            time.sleep(1)  # Give it time to terminate
        except Exception as e:
            logger.error(f"Error stopping previous process: {e}")
    
    # We'll just provide the camera feeds from Flask instead of running the script directly
    # This allows the client to see the camera feeds through the browser
    
    return jsonify({
        'success': True,
        'message': 'Checkin process started'
    })

@app.route('/start-checkout', methods=['POST'])
def start_checkout():
    global stop_threads
    stop_threads = False  # Reset the stop flag
    
    # Similar to start_checkin, we'll provide camera feeds through Flask
    
    return jsonify({
        'success': True,
        'message': 'Checkout process started'
    })

@app.route('/stop', methods=['POST'])
def stop_process():
    global current_process, stop_threads, face_camera, plate_camera
    
    stop_threads = True
    time.sleep(0.5)  # Give threads time to finish
    
    # Release cameras if they're open
    if face_camera is not None and face_camera.isOpened():
        face_camera.release()
    
    if plate_camera is not None and plate_camera.isOpened():
        plate_camera.release()
    
    # Kill any existing Python process
    if current_process and current_process.poll() is None:
        try:
            os.kill(current_process.pid, signal.SIGTERM)
        except Exception as e:
            logger.error(f"Error stopping process: {e}")
            
    current_process = None
    stop_threads = False  # Reset for next run
    
    return jsonify({'success': True, 'message': 'Process stopped'})

@app.route('/logs')
def get_logs():
    try:
        # Get logs from MongoDB
        logs_cursor = LOG_COLLECTION.find().sort("updated_at", pymongo.DESCENDING).limit(100)
        
        # Convert cursor to list and serialize ObjectId and datetime
        logs = []
        for log in logs_cursor:
            # Convert ObjectId to string
            log['_id'] = str(log['_id'])
            # Convert datetime to ISO format string
            if isinstance(log.get('updated_at'), datetime):
                log['updated_at'] = log['updated_at'].isoformat()
            logs.append(log)
            
        return jsonify({
            'success': True,
            'logs': logs
        })
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({
            'success': False,
            'message': f"Error fetching logs: {str(e)}",
            'logs': []
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)