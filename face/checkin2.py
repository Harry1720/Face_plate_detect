# plate
import pymongo
import datetime
import numpy as np
from easyocr import Reader
import cv2
import pymongo
from PIL import Image, ImageDraw, ImageFont  # Đảm bảo import đúng
import re

#face
import mediapipe as mp
import os
import face_recognition
from datetime import datetime
import time
import uuid
import heapq

# Khởi tạo MongoDB client
client =  pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["face_vectors"]
plate_collection = db["plates and face"]

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# In ra folder ảnh khuôn mặt -> dùng cho debug lưu hình ảnh là chủ yếu.
output_dir = "detected_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Khởi tạo EasyOCR - Plate
reader = Reader(['en', 'vi'])

# Cam Face tùy máy dùng 0 hoặc 1
cap_face = cv2.VideoCapture(0) # Renamed from cap for clarity
if not cap_face.isOpened():
    print("Không thể mở camera - face!")
    exit()

# Cam Plate
cap_plate = cv2.VideoCapture(1)
if not cap_plate.isOpened():
    print("Không thể mở camera - plate!")
    # Clean up face camera if plate camera fails
    if cap_face.isOpened():
        cap_face.release()
    exit()

# Lưu trạng thái nhận diện - Face
last_saved_time = 0
face_candidates = []  # Danh sách các khuôn mặt ứng viên (face_vector, confidence)
max_candidates = 3    # Số lượng ứng viên tối đa
min_confidence = 0.6  # Ngưỡng tin cậy tối thiểu

# Các biến cho font và màu sắc - plate
fontpath = "./arial.ttf"  # Đảm bảo đường dẫn font chính xác
font = ImageFont.truetype(fontpath, 32)  # Khởi tạo font với kích thước 32
b, g, r, a = 0, 255, 0, 0  # Màu xanh lá

# Danh sách lưu biển số đã nhận diện - Plate
detected_plates = []

# HÀM: Tìm user trong DB dựa vào vector khuôn mặt - Face
def find_existing_user(face_vector):
    users = collection.find()
    for user in users:
        stored_vector = np.array(user["vector"])
        match = face_recognition.compare_faces([stored_vector], face_vector, tolerance=0.5)
        if match[0]:
            return user["user_id"]  # Trả về user_id nếu khớp
    return None  # Không tìm thấy

# HÀM: Nhận diện khuôn mặt - Face
def detect_face(frame):
    """
    Nhận diện khuôn mặt trong khung hình và trả về ID người dùng nếu tìm thấy
    Modifies the input frame with drawings.
    """
    global last_saved_time, face_candidates

    # Đổi màu từ BGR (OpenCV) sang RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chạy nhận diện khuôn mặt
    results = face_detection.process(rgb_frame)

    user_id_detected = None # Track detected user_id in this call

    # Nếu phát hiện khuôn mặt
    if results.detections:
        for detection in results.detections:
            # Lấy confidence score
            confidence = detection.score[0]

            # Lấy tọa độ bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)

            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Hiển thị confidence score
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Chỉ xử lý khuôn mặt có độ tin cậy trên ngưỡng
            if confidence > min_confidence:
                # Cắt khuôn mặt
                face_roi = frame[y:y+height, x:x+width]

                # Kiểm tra khuôn mặt có hợp lệ không
                if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    continue

                # Chuyển thành dlib.rectangle để dễ sử dụng với face_recognition
                top, right, bottom, left = y, x + width, y + height, x

                # Tìm face encoding
                try:
                    face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]

                        # Thêm vào danh sách ứng viên
                        heapq.heappush(face_candidates, (confidence, face_encoding))

                        # Chỉ giữ max_candidates khuôn mặt có độ tin cậy cao nhất
                        if len(face_candidates) > max_candidates:
                            heapq.heappop(face_candidates)  # Loại bỏ khuôn mặt có độ tin cậy thấp nhất

                        # Kiểm tra nếu đủ ứng viên và đã qua 2 giây kể từ lần lưu cuối cùng
                        current_time = time.time()
                        if len(face_candidates) >= max_candidates and (current_time - last_saved_time) > 2:
                            # Lấy khuôn mặt có độ tin cậy cao nhất
                            best_confidence, best_face_encoding = max(face_candidates)

                            # Kiểm tra xem khuôn mặt này đã tồn tại trong DB chưa
                            existing_user_id = find_existing_user(best_face_encoding)

                            if existing_user_id:
                                print(f"[EXISTING USER] Đã nhận diện user {existing_user_id}!")

                                # Cập nhật thời gian truy cập
                                collection.update_one(
                                    {"user_id": existing_user_id},
                                    {"$set": {"last_access": current_time}}
                                )

                                # Reset danh sách ứng viên
                                face_candidates = []
                                last_saved_time = current_time

                                user_id_detected = existing_user_id # Store the ID
                            else:
                                # Tạo user mới chỉ khi chưa tồn tại
                                new_user_id = uuid.uuid4().hex[:8]  # Tạo ID ngắn gọn

                                # Lưu ảnh khuôn mặt
                                timestamp = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d_%H-%M-%S')
                                filename = f"{new_user_id}_{best_confidence:.2f}_{timestamp}.jpg"
                                filepath = os.path.join(output_dir, filename)
                                cv2.imwrite(filepath, face_roi)

                                # Lưu vector khuôn mặt vào DB
                                collection.insert_one({
                                    "user_id": new_user_id,
                                    "vector": best_face_encoding.tolist(),
                                    "created_at": current_time,
                                    "last_access": current_time
                                })

                                print(f"[NEW USER] Đã thêm user {new_user_id} vào database!")
                                print(f"[INFO] Lưu ảnh {filename} cho user {new_user_id}")

                                # Reset danh sách ứng viên
                                face_candidates = []
                                last_saved_time = current_time

                                user_id_detected = new_user_id # Store the ID
                except Exception as e:
                    print(f"[ERROR] Face encoding/processing error: {e}")

    # Hiển thị số lượng ứng viên đã thu thập
    cv2.putText(frame, f"Candidates: {len(face_candidates)}/{max_candidates}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return user_id_detected # Return the ID found in this specific frame processing

# -----------------------------------------------------------------------------------

# Hàm tìm user gần nhất (dựa trên thời gian truy cập) - Plate/Shared
def find_most_recent_user():
    try:
        # Find returns a cursor, need to check if it has results
        users_cursor = collection.find().sort("last_access", pymongo.DESCENDING).limit(1)
        user = next(users_cursor, None) # Get the first document or None
        return user if user else None
    except Exception as e:
        print(f"Error finding most recent user: {e}")
        return None
def is_valid_plate(plate):
    # Chuẩn hóa biển số thành chữ hoa và loại bỏ khoảng trắng dư thừa
    plate = plate.upper().strip()

    # Loại bỏ các ký tự đặc biệt ngoài dấu gạch nối và dấu chấm
    plate = re.sub(r'[^A-Z0-9-.]', '', plate)

    # Cho phép các định dạng như: 86-A12345, 86-AA08199, 86-AA08199.01,...
    if re.match(r'^\d{2}-[A-Z]{1,2}\d{3,5}(\.\d{2})?$', plate):
        return True
    return False
# Hàm lưu biển số và user_id vào MongoDB - Plate/Shared
def update_plate_with_user(plate_text, user_id):
    if not is_valid_plate(plate_text):
        print(f"Biển số {plate_text} không hợp lệ. Không thể gán cho user.")
        return
    # Kiểm tra nếu user đã có biển số nào
    existing_plate_entry = plate_collection.find_one({"user_id": user_id})
    if existing_plate_entry:
        print(f"User {user_id} đã có biển số: {existing_plate_entry['plate_text']}. Không thêm biển số mới.")
        return  # Nếu đã có biển số, không thêm nữa

    # Kiểm tra nếu biển số này đã tồn tại (có thể chưa gán user_id)
    existing_plate_text = plate_collection.find_one({"plate_text": plate_text})
    if existing_plate_text:
        # Nếu biển số tồn tại nhưng chưa có user_id, cập nhật nó
        if "user_id" not in existing_plate_text or existing_plate_text["user_id"] is None:
             result = plate_collection.update_one(
                {"plate_text": plate_text},
                {
                    "$set": {
                        "user_id": user_id,
                        "updated_at": datetime.now() # Use standard datetime
                    }
                }
            )
             if result.modified_count > 0:
                 print(f"Đã cập nhật biển số: {plate_text} với user_id: {user_id}")
             else:
                 print(f"Không thể cập nhật biển số {plate_text} cho user {user_id} (có thể đã được gán).")
        else:
            # Biển số đã tồn tại và đã có user_id khác
            print(f"Biển số {plate_text} đã được gán cho user {existing_plate_text['user_id']}. Không cập nhật.")
    else:
        # Nếu không tìm thấy biển số trong cơ sở dữ liệu, thêm mới vào
        try:
            result = plate_collection.insert_one({
                "plate_text": plate_text,
                "user_id": user_id,
                "updated_at": datetime.now() # Use standard datetime
            })
            print(f"Đã thêm mới biển số: {plate_text} với user_id: {user_id}")
        except Exception as e:
             print(f"Lỗi khi thêm biển số mới: {e}")


# ------------------ MERGED WHILE LOOP ------------------
while True:
    # --- Read Frames ---
    ret_plate, frame_plate = cap_plate.read()
    ret_face, frame_face = cap_face.read()

    # --- Check Frame Validity ---
    if not ret_plate:
        print("Không thể đọc khung hình từ camera plate!")
        break
    if not ret_face:
        print("Không thể đọc khung hình từ camera face!")
        break

    # --- Plate Processing ---
    frame_plate_processed = frame_plate.copy() # Work on a copy for drawing
    frame_plate_processed = cv2.resize(frame_plate_processed, (640, 480)) # Resize for processing

    grayscale = cv2.cvtColor(frame_plate_processed, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    number_plate_shape = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approximation) == 4:
            number_plate_shape = approximation
            break

    # Prepare for drawing text using PIL
    # Convert BGR frame (potentially with contours) to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame_plate_processed, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    plate_text_detected_this_frame = None # Track if a new plate was detected

    if number_plate_shape is None:
        draw.text((150, 400), "Không thấy bảng số xe", font=font, fill=(b, g, r, a)) # Adjusted Y position
    else:
        # Draw contour directly on the OpenCV frame *before* converting to PIL
        cv2.drawContours(frame_plate_processed, [number_plate_shape], -1, (255, 0, 0), 3)

        # Re-convert the frame with contour to PIL for subsequent text drawing
        img_pil = Image.fromarray(cv2.cvtColor(frame_plate_processed, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        x, y, w, h = cv2.boundingRect(number_plate_shape)
        # Ensure ROI coordinates are valid
        y_end = min(y + h, frame_plate_processed.shape[0])
        x_end = min(x + w, frame_plate_processed.shape[1])
        number_plate_roi = grayscale[y:y_end, x:x_end]

        if number_plate_roi.size > 0: # Check if ROI is valid
            detection = reader.readtext(number_plate_roi)
            if len(detection) == 0:
                draw.text((150, 400), "Không thấy bảng số xe", font=font, fill=(b, g, r, a))
            else:
                # Sort detections by y-coordinate, then x-coordinate to handle multi-line plates
                sorted_detection = sorted(detection, key=lambda det: (det[0][0][1], det[0][0][0]))
                combined_text = "".join([det[1] for det in sorted_detection]).replace(" ", "").upper() # Standardize

                # Basic filtering for common OCR errors or very short strings (adjust as needed)
                if len(combined_text) > 4: # Example: require at least 5 characters
                    if combined_text not in detected_plates:
                        detected_plates.append(combined_text)
                        plate_text_detected_this_frame = combined_text # Store detected plate

                        draw.text((150, 400), "Biển số: " + combined_text, font=font, fill=(b, g, r, a))
                        draw.text((150, 440), "Đã chụp biển số!", font=font, fill=(b, g, r, a))
                        print("Phát hiện mới:", combined_text)

                        # --- Link Plate to User ---
                        most_recent_user = find_most_recent_user()
                        if most_recent_user:
                            user_id_for_plate = most_recent_user["user_id"]
                            print(f"User gần nhất: {user_id_for_plate}")
                            update_plate_with_user(combined_text, user_id_for_plate)
                        else:
                            print("Không tìm thấy user gần nhất để gán biển số!")
                        # --------------------------
                    else:
                        # Plate already seen, display it but maybe different color/message
                        draw.text((150, 400), "Biển số: " + combined_text, font=font, fill=(255, 255, 0, a)) # Yellow for seen
                        draw.text((150, 440), "(Đã quét trước đó)", font=font, fill=(255, 255, 0, a))
                else:
                    # Detected text too short or likely invalid
                    draw.text((150, 400), "Phát hiện không rõ", font=font, fill=(255, 165, 0, a)) # Orange for unclear
        else:
             draw.text((150, 400), "Lỗi ROI biển số", font=font, fill=(255, 0, 0, a)) # Red for error

    # Convert final PIL image (with text/contours) back to BGR for OpenCV display
    frame_plate_display = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


    # --- Face Processing ---
    frame_face_processed = frame_face.copy() # Work on a copy
    # detect_face modifies frame_face_processed in place and returns user_id if found/created
    user_id_detected = detect_face(frame_face_processed)

    if user_id_detected:
        # Display the detected/created user ID on the face frame
        cv2.putText(frame_face_processed, f"User: {user_id_detected}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # --- Display Frames ---
    cv2.imshow('Plate Detection', frame_plate_display)
    cv2.imshow("Face Detection", frame_face_processed)

    # --- Exit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap_plate.release()
cap_face.release()
cv2.destroyAllWindows()

print("Chương trình kết thúc.")