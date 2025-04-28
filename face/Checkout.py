import cv2
import mediapipe as mp
import numpy as np
import pymongo
import time
from easyocr import Reader
from datetime import datetime
import face_recognition

# Kết nối MongoDB
client = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
face_collection = db["face_vectors"]
plate_collection = db["plates_and_face"]
log_collection = db["logs"]

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Khởi tạo EasyOCR
reader = Reader(['en', 'vi'])

# Lưu thời gian lần log cuối cùng
last_log_time = 0

# Lưu các giá trị nhận diện mặt và biển số trước đó
last_user_id = None
last_plate_text = None

# Hàm: Tìm user trong DB dựa vào vector khuôn mặt
def find_existing_user(face_vector):
    users = face_collection.find()
    for user in users:
        stored_vector = np.array(user["vector"])
        match = face_recognition.compare_faces([stored_vector], face_vector, tolerance=0.5)
        if match[0]:
            return user["user_id"]
    return None

# Hàm: Cập nhật biển số vào DB với user_id
def update_plate_with_user(plate_text, user_id):
    plate_collection.insert_one({
        "plate_text": plate_text,
        "user_id": user_id,
        "updated_at": datetime.now()
    })
    print(f"Đã thêm biển số: {plate_text} với user_id: {user_id}")

# Hàm: Xác nhận độ tin cậy của biển số
def get_best_plate(detection_results):
    max_confidence = 0
    best_plate = None
    for result in detection_results:
        text, confidence = result
        if confidence > max_confidence:
            max_confidence = confidence
            best_plate = text
    return best_plate

# Hàm: Nhận diện khuôn mặt
def detect_face_and_checkout(frame_face):
    rgb_frame = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            confidence = detection.score[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame_face.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            
            cv2.rectangle(frame_face, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame_face, f"Confidence: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if confidence > 0.6:
                face_roi = frame_face[y:y + height, x:x + width]
                if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    continue
                try:
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + width, y + height, x)])[0]
                    user_id = find_existing_user(face_encoding)
                    if user_id:
                        print(f"Đã nhận diện khuôn mặt của user {user_id}!")
                        return user_id
                except Exception as e:
                    print(f"[ERROR] {e}")
    return None

# Hàm: Nhận diện biển số
def detect_plate_and_checkout(frame_plate):
    grayscale = cv2.cvtColor(frame_plate, cv2.COLOR_BGR2GRAY)
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
    
    if number_plate_shape is None:
        if 'plate_not_found' not in globals():
            print("Không tìm thấy biển số.")
            global plate_not_found
            plate_not_found = True
        return None
    
    if 'plate_not_found' in globals():
        del plate_not_found

    x, y, w, h = cv2.boundingRect(number_plate_shape)
    number_plate = grayscale[y:y + h, x:x + w]

    detection_results = []
    for _ in range(3):
        detection = reader.readtext(number_plate)
        if detection:
            sorted_detection = sorted(detection, key=lambda x: x[0][0][1])
            combined_text = "".join([det[1] for det in sorted_detection]).replace(" ", "")
            confidence = max([det[2] for det in sorted_detection])
            detection_results.append((combined_text, confidence))

    if detection_results:
        best_plate = get_best_plate(detection_results)
        print(f"Biển số có độ tin cậy cao nhất: {best_plate}")
        return best_plate

    return None

# Hàm: Kiểm tra và thực hiện checkout
def checkout(user_id, plate_text):
    global last_log_time, last_user_id, last_plate_text
    current_time = time.time()
    
    if current_time - last_log_time > 2:
        if user_id and plate_text:
            if user_id != last_user_id or plate_text != last_plate_text:
                plate_collection.delete_one({"plate_text": plate_text, "user_id": user_id})
                face_collection.delete_one({"user_id": user_id})
                
                log_collection.insert_one({
                    "plate_text": plate_text,
                    "user_id": user_id,
                    "status": "checkout thành công",
                    "updated_at": datetime.now()
                })
                
                print(f"Biển số xe {plate_text} và khuôn mặt {user_id} đã khớp và checkout thành công!")
                
                last_user_id = user_id
                last_plate_text = plate_text
                last_log_time = current_time
            else:
                pass
        else:
            last_log_time = current_time
    else:
        pass

# Hàm chính để xử lý một cặp khung hình
def process_checkout_frame(frame_face, frame_plate):
    """
    Process a single pair of face and plate frames for checkout.
    Returns the processed frames and detection results.
    """
    # Xử lý nhận diện khuôn mặt
    user_id = detect_face_and_checkout(frame_face)

    # Xử lý nhận diện biển số
    plate_text = detect_plate_and_checkout(frame_plate)

    # Thực hiện checkout nếu cả khuôn mặt và biển số đều có
    checkout(user_id, plate_text)

    return frame_face, frame_plate, user_id, plate_text

if __name__ == "__main__":
    print("This script should be called by api.py with frames as input.")


import cv2
import mediapipe as mp
import numpy as np
import pymongo
import time
from easyocr import Reader
from datetime import datetime
import face_recognition

# Kết nối MongoDB
client = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
face_collection = db["face_vectors"]
plate_collection = db["plates_and_face"]

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Khởi tạo EasyOCR 
reader = Reader(['en', 'vi'])

# Khởi tạo hai camera 
cap_face = cv2.VideoCapture(1)  # Camera cho nhận diện khuôn mặt
cap_plate = cv2.VideoCapture(0)  # Camera cho nhận diện biển số xe

# Kiểm tra nếu mở camera thành công
if not cap_face.isOpened() or not cap_plate.isOpened():
    print("Không thể mở một trong các camera.")
    exit()

# Lưu thời gian lần log cuối cùng
last_log_time = 0

# Lưu các giá trị nhận diện mặt và biển số trước đó
last_user_id = None
last_plate_text = None

# Hàm: Tìm user trong DB dựa vào vector khuôn mặt
def find_existing_user(face_vector):
    users = face_collection.find()
    for user in users:
        stored_vector = np.array(user["vector"])
        match = face_recognition.compare_faces([stored_vector], face_vector, tolerance=0.5)
        if match[0]:  
            return user["user_id"]
    return None

# Hàm: Cập nhật biển số vào DB với user_id
def update_plate_with_user(plate_text, user_id):
    plate_collection.insert_one({
        "plate_text": plate_text,
        "user_id": user_id,
        "updated_at": datetime.now()
    })
    print(f"Đã thêm biển số: {plate_text} với user_id: {user_id}")

# Hàm: Xác nhận độ tin cậy của biển số
def get_best_plate(detection_results):
    max_confidence = 0
    best_plate = None
    for result in detection_results:
        text, confidence = result
        if confidence > max_confidence:
            max_confidence = confidence
            best_plate = text
    return best_plate

# Hàm: Nhận diện khuôn mặt
def detect_face_and_checkout(frame_face):
    rgb_frame = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            confidence = detection.score[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame_face.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            
            cv2.rectangle(frame_face, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame_face, f"Confidence: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if confidence > 0.6:  # Nếu độ tin cậy cao hơn ngưỡng
                face_roi = frame_face[y:y + height, x:x + width]
                if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    continue
                try:
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + width, y + height, x)])[0]
                    user_id = find_existing_user(face_encoding)
                    if user_id:
                        print(f"Đã nhận diện khuôn mặt của user {user_id}!")
                        return user_id
                except Exception as e:
                    print(f"[ERROR] {e}")
    return None

# Hàm: Nhận diện biển số
# Hàm: Nhận diện biển số
def detect_plate_and_checkout(frame_plate):
    grayscale = cv2.cvtColor(frame_plate, cv2.COLOR_BGR2GRAY)
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
    
    if number_plate_shape is None:
        # Chỉ log một lần khi không tìm thấy biển số
        if 'plate_not_found' not in globals():
            print("Không tìm thấy biển số.")
            global plate_not_found
            plate_not_found = True  # Đánh dấu là không tìm thấy biển số
        return None
    
    # Nếu tìm thấy biển số, reset trạng thái không tìm thấy biển số
    if 'plate_not_found' in globals():
        del plate_not_found  # Xóa biến đánh dấu khi đã tìm thấy biển số

    x, y, w, h = cv2.boundingRect(number_plate_shape)
    number_plate = grayscale[y:y + h, x:x + w]

    detection_results = []
    for _ in range(3):  # Quét biển số 3 lần
        detection = reader.readtext(number_plate)
        if detection:
            sorted_detection = sorted(detection, key=lambda x: x[0][0][1])
            combined_text = "".join([det[1] for det in sorted_detection]).replace(" ", "")
            confidence = max([det[2] for det in sorted_detection])
            detection_results.append((combined_text, confidence))

    if detection_results:
        best_plate = get_best_plate(detection_results)
        print(f"Biển số có độ tin cậy cao nhất: {best_plate}")
        return best_plate

    return None


# Hàm: Kiểm tra và thực hiện checkout
# Hàm: Kiểm tra và thực hiện checkout
# Hàm: Kiểm tra và thực hiện checkout
# Hàm: Kiểm tra và thực hiện checkout
# Hàm: Kiểm tra và thực hiện checkout
# Hàm: Kiểm tra và thực hiện checkout
def checkout(user_id, plate_text):
    global last_log_time, last_user_id, last_plate_text
    current_time = time.time()
    
    # Kiểm tra nếu đã qua 2 giây kể từ lần log cuối cùng
    if current_time - last_log_time > 2:
        if user_id and plate_text:
            if user_id != last_user_id or plate_text != last_plate_text:
                # Xóa dữ liệu của user với biển số này trong collection plates_and_face
                plate_collection.delete_one({"plate_text": plate_text, "user_id": user_id})
                
                # Xóa dữ liệu của khuôn mặt trong collection face_vectors
                face_collection.delete_one({"user_id": user_id})
                
                # Chuyển dữ liệu sang collection log
                log_collection = db["logs"]  # Sử dụng collection 'logs'
                log_collection.insert_one({
                    "plate_text": plate_text,
                    "user_id": user_id,
                    "status": "checkout thành công",
                    "updated_at": datetime.now()
                })
                
                # In ra thông báo checkout thành công
                print(f"Biển số xe {plate_text} và khuôn mặt {user_id} đã khớp và checkout thành công!")
                
                last_user_id = user_id
                last_plate_text = plate_text
                last_log_time = current_time
            else:
                # Nếu biển số và user đã xác nhận trước đó, không làm gì cả
                pass
        else:
            # Nếu không nhận diện được khuôn mặt hoặc biển số, không làm gì
            last_log_time = current_time
    else:
        # Không log nếu chưa đủ 2 giây
        pass





# Chạy chương trình chính
if __name__ == "__main__":
    while True:
        # Đọc khung hình từ hai camera
        ret_face, frame_face = cap_face.read()
        ret_plate, frame_plate = cap_plate.read()

        if not ret_face or not ret_plate:
            print("Không thể đọc khung hình từ một trong các camera.")
            break

        # Xử lý nhận diện khuôn mặt từ camera 1
        user_id = detect_face_and_checkout(frame_face)

        # Xử lý nhận diện biển số từ camera 2
        plate_text = detect_plate_and_checkout(frame_plate)

        # Thực hiện checkout nếu cả khuôn mặt và biển số đều có
        checkout(user_id, plate_text)

        # Hiển thị kết quả
        cv2.imshow("Khuôn mặt và Biển số", frame_face)
        cv2.imshow("Biển số xe", frame_plate)

        # Dừng khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng các camera và đóng cửa sổ
    cap_face.release()
    cap_plate.release()
    cv2.destroyAllWindows()
    print("Đã kết thúc!")