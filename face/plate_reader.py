import pymongo
import datetime
import numpy as np
from easyocr import Reader
import cv2
from pymongo import MongoClient
from PIL import Image, ImageDraw, ImageFont  # Đảm bảo import đúng
import face_recognition  # Thêm dòng này để sử dụng face_recognition

# Khởi tạo MongoDB client
client = MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["face_vectors"]
plate_collection = db["plates and face"]

# Khởi tạo EasyOCR
reader = Reader(['en', 'vi'])

# Khởi tạo camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

# Các biến cho font và màu sắc
fontpath = "./arial.ttf"  # Đảm bảo đường dẫn font chính xác
font = ImageFont.truetype(fontpath, 32)  # Khởi tạo font với kích thước 32
b, g, r, a = 0, 255, 0, 0  # Màu xanh lá

# Danh sách lưu biển số đã nhận diện
detected_plates = []

# Hàm tìm user gần nhất (dựa trên thời gian truy cập)
def find_most_recent_user():
    user = collection.find().sort("last_access", pymongo.DESCENDING).limit(1)
    return user[0] if user.alive else None

# Hàm lưu biển số và user_id vào MongoDB
def update_plate_with_user(plate_text, user_id):
    # Kiểm tra nếu user đã có biển số nào
    existing_plate = plate_collection.find_one({"user_id": user_id})
    if existing_plate:
        print(f"User {user_id} đã có biển số: {existing_plate['plate_text']}. Không thêm biển số mới.")
        return  # Nếu đã có biển số, không thêm nữa

    # Kiểm tra nếu biển số đã có trong cơ sở dữ liệu
    result = plate_collection.update_one(
        {"plate_text": plate_text},
        {
            "$set": {
                "user_id": user_id,
                "updated_at": datetime.datetime.now()
            }
        }
    )

    if result.matched_count > 0:
        print(f"Đã cập nhật biển số: {plate_text} với user_id: {user_id}")
    else:
        # Nếu không tìm thấy biển số trong cơ sở dữ liệu, thêm mới vào
        result = plate_collection.insert_one({
            "plate_text": plate_text,
            "user_id": user_id,
            "updated_at": datetime.datetime.now()
        })
        print(f"Đã thêm mới biển số: {plate_text} với user_id: {user_id}")


while True:
    ret, img = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera!")
        break

    img = cv2.resize(img, (640, 480))  # Giảm kích thước ảnh để xử lý nhanh hơn


    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    if number_plate_shape is None:
        draw.text((150, 500), "Không thấy bảng số xe", font=font, fill=(b, g, r, a))
    else:
        cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
        x, y, w, h = cv2.boundingRect(number_plate_shape)
        number_plate = grayscale[y:y + h, x:x + w]

        detection = reader.readtext(number_plate)
        if len(detection) == 0:
            draw.text((150, 500), "Không thấy bảng số xe", font=font, fill=(b, g, r, a))
        else:
            sorted_detection = sorted(detection, key=lambda x: x[0][0][1])
            combined_text = "".join([det[1] for det in sorted_detection]).replace(" ", "")

            if combined_text not in detected_plates:
                detected_plates.append(combined_text)
                draw.text((200, 500), "Biển số: " + combined_text, font=font, fill=(b, g, r, a))
                draw.text((200, 550), "Đã chụp biển số!", font=font, fill=(b, g, r, a))
                print("Phát hiện mới:", combined_text)

                # Tìm user gần nhất
                most_recent_user = find_most_recent_user()
                if most_recent_user:
                    user_id = most_recent_user["user_id"]
                    print(f"User gần nhất: {user_id}")

                    # Cập nhật biển số vào cơ sở dữ liệu với user_id
                    update_plate_with_user(combined_text, user_id)
                else:
                    print("Không tìm thấy user gần nhất!")

            else:
                draw.text((200, 500), "Biển số đã quét trước đó", font=font, fill=(255, 255, 0, a))

    img = np.array(img_pil)
    cv2.imshow('Plate Detection', img)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Tổng số biển số nhận diện:", detected_plates)
