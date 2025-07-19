# Dự án Face Plate Detect

## Giới thiệu
Dự án **Face Plate Detect** là một ứng dụng sử dụng công nghệ nhận diện biển số xe đi kèm với khuôn mặt để làm nền tảng cho hệ thống bãi giữ xe thông minh, không cần sử dụng thẻ từ như các bãi xe hiện hành. Dự án này chỉ ở mức demo đơn giản.

## Cách sử dụng
Các bước cài đặt thư viện trên hệ điều hành Windows: (Đối với bước 1, 2 nếu đã thực hiện có thể bỏ qua).
- **Bước 1:** Tải python từ đường dẫn: https://www.python.org/downloads/ 
Sau khi cài đặt, tìm edit the system environment variables => Environment Variables => ở phần System variables, chọn Path => Edit => New => đưa đường dẫn sau vào C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python3x\ => Nhấn OK đến khi thoát. Khuyến khích tải phiên bản Python 3.11.x.
- **Bước 2:** Tải Git theo đường dẫn sau: https://git-scm.com/downloads
- **Bước 3:** Sau khi tải git, tìm git bash trên máy tính và khởi chạy => chọn nơi muốn lưu: “cd <đường_dẫn>” => để clone dự án, chạy lệnh sau: git clone https://github.com/Harry1720/Face_plate_detect
***Trường hợp máy tính có Python phiên bản cao hơn:***
- **Bước 4:** Truy cập vào https://github.com/pyenv-win/pyenv-win.  
- **Bước 5:** Kéo xuống mục Quick start và làm theo từng bước tương ứng để cài phiên bản Python 3.11.0 trên Pyenv.
***Sau khi cài đặt xong Pyenv trên CLI, chúng ta quay lại vào VSCode.***
- **Bước 6:** Ctrl + Shift + P để mở Command Palette.
- **Bước 7:** Bấm Selected Interpreter và chọn Pyenv vừa mới được tải bước trên với phiên bản Python 3.11.0.
- **Bước 8:** Mở một Terminal mới trên VSCode và kiểm tra phiên bản Python bằng lệnh "python --version" và nhận thông báo Python 3.11.0.
- **Bước 9:** Cài đặt môi trường ảo để cài thư viện bằng lệnh "python -m venv .venv".
- **Bước 10:** Kích hoạt môi trường ảo dùng lệnh ".venv/Scripts/activate"
- **Bước 11:** Cài đặt các thư viện cần thiết. Do phần nhận diện khuôn mặt trong dự án không hỗ trợ trên window nên phải cài thêm cmake và dlib.
  + ***Bước 11.1:*** cài đặt tay theo tuần tự các thư viện "pip install cmake" 
  + ***Bước 11.2:*** cài đặt tay theo tuần tự các thư viện "pip install dlib wheel". Nếu bị lỗi cài wheel, ta có thể sử dụng cách sau:
    - Tải file wheel qua đường dẫn https://github.com/Murtaza-Saeed/Dlib-Precompiled-Wheels-for-Python-on-Windows-x64-Easy-Installation/blob/master/dlib-19.24.1-cp311-cp311-win_amd64.whl (dùng cho Python 3.11 trên Windows 64-bit)
    - Cài đặt: pip install đường\dẫn\đến\dlib-19.24.6-cp311-cp311-win_amd64.whl
  + ***Bước 11.3:*** chạy lệnh "pip install -r requirements.txt" để cài đặt các thư viện còn lại 
  - Có thể dùng pip show <tên_thư_viện> để kiểm tra cài đặt.
- **Bước 12:** Cài đặt các thư viện cho FE 
  - Mở terminal mới và chuyển folder cd fe/vite-project
  - Gõ lệnh npm install
  - Tiếp tục gõ lệnh npm install axios
  - Cuối cùng là npm install react-router-dom
- **Bước 13:** Mở 2 terminal, tạm gọi là terminal FE và terminal BE.
- **Bước 14:** Ở terminal FE gõ "cd fe/vite-project", sau đó là "npm run dev". Cuối cùng nhấn đường dẫn trên terminal để chạy FE.
- **Bước 15:** Ở terminal BE gõ ".venv/Scripts/activate", sau đó gõ "cd face", rồi gõ "python api.py" để chạy BE.
***
Cảm ơn bạn đã quan tâm đến dự án của chúng tôi!
