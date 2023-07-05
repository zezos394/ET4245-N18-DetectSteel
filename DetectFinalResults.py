import cv2
import numpy as np
from ultralytics import YOLO

# Tải mô hình YOLO đã được huấn luyện trước
model = YOLO("best.pt")

# Đường dẫn đến ảnh
image_path = "testImg2.jpg"
image = cv2.imread(image_path)

# Thay đổi kích thước ảnh để vừa với 640x640 và vẫn giữ tỷ lệ khung hình
height, width, _ = image.shape
max_size = max(height, width)
scale = 640 / max_size
resized_image = cv2.resize(image, None, fx=scale, fy=scale)

# Phát hiện đối tượng
predictions = model.predict(resized_image)

# Khởi tạo đếm đối tượng
object_count = 0

# Vẽ  trên ảnh đã thay đổi kích thước và hiển thị nhãn
for prediction in predictions:
    for box in prediction.boxes:
        x_min, y_min, x_max, y_max = box.xyxy[0]
        conf = box.conf[0]

        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        if conf > 0.8:
            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            labelcount = f"{object_count + 1}" 
            text_size, _ = cv2.getTextSize(labelcount, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            text_x = int((x_min + x_max) / 2 - text_size[0] / 2)
            text_y = int((y_min + y_max) / 2 + text_size[1] / 2)
            cv2.putText(resized_image, labelcount, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255),2)

            object_count += 1

# Hiển thị số lượng đối tượng
count_steel = f"CountingSteel: {object_count}"
cv2.putText(resized_image, count_steel, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("CountingSteel", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
