import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Tải mô hình đã được huấn luyện
model = keras.models.load_model('/home/neeyuhuynh/cggm-mammography-classification/best_xception_model.h5')

def load_and_preprocess_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm trục batch
    img_array = img_array / 255.0  # Chuẩn hóa hình ảnh
    return img_array

# Đường dẫn đến hình ảnh thử nghiệm
test_image_path = ['/home/neeyuhuynh/cggm-mammography-classification/Selection_033.png', 
                '/home/neeyuhuynh/cggm-mammography-classification/Selection_034.png', 
                '/home/neeyuhuynh/cggm-mammography-classification/Selection_035.png',
                '/home/neeyuhuynh/cggm-mammography-classification/Selection_036.png',
                '/home/neeyuhuynh/cggm-mammography-classification/Selection_038.png',
                '/home/neeyuhuynh/cggm-mammography-classification/455268501_1248216242841461_8188819718041734836_n.jpg',
                '/home/neeyuhuynh/cggm-mammography-classification/456363928_1032931481791112_5728557553983587609_n.jpg']

# Tiền xử lý hình ảnh
# Lặp qua các ảnh và dự đoán
# Ngưỡng xác định phân loại
threshold = 0.5

# Lặp qua các dự đoán và in kết quả
for img_path in test_image_path:
    processed_image = load_and_preprocess_image(img_path)
    prediction = model.predict(processed_image)[0][0]  # Lấy giá trị dự đoán từ mảng trả về

    if prediction > threshold:
        print(f"Dự đoán cho {img_path}: Malignant (Xác suất: {prediction:.2f})")
    else:
        print(f"Dự đoán cho {img_path}: Benign (Xác suất: {prediction:.2f})")

