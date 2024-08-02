import glob
import cv2
from PIL import Image
import faiss
import pickle
from src.emb import FaceNet

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = FaceNet()

# Tải chỉ mục Faiss và tên từ file
index = faiss.read_index("faiss_index.index")
with open("data/face_names.pkl", "rb") as f:
    known_face_names = pickle.load(f)


# Hàm nhận diện khuôn mặt
def recognize_face(image_path):
    img = Image.open(image_path)
    emb_img = model.get_embedding(img)
    distances, indices = index.search(emb_img, 3)
    if distances[0][0] < 0.7:  # Ngưỡng để xác định sự tương đồng, có thể điều chỉnh
        match_index = indices[0][0]
        return known_face_names[match_index]

    print(f"Anh khong nhan dang duoc: {image_path}")
    return "Không tìm thấy kết quả"


# Ví dụ sử dụng
dir_image = r"E:\Data\Faces\V1092473"
list_image_paths = glob.glob(os.path.join(dir_image, "*.jpg"))
for img_path in list_image_paths:
    print(f"Kết quả nhận diện: {recognize_face(img_path)}")
