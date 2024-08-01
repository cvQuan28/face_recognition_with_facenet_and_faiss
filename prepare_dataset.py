import numpy as np
import pickle
import faiss

# Thư mục chứa ảnh khuôn mặt đã biết
known_faces_dir = r"E:\Data\Faces"

# Danh sách lưu tên và vector đặc trưng
known_face_encodings = []
known_face_names = []


