import numpy as np
import pickle
import faiss
import os
from PIL import Image


class PrepareDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def prepare(self, data_dir, model, save_path):
        # Danh sách lưu tên và vector đặc trưng
        list_face_encodings = []
        list_face_names = []

        # Duyệt qua các tệp trong thư mục
        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Đọc ảnh
                image_path = os.path.join(data_dir, filename)
                name = filename.split(".")[0]
                image = Image.open(image_path)
                # embedding
                emb_img = model.get_embedding(image)
                list_face_encodings.append(emb_img)
                list_face_names.append(name)

        # Chuyển danh sách vector đặc trưng thành numpy array
        faces_arr = np.array(list_face_encodings).reshape(-1, 512)
        # Đảm bảo rằng các vector đặc trưng có đúng số chiều
        n, d = faces_arr.shape[0], faces_arr.shape[1]
        # Xây dựng chỉ mục Faiss
        # dimension = known_face_encodings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(faces_arr)
        # Lưu chỉ mục Faiss và tên vào file
        faiss.write_index(index, os.path.join(save_path, "faiss_index.index"))
        with open(os.path.join(save_path, "face_names.pkl"), "wb") as f:
            pickle.dump(list_face_names, f)
