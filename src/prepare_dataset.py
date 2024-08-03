import cv2
import numpy as np
import pickle
import faiss
import os
from PIL import Image
import torchvision
from src.emb import FaceNet
import logging
import json

# logging
logging.basicConfig(level=logging.INFO)


class PrepareDataset:
    def __init__(self):
        ...

    def add_a_new_face(self, data_dir, faiss_index_path, face_names_path, emb_model):
        # load index Faiss and list names from file
        index = faiss.read_index(faiss_index_path)
        with open(face_names_path, "rb") as f:
            known_face_names = pickle.load(f)
        data_folder = torchvision.datasets.ImageFolder(root=data_dir)
        for img_path_inf in data_folder.imgs:
            image = Image.open(img_path_inf[0])
            name = data_folder.classes[img_path_inf[1]]
            # embedding
            emb_img = emb_model.get_embedding(image)
            if emb_img is not None:
                index.add(emb_img)
                known_face_names.append(name)
        # save index Faiss and list names
        faiss.write_index(index, faiss_index_path)
        with open(face_names_path, "wb") as f:
            pickle.dump(known_face_names, f)
        logging.info(f"Finish adding a new face with Number of embeddings: {len(known_face_names)}")

    def prepare_data_from_dir(self, data_dir, emb_model, save_path):
        # List save names and feature vector
        list_face_encodings = []
        list_face_names = []

        data_folder = torchvision.datasets.ImageFolder(root=data_dir)
        # Save information of data
        with open(os.path.join(save_path, "infor.json"), "w") as f:
            json.dump(data_folder.class_to_idx, f, ensure_ascii=False, indent=4)
            logging.info(f"Save information of data to {os.path.join(save_path, 'infor.json')}")
        logging.info(f"Start embedding with Number of images: {len(data_folder.imgs)}")
        for img_path_inf in data_folder.imgs:
            image = Image.open(img_path_inf[0])
            name = data_folder.classes[img_path_inf[1]]
            # embedding
            emb_img = emb_model.get_embedding(image)
            if emb_img is not None:
                list_face_encodings.append(emb_img)
                list_face_names.append(name)
        logging.info(f"Finish embedding with Number of embeddings: {len(list_face_encodings)}")

        # convert list feature vector to numpy array
        faces_arr = np.array(list_face_encodings).reshape(-1, 512)
        np.save(os.path.join(save_path, "emb_data.npy"), faces_arr)

        logging.info(f"Save embeddings to {os.path.join(save_path, 'emb_data.npy')}")

        # Build index Faiss
        n, d = faces_arr.shape[0], faces_arr.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(faces_arr)

        # Save index Faiss and names to file
        faiss.write_index(index, os.path.join(save_path, "faiss_index.index"))
        logging.info(f"Save faiss index to {os.path.join(save_path, 'faiss_index.index')}")

        with open(os.path.join(save_path, "face_names.pkl"), "wb") as f:
            pickle.dump(list_face_names, f)
            logging.info(f"Save face names to {os.path.join(save_path, 'face_names.pkl')}")

    def extract_faces_from_dir(self, data_dir, save_path, detect_model):
        data_folder = torchvision.datasets.ImageFolder(root=data_dir)
        for img_path_inf in data_folder.imgs:
            image = cv2.imread(img_path_inf[0])
            image_basename = os.path.basename(img_path_inf[0])
            if image.any():
                name = data_folder.classes[img_path_inf[1]]
                name_path = os.path.join(save_path, name)
                os.makedirs(name_path, exist_ok=True)
                detect_results = detect_model.Predict(image)
                for box, label in zip(detect_results['boxRects'], detect_results['labels']):
                    if label == 0:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        face_area = image[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(name_path, image_basename), cv2.resize(face_area, (160, 160)))

    def extract_faces_from_video(self, video_path, save_path, detect_model, name):
        cap = cv2.VideoCapture(video_path)
        name_path = os.path.join(save_path, name)
        os.makedirs(name_path, exist_ok=True)
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img_process = frame
            if cnt % 10 == 5:
                detect_results = detect_model.Predict(frame)
                img_process = detect_results['imProcess']
                for box, label in zip(detect_results['boxRects'], detect_results['labels']):
                    if label == 0:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        face_area = frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(name_path, f'{name}_{cnt}.jpg'), cv2.resize(face_area, (160, 160)))
            cnt += 1
            cv2.imshow('video', img_process)
            c = cv2.waitKey(1)
            if c == 27 or cnt >= 100:
                break


if __name__ == "__main__":
    from src.ObjectDetection.ObjectDetection import Yolo_V8_Detection

    dataset = PrepareDataset()
    # dataset.prepare_data_from_dir(r"E:\Data\Faces\train", FaceNet(), r"D:\QDev\face_recognition\data")
    dataset.add_a_new_face(data_dir=r"E:\Data\new_face", emb_model=FaceNet(),
                           face_names_path=r'D:\QDev\face_recognition\data\face_names.pkl',
                           faiss_index_path=r'D:\QDev\face_recognition\data\faiss_index.index')
    # model = Yolo_V8_Detection(r"D:\QDev\face_recognition\data\Y8_Face_Detection.onnx", method=2)
    # dataset.extract_faces_from_video(0, r"E:\Data\new_face", model, "Hieu")
