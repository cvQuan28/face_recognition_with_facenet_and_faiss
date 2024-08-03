import glob
import cv2
from PIL import Image
import faiss
import pickle
from src.emb import FaceNet
from src.ObjectDetection.ObjectDetection import Yolo_V8_Detection

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = FaceNet()
detect_model = Yolo_V8_Detection(r"D:\QDev\face_recognition\data\Y8_Face_Detection.onnx", method=2)

# Load index faiss and list names from file
index = faiss.read_index("data/faiss_index.index")

with open("data/face_names.pkl", "rb") as f:
    known_face_names = pickle.load(f)


def putText_to_image(img, org, caption, color=(0, 255, 0), size=0):
    img_height, img_width = img.shape[:2]
    if size != 0:
        size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    cv2.putText(img, caption, org,
                cv2.FONT_HERSHEY_SIMPLEX, size, color, text_thickness, cv2.LINE_AA)


def recognize_face(image, threshold=0.5):
    emb_img = model.get_embedding(image)
    distances, indices = index.search(emb_img, 1)
    if distances[0][0] < threshold:  # (*)
        match_index = indices[0][0]
        return known_face_names[match_index]
    return "Unknown"


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img_process = frame
    detect_results = detect_model.Predict(frame)
    img_process = detect_results['imProcess']
    for box, label in zip(detect_results['boxRects'], detect_results['labels']):
        if label == 0:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_area = frame[y1:y2, x1:x2]
            res = recognize_face(face_area)
            putText_to_image(img=img_process, org=(x1, y2 + 10), caption=res, size=5)
    cv2.imshow('video', img_process)
    c = cv2.waitKey(1)
    if c == 27:
        break
