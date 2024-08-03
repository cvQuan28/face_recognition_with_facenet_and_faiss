import cv2
import os
from src.ObjectDetection.utils import draw_detections


class Yolo_V8_Detection:
    def __init__(self, pathWeight='', method=2, conf_thres=0.7, iou_thres=0.5):
        self.net = None
        self.method = method
        self.conf = conf_thres
        self.iou = iou_thres
        if method == 1:
            from ultralytics import YOLO
            self.net = YOLO(task='predict', model=pathWeight)
        if method == 2:
            from src.ObjectDetection.Yolo_v8_onnxruntime import YOLOv8_onnxruntime
            self.net = YOLOv8_onnxruntime(pathWeight, conf_thres, iou_thres)

    def __call__(self, image):
        return self.Predict(image)

    def Predict(self, input_image):
        imProcess = input_image.copy()
        mResults_Detect = {}

        if self.method == 1:
            results = self.net.predict(imProcess, conf=self.conf, iou=self.iou, verbose=False)
            boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
            labels = results[0].boxes.cls.int().cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()
            annotated_frame = results[0].plot()
            mResults_Detect['scores'] = scores
            mResults_Detect['imProcess'] = annotated_frame
            mResults_Detect['labels'] = labels
            mResults_Detect['boxRects'] = boxes_xyxy
        if self.method == 2:
            boxes_xyxy, scores, labels = self.net.detect_objects(imProcess)
            # if boxes_xyxy != [] and scores != [] and labels != [] :
            #     boxes_xyxy = boxes_xyxy.tolist()
            #     scores = scores.tolist()
            #     labels = labels.tolist()
            # Visualize the results on the frame
            annotated_frame = self.net.draw_detections(imProcess)
            mResults_Detect['scores'] = scores
            mResults_Detect['imProcess'] = annotated_frame
            mResults_Detect['labels'] = labels
            mResults_Detect['boxRects'] = boxes_xyxy

        return mResults_Detect

    def drawLabel(self, image, x, y, caption, color):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        cv2.putText(det_img, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, text_thickness, cv2.LINE_AA)

        return det_img


class Yolo_V5_Detection:
    def __init__(self, pathWeight='', method=2, conf_thres=0.7, iou_thres=0.5):
        self.net = None
        self.method = method
        self.conf = conf_thres
        self.iou = iou_thres
        if method == 1:
            self.load_weight_pt(pathWeight)
        if method == 2:
            from src.ObjectDetection.Yolo_V5_onnx import YOLOv5_onnx
            self.net = YOLOv5_onnx(pathWeight, conf_thres, iou_thres)
        pass

    def load_weight_pt(self, pathWeight):
        import torch
        dir_weight = os.path.dirname(pathWeight)
        self.net = torch.hub._load_local(dir_weight, 'custom', path=pathWeight, device='cpu')
        self.net.iou = self.iou
        self.net.conf = self.conf
        self.net.classes = [0, 1]

    def ObjectDetection(self, input_image):
        imProcess = input_image.copy()
        mResults_Detect = {}

        if self.method == 1:
            scores = []
            labels = []
            boxes_xyxy = []

            results = self.net(imProcess)
            result_pandas = results.pandas().xyxy[0]
            list_data_results = result_pandas.values.tolist()
            if len(list_data_results) == 1:
                lb = list_data_results[0][5]
                x1 = int(list_data_results[0][0])
                y1 = int(list_data_results[0][1])
                x2 = int(list_data_results[0][2])
                y2 = int(list_data_results[0][3])
                score = float(list_data_results[0][4])
                boxes_xyxy.append([x1, y1, x2, y2])
                labels.append(lb)
                scores.append(score)
            annotated_frame = draw_detections(imProcess, boxes_xyxy, scores, labels, 0.4)
            mResults_Detect['scores'] = scores
            mResults_Detect['imProcess'] = annotated_frame
            mResults_Detect['labels'] = labels
            mResults_Detect['boxRects'] = boxes_xyxy
        if self.method == 2:
            boxes_xyxy, scores, labels = self.net.detect_objects(imProcess)
            # if boxes_xyxy != [] and scores != [] and labels != [] :
            #     boxes_xyxy = boxes_xyxy.tolist()
            #     scores = scores.tolist()
            #     labels = labels.tolist()
            # Visualize the results on the frame
            annotated_frame = self.net.draw_detections(imProcess)
            mResults_Detect['scores'] = scores
            mResults_Detect['imProcess'] = annotated_frame
            mResults_Detect['labels'] = labels
            mResults_Detect['boxRects'] = boxes_xyxy

        return mResults_Detect
