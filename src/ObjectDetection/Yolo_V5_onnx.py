import cv2, time
import numpy as np
from src.ObjectDetection.utils import xywh2xyxy, nms, draw_detections


# Infer in onnxruntime
class YOLOv5_onnx:
    def __init__(self, path, conf_thres=0.45, iou_thres=0.45, INPUT_WIDTH=640, INPUT_HEIGHT=640, score_thres=0.5):
        self.CONFIDENCE_THRESHOLD = conf_thres
        self.NMS_THRESHOLD = iou_thres
        self.INPUT_WIDTH = INPUT_WIDTH
        self.INPUT_HEIGHT = INPUT_HEIGHT
        self.SCORE_THRESHOLD = score_thres
        self.net = None

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.net = cv2.dnn.readNetFromONNX(path)
        print("Running on CPU")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_objects(self, image):
        outputs = self.pre_process(image)

        self.boxes, self.scores, self.class_ids = self.post_process(image, outputs)

        return self.boxes, self.scores, self.class_ids

    def pre_process(self, image):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
        # Sets the input to the network.
        self.net.setInput(blob)

        # Run the forward pass to get output of the output layers.
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs

    def post_process(self, input_image, outputs):
        """

            :param input_image:
            :param outputs:
            :param image_infor:
            :param hide_labels_type: =0,1,2,3:  =0 ẩn, =1: hiện số đơn giản, =2: hiện số có nền, =3: hiện số + độ chính xác
            :return:
            """
        # Display_Canvas_Width = taConf.cfg_lv0['Display_Canvas_Width']
        # Display_Canvas_Height = taConf.cfg_lv0['Display_Canvas_Height']
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # Resizing factor.
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT

        # Iterate through detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                # Get the index of max class score.
                class_id = np.argmax(classes_scores)
                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        # Perform non-maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        out_classes = []
        out_boxesRect = []
        out_scores = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            out_boxesRect.append([left, top, left + width, top + height])
            out_classes.append(class_ids[i])
            out_scores.append(confidences[i])

        return out_boxesRect, out_scores, out_classes

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)
