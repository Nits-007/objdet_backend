import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        # Load YAML for label names
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        """
        Perform predictions on the input image and process results.
        """
        row, col, _ = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[:row, :col] = image

        # YOLO input size
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(
            input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False
        )
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Process predictions and annotate image
        detections, processed_image = self.process_predictions(preds, input_image)
        return processed_image, detections

    def process_predictions(self, preds, input_image):
        """
        Process YOLO predictions, apply NMS, and annotate the image.
        """
        INPUT_WH_YOLO = 640
        image_height, image_width = input_image.shape[:2]
        x_factor = image_width / INPUT_WH_YOLO
        y_factor = image_height / INPUT_WH_YOLO

        # Extract bounding boxes, confidences, and class IDs
        confidences = []
        boxes = []
        class_ids = []

        for detection in preds[0]:
            confidence = detection[4]
            if confidence >= 0.4:  # Confidence threshold
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                if class_scores[class_id] > 0.25:  # Class confidence threshold
                    cx, cy, w, h = detection[0:4]

                    # Convert YOLO coordinates to bounding box
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.45)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h
                label = self.labels[class_ids[i]]
                confidence = confidences[i]

                # Append detection details
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                # Annotate the image
                cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    input_image,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return detections, input_image

    def compute_manhattan_distance(self, detections):
        """
        Compute Manhattan distances between detected objects.
        """
        distances = []
        num_objects = len(detections)

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                obj1, obj2 = detections[i], detections[j]
                x1_center = (obj1['x1'] + obj1['x2']) // 2
                y1_center = (obj1['y1'] + obj1['y2']) // 2
                x2_center = (obj2['x1'] + obj2['x2']) // 2
                y2_center = (obj2['y1'] + obj2['y2']) // 2

                distance = abs(x1_center - x2_center) + abs(y1_center - y2_center)
                distances.append({
                    "object1": obj1['label'],
                    "object2": obj2['label'],
                    "distance": distance
                })

        return distances
