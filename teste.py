from ultralytics import YOLO 
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2 

model = YOLO("yolov8x.pt")

results = model.predict(source="1" , show=True)

print(results)

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs