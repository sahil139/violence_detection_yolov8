import os

from ultralytics import YOLO
import cv2

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

results = model.predict(source='0', show = True, conf=0.5)
print(results)