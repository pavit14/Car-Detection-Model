from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from threading import Thread
from jinja2 import Environment, FileSystemLoader
import cv2
from ultralytics import YOLO
import pandas as pd
from collections import deque
from queue import Queue
import time
import math
model = YOLO('yolov8l.pt')

class Tracker:

    def __init__(self):
        self.center_points = {}
        self.id_count = 0
 
    def update(self, objects_rect):
       
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
 
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
 
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
 
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
 
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
 
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    

car_count = Queue()
car_count.put(0)

tracker = Tracker()

class_list = [
    'car', 'person', 'bicycle', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv2.VideoCapture('/Users/pavithra/Downloads/854671-hd_1920_1080_25fps.mp4')
if not cap.isOpened():
    print("Error opening video stream")
    
frame_counts = deque(maxlen=10)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(source=frame, conf=0.75)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    car_count_in_frame = 0
    list = []


    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        print(d)
        c = class_list[d]
        print(c)
        if c in ['car', 'truck', 'boat','bicycle','motorcycle','bus']:
            car_count_in_frame += 1
 
        bbox_id = tracker.update(list)
 
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)


    frame_counts.append(car_count_in_frame)
    car_count.put(min(frame_counts))
    print(car_count_in_frame)

    cv2.putText(frame, 'Count: ' + str(car_count_in_frame), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(frame, 'Count: ' + str(car_count.queue[0]), (60, 150), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imshow('Frame', frame)
    #cv2.imwrite('saved_image.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


 
