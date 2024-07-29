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

app = FastAPI()

# Initialize Jinja environment
templates = Environment(loader=FileSystemLoader("templates"))

# Global variable to hold the car count
car_count = 0

# Global variable to hold the frame counter
ab = 0

# Initialize YOLO model
model = YOLO('yolov8l.pt')

# Define class list for YOLO model
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

def detect_cars():
    global car_count
    global ab
    cap = cv2.VideoCapture('/Users/pavithra/Downloads/854671-hd_1920_1080_25fps.mp4')
    if not cap.isOpened():
        print("Error opening video stream")
        return

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

        for index, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row)
            c = class_list[d]
            if c in ['car', 'truck', 'boat','bicycle','motorcycle','bus']:
                car_count_in_frame += 1

        frame_counts.append(car_count_in_frame)
        #car_count.put(min(frame_counts))
        print(car_count_in_frame)
        ab=car_count_in_frame


        cv2.putText(frame, 'Count: ' + str(car_count_in_frame), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def increment_count():
    global car_counts
    while True:
        car_count.put(car_counts + 5)
        time.sleep(1)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    template = templates.get_template("index.html")
    html_content = template.render(car_count=car_count,ab=ab)
    return HTMLResponse(content=html_content, status_code=200)

# Start a separate thread for car detection
car_detection_thread = Thread(target=detect_cars)
car_detection_thread.daemon = True
car_detection_thread.start()

# Start a separate thread to increment count
increment_thread = Thread(target=increment_count)
increment_thread.daemon = True
increment_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
