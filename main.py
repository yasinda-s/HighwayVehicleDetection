import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import numpy as np
from collections import defaultdict
import random

model=YOLO('yolov9c.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cap = cv2.VideoCapture('traffic2.mp4')

track_history = defaultdict(lambda: [])
track_colors = defaultdict(lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
counter_incoming = set()
counter_outgoing = set()
counter_base = set()

line_level = 325
incoming_line_x_range = (10, 600)
outgoing_line_x_range = (700, 1270)
offset = 7

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'avc1' if 'mp4v' doesn't work
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
out = cv2.VideoWriter('output.mp4', fourcc, 60, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, tracker='bytetrack.yaml', persist=True)
    needed_results = results[0].boxes.data
    formatted_results = needed_results.detach().cpu().numpy()
    df = pd.DataFrame(formatted_results).astype("float")

    for index, row in df.iterrows():
        x1, y1, x2, y2, track_id, conf, class_id = row
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        color = track_colors[track_id]

        track = track_history[track_id]
        track.append((cx, cy))
        if len(track) > 60:
            track.pop(0)

        if class_id == 2:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            counter_base.add(track_id)

            if incoming_line_x_range[0] <= cx <= incoming_line_x_range[1] and line_level - offset < cy < line_level + offset:
                counter_incoming.add(track_id)

            # Check for crossing the outgoing line
            if outgoing_line_x_range[0] <= cx <= outgoing_line_x_range[1] and line_level - offset < cy < line_level + offset:
                counter_outgoing.add(track_id)

        points = np.array(track).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.line(frame, (incoming_line_x_range[0], line_level), (incoming_line_x_range[1], line_level), (0, 0, 255), 3)
    cv2.line(frame, (outgoing_line_x_range[0], line_level), (outgoing_line_x_range[1], line_level), (0, 255, 0), 3)
    
    cv2.rectangle(frame, (50, 20), (300, 130), (0, 0, 0), -1)
    cv2.putText(frame, f'Total Unique Cars Seen - {len(counter_base)}', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Incoming Cars - {len(counter_incoming)}', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Outgoing Cars - {len(counter_outgoing)}', (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()



    
    
