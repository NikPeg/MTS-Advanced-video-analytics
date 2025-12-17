from ultralytics import YOLO
from collections import defaultdict
from google.colab.patches import cv2_imshow
import cv2
import numpy as np

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("/content/1203.mp4")
tracks = defaultdict(lambda: [])  # id -> [(x, y), ...]

# Get video properties for VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_filename = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can also try 'XVID', 'DIVX'
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    highest_conf_class_name = "N/A"
    highest_conf_score = 0.0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        if len(confidences) > 0:
            max_conf_idx = np.argmax(confidences)
            highest_conf_class_id = class_ids[max_conf_idx]
            highest_conf_class_name = model.names[highest_conf_class_id]
            highest_conf_score = confidences[max_conf_idx]

        for box, tid in zip(boxes, ids):
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            tracks[tid].append((cx, cy))
            tracks[tid] = tracks[tid][-50:]  # последние 50 точек

            # bbox + id
            cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # траектория
            pts = np.array(tracks[tid], np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 255), 2)

    # Display highest confidence object class and score
    cv2.putText(frame, f"Highest Conf: {highest_conf_class_name} ({highest_conf_score:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Write the frame into the file 'output_video.mp4'
    out.write(frame)

    # Display the frame in Colab
    #cv2_imshow(frame)

cap.release()
out.release()
