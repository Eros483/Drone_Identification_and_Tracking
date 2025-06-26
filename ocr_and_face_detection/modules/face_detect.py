import cv2
import json
from ultralytics import YOLO
from PIL import Image
from supervision import Detections

# Path to YOLOv8 model
MODEL_PATH = "modules/models/face_detect.pt"

# Load the model once
model = YOLO(MODEL_PATH)

def run_face_detection(config):
    cam_index = config.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}")
        return

    print("[INFO] YOLOv8 Face Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Convert frame to PIL for YOLO input
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = model(pil_img)
        results = Detections.from_ultralytics(output[0])

        # Draw bounding boxes
        for box in results.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
