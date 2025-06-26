import cv2
import easyocr

def run_easyocr(config):
    cam_index = config.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"[ERROR] Camera index {cam_index} could not be opened.")
        return

    reader = easyocr.Reader(['en'], gpu=config.get("use_gpu", False))
    print("[INFO] EasyOCR Camera Mode Started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not read.")
            break

        results = reader.readtext(frame)

        for (bbox, text, confidence) in results:
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            br = tuple(map(int, br))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("EasyOCR Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
