import cv2
import pytesseract

def run_color_ocr_improved(config):
    cam_index = config.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera index {cam_index}")
        return

    print("[INFO] Starting live OCR with camera feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        # Process like before
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        try:
            text = pytesseract.image_to_string(binary, config='--psm 6 -c tessedit_char_whitelist=0123456789').strip()
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            text = ""

        digits_only = ''.join([c for c in text if c.isdigit()])
        if len(digits_only) >= 3:
            display_text = digits_only[:2] + "." + digits_only[2:]
        elif digits_only:
            display_text = digits_only
        else:
            display_text = "N/A"

        print(f"[INFO] OCR: {text} -> '{display_text}'")
        cv2.putText(frame, display_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Live OCR", frame)
        cv2.imshow("Binary", binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
