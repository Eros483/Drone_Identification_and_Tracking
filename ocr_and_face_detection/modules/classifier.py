import cv2
import numpy as np
import tensorflow as tf
import json
import os

def run_custom_classifier(config):
    model_path = "modules/models/classifier_model.tflite"
    label_map_path = "modules/models/class_mapping.json"
    camera_index = config.get("camera_index", 0)

    # Load class mapping
    with open(label_map_path, "r") as f:
        class_mapping = json.load(f)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    img_size = input_shape[1]

    cap = cv2.VideoCapture(camera_index)

    #-------->comment out from below here for static image test
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}")
        return

    print("[INFO] Running TFLite classifier. Press 'q' to quit.")


    #from webcam or camera source

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        # Preprocess frame
        img = cv2.resize(frame, (img_size, img_size))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
        input_tensor = np.expand_dims(img, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        confidence = np.max(output_data)

        label = class_mapping[str(prediction)]

        # Overlay prediction
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("TFLite Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    #-------->comment out from above here for static image test

    #-------->comment out from below here for dynamic image test
    '''
    test_image_path = "modules/data/Closed_Eyes/s0001_00001_0_0_0_0_0_01.png"  # put any test image here

    if not os.path.exists(test_image_path):
        print(f"[ERROR] Test image not found at: {test_image_path}")
        return

    img = cv2.imread(test_image_path)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32))
    input_tensor = np.expand_dims(img_preprocessed, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = np.max(output_data)

    label = class_mapping[str(prediction)]
    print(f"[INFO] Predicted: {label} ({confidence*100:.2f}%)")

    # Optional display
    cv2.putText(img, f"{label} ({confidence*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

