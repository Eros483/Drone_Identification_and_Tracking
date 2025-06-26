import yaml

from modules.face_detect import run_face_detection
from modules.ocr_7seg import run_color_ocr_improved
from modules import classifier
from modules.ocr import run_easyocr

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config=load_config()
    print(f"[INFO] Running mode: {config['mode']}")
    mode=config.get("mode", "face")

    if mode=="face":
        run_face_detection(config)

    elif mode=="ocr-7-seg":
        run_color_ocr_improved(config)

    elif mode=="ocr":
        run_easyocr(config)

    elif mode=="classifier":    
        classifier.run_custom_classifier(config)

    else:
        print(f"Unsupported mode: {mode}")

if __name__ == "__main__":
    main()
