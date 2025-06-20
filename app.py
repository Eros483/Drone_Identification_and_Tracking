import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import shutil
import tempfile
import os
import time

# Load the YOLO model
model = YOLO("runs/detect/drone-detector-extended7/weights/best.pt")

st.set_page_config(page_title="Drone Detector", layout="centered")
st.title("📡 Drone Detection")

# App state
mode = st.radio("Choose Mode", ["Image Detection", "Video Tracking"], horizontal=True)
reset = st.button("Reset", type="secondary")
analyse = st.button("Analyse", type="primary")

if reset:
    st.rerun()

if mode == "Image Detection":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file and analyse:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        results = model.predict(source=temp_path, save=True, conf=0.25)

        saved_dir = Path(results[0].save_dir)
        result_path = saved_dir / Path(temp_path).name

        for _ in range(10):
            if result_path.exists():
                break
            time.sleep(0.1)

        if result_path.exists():
            st.image(str(result_path), caption="Detection Output", use_column_width=True)
        else:
            st.error("Detection image could not be found. Try again.")

elif mode == "Video Tracking":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video and analyse:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
            temp_vid.write(uploaded_video.read())
            video_path = temp_vid.name

        # Run tracking
        results = model.track(source=video_path, conf=0.25, save=True)

        # Find the latest auto-generated 'track' directory
        track_root = Path("runs/detect/")
        track_dirs = sorted(track_root.glob("track*"), key=os.path.getmtime, reverse=True)

        if not track_dirs:
            st.error("No tracking folder found.")
        else:
            print(track_dirs)
            latest_track_dir = track_dirs[0]
            print(latest_track_dir)

            # Get first .mp4 from it
            tracked_video_path = None
            for f in latest_track_dir.glob("*.mp4"):
                tracked_video_path = f
                if tracked_video_path and tracked_video_path.exists():
                    print("Paths exist at first mp4")
                else:
                    if not tracked_video_path.exists():
                        print("Tracked video path not found 1")
                    if not tracked_video_path:
                        print("Tracked video path not found 2")
                    print("not found at 1st attempt")
                break

            for _ in range(20):
                if tracked_video_path and tracked_video_path.exists():
                    print("Paths exist  at second attempt")
                    break
                else:
                    if not tracked_video_path.exists():
                        print("Tracked video path not found 1")
                    if not tracked_video_path:
                        print("Tracked video path not found 2")
                    print("not found at 2nd attempt")
                time.sleep(0.5)

            if tracked_video_path and tracked_video_path.exists():
                # Copy to result_videos dir
                result_videos_dir = Path("result_videos")
                result_videos_dir.mkdir(exist_ok=True)

                final_video_path = result_videos_dir / tracked_video_path.name
                shutil.copy(tracked_video_path, final_video_path)

                # Remove original tracking dir
                shutil.rmtree(latest_track_dir)

                st.success("Tracking complete.")
                st.video(str(final_video_path))
            else:
                st.error("Tracking video could not be found at 3rd attempt.")
