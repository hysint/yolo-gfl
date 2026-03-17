import os
import cv2
import time
import tempfile
import streamlit as st
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO DEMO",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATHS = {
    "YOLO-GFL": "main/yolo-gfl/results/weights/best.pt",
    "YOLOv12": "main/yolov12/results/weights/best.pt",
}

OUTPUT_DIR = {"videos": os.path.join("demo/webui", "videos"), "images": os.path.join("demo/webui", "images")}
for path in OUTPUT_DIR.values():
    os.makedirs(path, exist_ok=True)

defaults = {
    "model_choice": list(MODEL_PATHS.keys())[0],
    "is_processing": False,
    "mode": "Images",
    "processing_complete": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def clear_uploads():
    for k in ("image_uploader", "video_uploader"):
        if k in st.session_state:
            try:
                st.session_state.pop(k)
            except Exception:
                pass
    st.session_state.is_processing = False
    st.session_state.processing_complete = False

st.sidebar.title("YOLO Demo")

st.sidebar.subheader("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select YOLO model",
    list(MODEL_PATHS.keys()),
    index=list(MODEL_PATHS.keys()).index(st.session_state.model_choice),
    key="model_choice",
    on_change=clear_uploads
)

confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.15, 0.05)

st.sidebar.subheader("Media Type")
mode = st.sidebar.radio("Choose type", ["Images", "Videos"], key="mode", on_change=clear_uploads)

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    return YOLO(path)

model_path = MODEL_PATHS.get(st.session_state.model_choice)
if not model_path or not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)
if model is None:
    st.stop()

def generate_filename(base_name: str, model_name: str, ext: str, output_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        OUTPUT_DIR[output_type],
        f"{Path(base_name).stem}_{model_name}_{timestamp}{ext}"
    )

def process_image(uploaded_file, model, confidence, model_choice):
    try:
        t0 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        uploaded_file.seek(0)

        results = model(tmp_path, conf=confidence, verbose=False)
        annotated = results[0].plot()

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.subheader("Original Image")
            st.image(
                uploaded_file,
                caption=f"Original ({uploaded_file.name})",
                use_container_width=True
            )
        with col2:
            st.subheader("Detection Results")
            st.image(
                annotated,
                caption=f"Detections ({uploaded_file.name})",
                channels="BGR",
                use_container_width=True
            )

        save_path = generate_filename(uploaded_file.name, model_choice, ".jpg", "images")
        cv2.imwrite(save_path, annotated)

        processing_time = time.time() - t0
        st.success(f"Image saved to: {save_path} (Processing time: {processing_time:.2f}s)")

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    except Exception as e:
        st.error(f"Error processing image {uploaded_file.name}: {str(e)}")

def process_video(uploaded_file, model, confidence, model_choice):
    cap = None
    out = None
    try:
        t0 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            st.error(f"Could not open video file: {uploaded_file.name}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        save_path = generate_filename(uploaded_file.name, model_choice, ".mp4", "videos")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        video_container = st.container()
        with video_container:
            stframe = st.empty()

        progress = st.progress(0)

        frame_count = 0

        while cap.isOpened() and st.session_state.is_processing:
            ret, frame = cap.read()
            if not ret:
                break

            if st.session_state.model_choice != model_choice:
                st.warning("Model changed during inference. Stopping...")
                break

            results = model(frame, conf=confidence, verbose=False)
            annotated = results[0].plot()

            out.write(annotated)
            frame_count += 1

            if frame_count % 5 == 0 or frame_count == 1:
                stframe.image(
                    annotated,
                    channels="BGR",
                    caption=f"Processing frame {frame_count}/{total_frames}",
                    use_container_width=True
                )

            if total_frames:
                progress.progress(min(frame_count / total_frames, 1.0))

        if frame_count > 0:
            processing_time = time.time() - t0
            st.success(f"Video saved to: {save_path} "
                       f"(Processing time: {processing_time:.2f}s)")
        else:
            st.warning("No frames were processed")

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    except Exception as e:
        st.error(f"Error processing video {uploaded_file.name}: {str(e)}")
    finally:
        try:
            if cap:
                cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

st.header("YOLO Object Detection")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", st.session_state.model_choice)
    with col2:
        st.metric("Confidence", f"{confidence:.2f}")
    with col3:
        st.metric("Mode", st.session_state.mode)

st.divider()

if st.session_state.mode == "Images":
    _ = st.file_uploader(
        "Upload image(s) - Images will be displayed in full column width",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="image_uploader",
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
else:
    _ = st.file_uploader(
        "Upload video(s) - Videos will be displayed in full width",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        accept_multiple_files=True,
        key="video_uploader",
        help="Supported formats: MP4, AVI, MOV, MKV, WMV"
    )

current_uploaded = st.session_state.get("image_uploader") if st.session_state.mode == "Images" else st.session_state.get("video_uploader")

if current_uploaded:
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.is_processing:
            if st.button("Start Processing", type="primary", use_container_width=True):
                st.session_state.is_processing = True
                st.session_state.processing_complete = False
                st.rerun()
        else:
            if st.button("End Processing", type="secondary", use_container_width=True):
                st.session_state.is_processing = False
                st.rerun()

if st.session_state.is_processing:
    st.divider()
    st.subheader("Processing Results")

    try:
        if st.session_state.mode == "Images":
            uploaded_files = st.session_state.get("image_uploader") or []
            total_files = len(uploaded_files)
            if total_files == 0:
                st.warning("No image files to process.")
            else:
                overall_progress = st.progress(0)
                st.info(f"Processing {total_files} image(s)...")

                for idx, uploaded_file in enumerate(uploaded_files):
                    if not st.session_state.is_processing:
                        break
                    with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
                        process_image(uploaded_file, model, confidence, st.session_state.model_choice)
                    overall_progress.progress((idx + 1) / total_files)

        elif st.session_state.mode == "Videos":
            uploaded_files = st.session_state.get("video_uploader") or []
            total_files = len(uploaded_files)
            if total_files == 0:
                st.warning("No video files to process.")
            else:
                st.info(f"Processing {total_files} video(s)...")
                for uploaded_file in uploaded_files:
                    if not st.session_state.is_processing:
                        break
                    with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
                        process_video(uploaded_file, model, confidence, st.session_state.model_choice)

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
    finally:
        st.session_state.is_processing = False
        st.session_state.processing_complete = True
