import streamlit as st
import torch
import cv2
import numpy as np
import os
import pyttsx3
import openai
import threading
import asyncio
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Real-Time Object Detection and Depth Estimation", layout="wide")

# Initialize session states
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "is_describing" not in st.session_state:
    st.session_state.is_describing = False
if "stop_tts" not in st.session_state:
    st.session_state.stop_tts = False
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# Global variables
narration_lock = threading.Lock()
narration_in_progress = False

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.warning("Running on CPU. Consider using a GPU for improved performance.")

try:
    # Load YOLOv5
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device)

    # Load MiDaS
    midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
    midas_model.to(device)
    midas_model.eval()
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    midas_transform = midas_transforms.default_transform

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Options")
mode = st.sidebar.selectbox("Select Input Mode", ["Upload Video", "Live Stream"])
yolo_conf_thres = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
yolo_model.conf = yolo_conf_thres

@torch.no_grad()
def estimate_depth(image):
    input_image = midas_transform(image).to(device)
    depth = midas_model(input_image)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:2],
        mode='bicubic',
        align_corners=False
    ).squeeze()
    depth = depth.cpu().numpy()
    return depth

tts_engine = pyttsx3.init()

def speak_text(text):
    def tts_thread():
        global narration_in_progress
        tts_engine.setProperty('rate', 150)
        with narration_lock:
            narration_in_progress = True
        tts_engine.say(text)
        tts_engine.runAndWait()
        with narration_lock:
            narration_in_progress = False
    thread = threading.Thread(target=tts_thread)
    thread.start()


async def describe_scene(objects):
    # Set API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # If no objects detected, return a generic message
    if not objects:
        return "No objects detected in the scene."

    # Compute center points of objects for spatial reasoning
    for obj in objects:
        obj['center_x'] = (obj['xmin'] + obj['xmax']) / 2.0
        obj['center_y'] = (obj['ymin'] + obj['ymax']) / 2.0

    # Derive relative spatial relationships:
    relations = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i == j:
                continue

            # Determine horizontal relation
            if obj1['center_x'] < obj2['center_x']:
                horizontal_relation = f"{obj1['name']} is to the left of {obj2['name']}"
            else:
                horizontal_relation = f"{obj1['name']} is to the right of {obj2['name']}"

            # Determine vertical relation
            if obj1['center_y'] < obj2['center_y']:
                vertical_relation = f"{obj1['name']} is above {obj2['name']}"
            else:
                vertical_relation = f"{obj1['name']} is below {obj2['name']}"

            # We include both horizontal and vertical relations
            relations.append(horizontal_relation)
            relations.append(vertical_relation)

    # Remove duplicate relations
    relations = list(set(relations))

    # Sort objects by depth to understand which is closer/farther
    # Lower median_depth might mean closer to the camera.
    objects_sorted_by_depth = sorted(objects, key=lambda x: x['median_depth'])
    depth_ordering = []

    for i, obj in enumerate(objects_sorted_by_depth):
        if i == 0:
            depth_ordering.append(f"The closest object is a {obj['name']} at about {obj['median_depth']:.2f} milimeters.")
        else:
            depth_ordering.append(f"There is a {obj['name']} farther at about {obj['median_depth']:.2f} milimeters.")

    # Build prompt
    object_list_str = "\n".join([
        f"- {obj['name']} at approx x={obj['center_x']:.1f}, y={obj['center_y']:.1f}, depth={obj['median_depth']:.2f}mm"
        for obj in objects
    ])

    relations_str = "\n".join(relations)
    depth_str = "\n".join(depth_ordering)
    prompt = f"""
                You are a helpful assistant for visually impaired users. You are given a set of objects detected in a scene, along with their approximate positions and relative distances from the camera.

                Your task:

                Create a clear, concise, and auditory-friendly description of the scene that can help the user build a mental picture.
                Use spatial terms like "to the left," "to the right," "in front of," "behind," "closer," and "farther" to describe the layout and relationships between objects.
                Avoid technical terms like "x/y-coordinates" or "depth values" and focus on intuitive descriptions.
                Information provided:

                Objects detected:
                {object_list_str}
                Positional relationships (spatial layout):
                {relations_str}
                Relative distances (closer/farther):
                {depth_str}
                Instructions for the description:

                Clearly mention what objects are present in the scene.
                Describe their spatial arrangement using terms that are natural and easy to visualize, such as "to the left of," "above," or "in front of."
                Highlight which objects are nearer or farther from the user's perspective.
                Ensure the description is simple, conversational, and paints a vivid auditory image of the scene.
            """
    try:
        # New syntax for OpenAI API
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        return f"Error communicating with OpenAI API: {e}"

def process_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    detections = results.pandas().xyxy[0]
    depth_map = estimate_depth(image_rgb)
    depth_map = depth_map.max() - depth_map

    objects = []
    annotated_frame = frame.copy()
    for _, row in detections.iterrows():
        if row['confidence'] < yolo_conf_thres:
            continue
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        object_name = row['name']
        object_depth = depth_map[ymin:ymax, xmin:xmax]
        median_depth = np.median(object_depth)

        objects.append({
            "name": object_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "median_depth": median_depth
        })

        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{object_name}: {median_depth:.2f}mm"
        cv2.putText(annotated_frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return annotated_frame, objects

async def handle_scene_description(objects):
    description = await describe_scene(objects)
    st.write(description)
    if not st.session_state.stop_tts:
        speak_text(description)

st.title("Real-Time Object Detection, Relative Depth Estimation, and Scene Narration")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Play"):
        # If currently describing, stop TTS
        if st.session_state.is_describing:
            st.session_state.stop_tts = True
            tts_engine.stop()
            st.session_state.is_describing = False
        st.session_state.is_playing = True

with col2:
    if st.button("Pause"):
        st.session_state.is_playing = False

with col3:
    if st.button("Describe Scene"):
        # Pause first
        st.session_state.is_playing = False
        st.session_state.is_describing = True
        st.session_state.stop_tts = False

stframe = st.empty()

def read_current_frame():
    # This function opens the video each run, seeks to current frame, and returns that frame
    # For "Live Stream", we handle differently.
    if st.session_state.video_path is None:
        return None
    cap = cv2.VideoCapture(st.session_state.video_path)
    if not cap.isOpened():
        st.error("Unable to open video.")
        return None
    # Seek to current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

if mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file and st.session_state.video_path is None:
        # Save once
        st.session_state.video_path = "uploaded_video.mp4"
        with open(st.session_state.video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.current_frame = 0

    if st.session_state.video_path is not None:
        frame = read_current_frame()
        if frame is not None:
            annotated_frame, objects = process_frame(frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            # If describing
            if st.session_state.is_describing:
                asyncio.run(handle_scene_description(objects))
                st.session_state.is_describing = False
                # Remain paused after describing

            # If playing, advance to next frame for next run
            if st.session_state.is_playing and not st.session_state.is_describing:
                # Attempt to go to next frame
                # We force a rerun by st.experimental_rerun()
                st.session_state.current_frame += 1
                st.experimental_rerun()

elif mode == "Live Stream":
    st.write("Press 'Start' to begin live streaming from your webcam.")
    # For live stream, we can't preserve exact frame position like a file.
    # But we can handle similarly by storing cap and not resetting.
    # However, the user specifically complains about video reset on button click.
    # Live stream doesn't have 'position' to reset. It's continuous.
    # We'll implement similar logic: keep capturing new frames only if playing.
    if "cap" not in st.session_state:
        st.session_state.cap = None

    if st.sidebar.button("Start"):
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if ret:
            annotated_frame, objects = process_frame(frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            if st.session_state.is_describing:
                asyncio.run(handle_scene_description(objects))
                st.session_state.is_describing = False

            # If playing, keep streaming by forcing rerun
            if st.session_state.is_playing:
                st.experimental_rerun()
        else:
            st.write("No frame captured. Check webcam.")
    else:
        st.write("Webcam not started or not available.")

st.sidebar.write("Use Play/Pause/Describe Scene without losing your current position.")
