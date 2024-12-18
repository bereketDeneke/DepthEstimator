import streamlit as st
import torch
import cv2
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Step-by-Step Visualization", layout="wide")

st.title("Step-by-Step Object Detection and Depth Estimation")
st.sidebar.title("Upload and Configuration")

uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
yolo_conf_thres = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.warning("Running on CPU. Consider using a GPU for improved performance.")

# Load Models
try:
    # Load YOLOv5
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device)
    yolo_model.conf = yolo_conf_thres
    
    # Load MiDaS
    midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
    midas_model.to(device)
    midas_model.eval()
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    midas_transform = midas_transforms.default_transform

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ------------------------------
# Helper Functions
# ------------------------------
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

def run_yolo(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    detections = results.pandas().xyxy[0]
    return detections

# ------------------------------
# Main Visualization Logic
# ------------------------------
if uploaded_image:
    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ------------------------------
    # Step 1: Show Uploaded Image
    # ------------------------------
    st.markdown("### Step 1: Uploaded Image")
    st.markdown(
        """**What’s happening:**  
        You've just uploaded your image. Below is the original image as-is. We'll now process it to identify objects and then measure their relative depths."""
    )
    st.image(frame, channels="BGR", use_container_width=True)

    # ------------------------------
    # Step 2: YOLOv5 Object Detection
    # ------------------------------
    st.markdown("---")
    st.markdown("### Step 2: Object Detection with YOLOv5")
    st.markdown(
        """**What’s happening now:**  
        We run the image through the YOLOv5 model to detect objects. The model returns bounding boxes, class labels, and confidence scores for each detected object.  
        
        Below, you'll see the image annotated with bounding boxes and labels indicating the identified objects."""
    )

    # Run YOLO
    detections = run_yolo(frame)
    annotated_frame = frame.copy()
    for _, row in detections.iterrows():
        if row['confidence'] < yolo_conf_thres:
            continue
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        object_name = row['name']
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{object_name} ({row['confidence']:.2f})"
        cv2.putText(annotated_frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    st.image(annotated_frame, channels="BGR", use_container_width=True)

    # ------------------------------
    # Step 3: Depth Estimation with MiDaS
    # ------------------------------
    st.markdown("---")
    st.markdown("### Step 3: Depth Estimation with MiDaS")
    st.markdown(
        """**What’s happening now:**  
        After detecting objects, we feed the same image into the MiDaS model to estimate depth. This provides a grayscale depth map where lighter regions represent areas closer to the camera and darker regions represent areas farther away.
        
        Below, you'll first see the raw depth map and then a combined visualization of objects and their approximate distances."""
    )

    # Estimate depth
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map = estimate_depth(image_rgb)
    depth_map = depth_map.max() - depth_map
    
    # Normalize the depth map for display
    depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)

    # Display Depth Map
    st.image(depth_colormap, caption="Depth Map (Colormap Applied)", use_container_width=True)

    # Annotate depth onto objects
    annotated_depth_frame = annotated_frame.copy()
    objects = []
    for _, row in detections.iterrows():
        if row['confidence'] < yolo_conf_thres:
            continue
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        object_depth = depth_map[ymin:ymax, xmin:xmax]
        median_depth = np.median(object_depth)

        # Overlay depth on bounding boxes
        label = f"{row['name']}: {median_depth:.2f}"
        cv2.putText(annotated_depth_frame, label, (xmin, ymax + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        objects.append((row['name'], median_depth))

    # Display annotated image with depth info
    st.markdown("### Combining Object Detection with Depth")
    st.markdown(
        """Here is the final result that shows detected objects along with their approximate distances. This helps in understanding not just *what* is in the image, but also *where* it is relative to the camera."""
    )
    st.image(annotated_depth_frame, channels="BGR", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Process Complete!**")
    st.markdown(
        """You have:
        1. **Uploaded an image.**
        2. **Detected objects using YOLOv5.**
        3. **Estimated relative depth using MiDaS.**

        This visualization pipeline helps break down each major step, making it easier to understand what’s happening under the hood."""
    )
else:
    st.markdown("### Please upload an image to begin the step-by-step visualization.")
