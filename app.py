import streamlit as st
import torch
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, retinanet_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
import shutil
import numpy as np
import cv2
from collections import defaultdict

# Set page config as the first Streamlit command
st.set_page_config(page_title="Neutrosophic Traffic Signal Controller", layout="centered")

# Initialize models with error handling
try:
    yolov8 = YOLO("yolov8n.pt")  
except Exception as e:
    st.error(f"Failed to load YOLOv8 model: {e}")
    yolov8 = None

try:
    if os.path.exists("yolov5s.pt"):
        yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=False)
    else:
        yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'yolov5s.pt')
except Exception as e:
    st.error(f"Failed to load YOLOv5 model: {e}")
    st.info("If the error persists, try clearing the torch hub cache by deleting the folder: ~/.cache/torch/hub/")
    yolov5 = None

try:
    ssd = ssdlite320_mobilenet_v3_large(pretrained=True)
    ssd.eval()
except Exception as e:
    st.error(f"Failed to load SSD model: {e}")
    ssd = None

try:
    retinanet = retinanet_resnet50_fpn(pretrained=True)
    retinanet.eval()
except Exception as e:
    st.error(f"Failed to load RetinaNet model: {e}")
    retinanet = None

# Define vehicle and obstacle classes
yolov8_vehicles = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
yolov5_vehicles = {'car', 'motorcycle', 'bus', 'truck'}
ssd_vehicles = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
retinanet_vehicles = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

yolov8_obstacles = {0: "person", 16: "cow", 17: "dog", 18: "horse", 19: "sheep"}
yolov5_obstacles = {'person', 'cow', 'dog', 'horse', 'sheep'}
ssd_obstacles = {1: "person", 21: "cow", 24: "dog", 25: "horse", 26: "sheep"}
retinanet_obstacles = {1: "person", 21: "cow", 24: "dog", 25: "horse", 26: "sheep"}

def save_to_file(img_name, vehicle_counts, vehicles, coverage, density, obstacle_coverage, t, i, f, model):
    file_name = f"{model.replace(' ', '')}.txt"
    with open(file_name, "a") as file:
        if os.path.getsize(file_name) == 0:
            file.write("Image\tSummary\n")
        
        summary = f"Total Vehicles: {len(vehicles)} | " + " | ".join([f"{v.title()}s: {c}" for v, c in vehicle_counts.items()])
        density_text = f"Coverage: {coverage:.1f}% | Obstacle Coverage: {obstacle_coverage:.1f}% | Density: {density.upper()}"
        logic = f"T: {t:.2f} | I: {i:.2f} | F: {f:.2f}"
        data = f"{summary} | {density_text} | {logic}"
        
        file.write(f"{img_name}\t{data}\n")

def yolov8_density(img_path, result):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    total = h * w
    heatmap = np.zeros((h, w), dtype=np.float32)
    vehicle_area = 0
    obstacle_area = 0
    boxes = []

    for box in result[0].boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if cls in yolov8_vehicles:
            vehicle_area += area
            boxes.append((x1, y1, x2, y2))
        elif cls in yolov8_obstacles:
            obstacle_area += area
        heatmap[y1:y2, x1:x2] += 1

    vehicle_coverage = (vehicle_area / total) * 100 if boxes else 0
    obstacle_coverage = (obstacle_area / total) * 100

    if vehicle_coverage < 10:
        level = "low"
    elif vehicle_coverage < 25:
        level = "medium"
    else:
        level = "high"

    heatmap = np.clip(heatmap, 0, 255)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255
    heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, colored_heatmap, 0.3, 0)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return level, overlay, vehicle_coverage, obstacle_coverage

def yolov5_density(img_path, result):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    total = h * w
    heatmap = np.zeros((h, w), dtype=np.float32)
    vehicle_area = 0
    obstacle_area = 0
    boxes = []

    for *box, conf, cls in result.xyxy[0]:
        name = result.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if name in yolov5_vehicles:
            vehicle_area += area
            boxes.append((x1, y1, x2, y2))
        elif name in yolov5_obstacles:
            obstacle_area += area
        heatmap[y1:y2, x1:x2] += 1

    vehicle_coverage = (vehicle_area / total) * 100
    obstacle_coverage = (obstacle_area / total) * 100

    if vehicle_coverage < 10:
        level = "low"
    elif vehicle_coverage < 25:
        level = "medium"
    else:
        level = "high"

    heatmap = np.clip(heatmap, 0, 255)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255
    heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, colored_heatmap, 0.3, 0)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return level, overlay, vehicle_coverage, obstacle_coverage, boxes

def ssd_density(img_path, result):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    total = h * w
    heatmap = np.zeros((h, w), dtype=np.float32)
    vehicle_area = 0
    obstacle_area = 0
    boxes = []

    box_list = result['boxes'].detach().cpu().numpy()
    label_list = result['labels'].detach().cpu().numpy()
    score_list = result['scores'].detach().cpu().numpy()

    for i in range(len(box_list)):
        if score_list[i] > 0.5:
            cls = label_list[i].item()
            x1, y1, x2, y2 = map(int, box_list[i])
            area = (x2 - x1) * (y2 - y1)
            if cls in ssd_vehicles:
                vehicle_area += area
                boxes.append((x1, y1, x2, y2))
            elif cls in ssd_obstacles:
                obstacle_area += area
            heatmap[y1:y2, x1:x2] += 1

    vehicle_coverage = (vehicle_area / total) * 100
    obstacle_coverage = (obstacle_area / total) * 100

    if vehicle_coverage < 10:
        level = "low"
    elif vehicle_coverage < 25:
        level = "medium"
    else:
        level = "high"

    heatmap = np.clip(heatmap, 0, 255)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255
    heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, colored_heatmap, 0.3, 0)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return level, overlay, vehicle_coverage, obstacle_coverage, boxes

def retinanet_density(img_path, result):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    total = h * w
    heatmap = np.zeros((h, w), dtype=np.float32)
    vehicle_area = 0
    obstacle_area = 0
    boxes = []

    box_list = result['boxes'].detach().cpu().numpy()
    label_list = result['labels'].detach().cpu().numpy()
    score_list = result['scores'].detach().cpu().numpy()

    for i in range(len(box_list)):
        if score_list[i] > 0.5:
            cls = label_list[i].item()
            x1, y1, x2, y2 = map(int, box_list[i])
            area = (x2 - x1) * (y2 - y1)
            if cls in retinanet_vehicles:
                vehicle_area += area
                boxes.append((x1, y1, x2, y2))
            elif cls in retinanet_obstacles:
                obstacle_area += area
            heatmap[y1:y2, x1:x2] += 1

    vehicle_coverage = (vehicle_area / total) * 100
    obstacle_coverage = (obstacle_area / total) * 100

    if vehicle_coverage < 10:
        level = "low"
    elif vehicle_coverage < 25:
        level = "medium"
    else:
        level = "high"

    heatmap = np.clip(heatmap, 0, 255)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255
    heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, colored_heatmap, 0.3, 0)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return level, overlay, vehicle_coverage, obstacle_coverage, boxes

st.title("🚦 Neutrosophic logic Based Traffic Management System 🚦")

model = st.selectbox("Select Model", ["YOLOv8", "YOLOv5", "SSD", "RetinaNet"])  # Removed "Faster R-CNN"
uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_name = uploaded.name
    with open("input.jpg", "wb") as f:
        f.write(uploaded.read())

    st.image("input.jpg", caption="Uploaded Image", use_column_width=True)

    if model == "YOLOv8":
        if yolov8 is None:
            st.error("YOLOv8 model not loaded. Please check the error above.")
        else:
            with st.spinner("Running YOLOv8 Detection..."):
                result = yolov8("input.jpg", save=True)
                result_path = os.path.join(result[0].save_dir, "input.jpg")
                density, density_img, vehicle_coverage, obstacle_coverage = yolov8_density("input.jpg", result)

                counts = defaultdict(int)
                vehicles = []

                for box in result[0].boxes:
                    cls = int(box.cls.item())
                    if cls in yolov8_vehicles:
                        vehicle = yolov8_vehicles[cls]
                        vehicles.append(vehicle)
                        counts[vehicle] += 1

                t = min(vehicle_coverage / 100, 1.0)
                i = min(obstacle_coverage / 50, 1.0)  # Scale obstacle coverage to 0-1
                f = max(0.0, 1.0 - (t + i) / 2)  # Improved F calculation

                if i > 0.6:  # High obstacle coverage
                    signal = "🔴 RED — High Obstacle Density"
                elif i > 0.3 and t > 0.4:  # Moderate obstacles and traffic
                    signal = "🟡 YELLOW — Traffic + Obstacles"
                elif i > 0.3:  # Moderate obstacles
                    signal = "🟡 YELLOW — Obstacles Detected"
                elif t > 0.5:  # High traffic
                    signal = "🟢 GREEN — High Traffic"
                elif t > 0.2:  # Moderate traffic
                    signal = "🟡 YELLOW — Moderate Traffic"
                else:  # Low traffic
                    signal = "🟢 GREEN — Low Traffic"

    elif model == "YOLOv5":
        if yolov5 is None:
            st.error("YOLOv5 model not loaded. Please check the error above.")
        else:
            with st.spinner("Running YOLOv5 Detection..."):
                result = yolov5("input.jpg")
                density, density_img, vehicle_coverage, obstacle_coverage, boxes = yolov5_density("input.jpg", result)

                counts = defaultdict(int)
                vehicles = []

                for *box, conf, cls in result.xyxy[0]:
                    name = result.names[int(cls)]
                    if name in yolov5_vehicles:
                        vehicles.append(name)
                        counts[name] += 1

                t = min(vehicle_coverage / 100, 1.0)
                i = min(obstacle_coverage / 50, 1.0)
                f = max(0.0, 1.0 - (t + i) / 2)

                if i > 0.6:
                    signal = "Red — 🚷 High Obstacle Density"
                elif i > 0.3 and t > 0.4:
                    signal = "Yellow — 🚷 Traffic + Obstacles"
                elif i > 0.3:
                    signal = "Yellow — 🚷 Obstacles Detected"
                elif t > 0.5:
                    signal = "Green — 🚦 High Traffic"
                elif t > 0.2:
                    signal = "Yellow — 🚦 Moderate Traffic"
                else:
                    signal = "Red — 🚦 Low Traffic"

    elif model == "SSD":
        if ssd is None:
            st.error("SSD model not loaded. Please check the error above.")
        else:
            with st.spinner("Running SSD Detection..."):
                img = Image.open("input.jpg").convert("RGB")
                tensor = F.to_tensor(img).unsqueeze(0)
                
                with torch.no_grad():
                    result = ssd(tensor)[0]
                
                density, density_img, vehicle_coverage, obstacle_coverage, boxes = ssd_density("input.jpg", result)

                counts = defaultdict(int)
                vehicles = []
                box_list = result['boxes'].detach().cpu().numpy()
                label_list = result['labels'].detach().cpu().numpy()
                score_list = result['scores'].detach().cpu().numpy()

                for idx in range(len(box_list)):
                    if score_list[idx] > 0.5:
                        cls = label_list[idx].item()
                        if cls in ssd_vehicles:
                            vehicle = ssd_vehicles[cls]
                            vehicles.append(vehicle)
                            counts[vehicle] += 1

                t = min(vehicle_coverage / 100, 1.0)
                i = min(obstacle_coverage / 50, 1.0)
                f = max(0.0, 1.0 - (t + i) / 2)

                if i > 0.6:
                    signal = "Red — 🚷 High Obstacle Density"
                elif i > 0.3 and t > 0.4:
                    signal = "Yellow — 🚷 Traffic + Obstacles"
                elif i > 0.3:
                    signal = "Yellow — 🚷 Obstacles Detected"
                elif t > 0.5:
                    signal = "Green — 🚦 High Traffic"
                elif t > 0.2:
                    signal = "Yellow — 🚦 Moderate Traffic"
                else:
                    signal = "Red — 🚦 Low Traffic"

    else:  # RetinaNet
        if retinanet is None:
            st.error("RetinaNet model not loaded. Please check the error above.")
        else:
            with st.spinner("Running RetinaNet Detection..."):
                img = Image.open("input.jpg").convert("RGB")
                tensor = F.to_tensor(img).unsqueeze(0)
                
                with torch.no_grad():
                    result = retinanet(tensor)[0]
                
                density, density_img, vehicle_coverage, obstacle_coverage, boxes = retinanet_density("input.jpg", result)

                counts = defaultdict(int)
                vehicles = []
                box_list = result['boxes'].detach().cpu().numpy()
                label_list = result['labels'].detach().cpu().numpy()
                score_list = result['scores'].detach().cpu().numpy()

                for idx in range(len(box_list)):
                    if score_list[idx] > 0.5:
                        cls = label_list[idx].item()
                        if cls in retinanet_vehicles:
                            vehicle = retinanet_vehicles[cls]
                            vehicles.append(vehicle)
                            counts[vehicle] += 1

                t = min(vehicle_coverage / 100, 1.0)
                i = min(obstacle_coverage / 50, 1.0)
                f = max(0.0, 1.0 - (t + i) / 2)

                if i > 0.6:
                    signal = "Red — 🚷 High Obstacle Density"
                elif i > 0.3 and t > 0.4:
                    signal = "Yellow — 🚷 Traffic + Obstacles"
                elif i > 0.3:
                    signal = "Yellow — 🚷 Obstacles Detected"
                elif t > 0.5:
                    signal = "Green — 🚦 High Traffic"
                elif t > 0.2:
                    signal = "Yellow — 🚦 Moderate Traffic"
                else:
                    signal = "Red — 🚦 Low Traffic"

    if globals().get(model.lower().replace(' ', '_'), None) is not None:
        save_to_file(img_name, counts, vehicles, vehicle_coverage, density, obstacle_coverage, t, i, f, model)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Vehicle Summary")
            if vehicles:
                st.success(f"Total Vehicles: {len(vehicles)}")
                for vehicle, count in counts.items():
                    st.write(f"• {vehicle.title()}s: {count}")
            else:
                st.warning("No vehicles detected")

        with col2:
            st.subheader("🧠 Density Analysis")
            st.metric("Road Coverage (Vehicles)", f"{vehicle_coverage:.1f}%")
            st.metric("Obstacle Coverage", f"{obstacle_coverage:.1f}%")
            st.write(f"Density Level: {density.upper()}")
            if density == "low":
                st.progress(30)
            elif density == "medium":
                st.progress(60)
            else:
                st.progress(90)

        st.subheader("🧠 Neutrosophic Logic Decision")
        st.write(f"✅ Truth (Traffic Level): {t:.2f}")
        st.write(f"❓ Indeterminacy (Obstacle): {i:.2f}")
        st.write(f"❌ Falsity: {f:.2f}")
        st.success(f"🚦 Suggested Traffic Signal: {signal}")

        tab1, tab2 = st.tabs(["Detection", "Density Heatmap"])

        with tab1:
            if model == "YOLOv8":
                st.image(result_path, use_column_width=True)
            elif model == "YOLOv5":
                result_img = np.squeeze(result.render())
                st.image(result_img, channels="BGR", use_column_width=True)
            else:
                img_with_boxes = cv2.imread("input.jpg")
                vehicle_ids = ssd_vehicles if model == "SSD" else retinanet_vehicles
                for idx in range(len(box_list)):
                    if score_list[idx] > 0.5 and label_list[idx].item() in vehicle_ids:
                        x1, y1, x2, y2 = map(int, box_list[idx])
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), use_column_width=True)

        with tab2:
            st.image(cv2.cvtColor(density_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        if st.button("Clear Results"):
            if model == "YOLOv8":
                shutil.rmtree("runs", ignore_errors=True)
            os.remove("input.jpg")
            st.rerun()