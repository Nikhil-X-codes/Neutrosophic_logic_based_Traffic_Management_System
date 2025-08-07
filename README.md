# 🚦 Neutrosophic-Based Traffic Management System

This project presents an intelligent AI-based traffic signal control system that uses **Neutrosophic Logic** and deep learning models to optimize traffic flow and ensure road safety. Developed using **Streamlit**, it enables users to upload road images, detect vehicle density and obstacles, and determine signal decisions (Red, Yellow, Green) using logic-based reasoning.

## 👨‍💻 Team

- [Tushar Gupta](https://github.com/gtushar8055) — Dataset, Graphs, Data Analysis  
- [Nitish Kumar Choubey](https://github.com/NitishChoubey) — Model Coding, Documentation  
- [Harshit Singhal](https://github.com/harshitsinghal226) — Tables, Flowcharts, Content 
- [Nikhil Nagar](https://github.com/Nikhil-X-codes) — Research Paper Analysis  
- [Devansh Bansal](https://github.com/devanshbansal16) — Documentation Content  
- **Institute:** IIIT Sonepat, Haryana, India

---

## 📌 Project Objective

To design a dynamic, adaptable traffic signal controller that:
- Detects vehicles and obstacles using real-time images
- Computes traffic density using heatmaps
- Applies **Neutrosophic Logic** to determine signal states

---

## 🧠 Core Technologies

- **Deep Learning Models:** YOLOv8, YOLOv5, SSD, RetinaNet
- **Frameworks & Tools:** PyTorch, OpenCV, Streamlit
- **Logic System:** Neutrosophic Logic (Truth, Indeterminacy, Falsity)

---

## ⚙️ Key Features

- **Vehicle Detection:** Cars, buses, trucks, motorcycles
- **Obstacle Detection:** Pedestrians, animals
- **Density Calculation:** Coverage metrics using bounding boxes
- **Signal Decision:** Based on T (truth), I (indeterminacy), F (falsity)
- **Model Flexibility:** Choose between 4 pretrained models
- **Interactive UI:** Upload images, view heatmaps, results & signal status

---

## 📊 Logic Interpretation

- **Truth (T):** Vehicle density (scaled 0–1)
- **Indeterminacy (I):** Obstacle presence
- **Falsity (F):** 1 - (T + I), indicating uncertainty

> Example Rule:
> - If T > 0.5 → Green Signal
> - If I > 0.6 → Red Signal
> - Else → Yellow Signal

---

## 📈 Model Performance Summary

| Model     | Best For                          | Notes                              |
|-----------|-----------------------------------|-------------------------------------|
| YOLOv8    | Fastest & most accurate           | Best for edge deployment            |
| YOLOv5    | Versatile & widely adopted        | Strong community support            |
| RetinaNet | High precision, small objects     | Ideal for surveillance intersections|
| SSD       | Lightweight, low-cost setups      | Good for Jetson/IoT devices         |

---

## 📉 Limitations

- Works on **static images**, not live video yet
- Model performance drops in poor lighting or extreme traffic
- Requires **manual analysis** per image
- Lacks **real-time feedback** or IoT signal control

---

## 🔮 Future Scope

- Integrate **live video feed** support
- Add **IoT-based traffic signal hardware**
- Enable **cloud deployment** for smart city scalability
- Improve **model training** with dynamic datasets

---

## 📚 References

1. [Traffic Management System Using YOLO](https://doi.org/10.3390/engproc2023059210)  
2. [YOLOv5 Performance Evaluation](https://doi.org/10.52756/ijerr.2024.v38.005)  
3. [Smart Traffic Systems](https://doi.org/10.5120/13123-0473)  
4. [Review of Intelligent Traffic Systems](https://doi.org/10.21608/njccs.2023.321169)

---
