from ultralytics import YOLO

# This will download yolov8n.pt the first time you run it
model = YOLO('yolov8n.pt')  # 'n' stands for Nano (light and fast)

# Run inference
results = model('D:\WhatsApp Image 2025-04-19 at 17.55.24_86f8d66d.jpg', save=True)

# This will save the results image in the 'runs/detect/predict' folder
