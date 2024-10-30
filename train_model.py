from ultralytics import YOLO

# Load a pretrained YOLO model for classification
model = YOLO("yolo11n-cls.yaml")

# Train the model on Fashion-MNIST
results = model.train(data="fashion-mnist", epochs=20, imgsz=28)

# After training, your best model will be saved automatically in the `runs/train/exp` folder.
# Note the path for later use, typically something like:
# "runs/train/exp/weights/best.pt"
