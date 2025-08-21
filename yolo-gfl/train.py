import torch
from ultralytics import YOLO


def train_yolo(name: str):
    model = YOLO('yolo-gfl.yaml')

    device = 0 if torch.cuda.is_available() else 'cpu'
    data = r"../dataset/HOME-FIRE/data.yaml"

    # Train the model
    train_results = model.train(
        data=data,
        epochs=300,
        imgsz=640,
        batch=32,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=100,
        device=device,
        pretrained=False,
        optimizer='SGD',
        project='runs',
        name=name,
    )

    # Export the model to ONNX format
    path = model.export(format="onnx")

if __name__ == "__main__":
    train_yolo('YOLO-GFL')
