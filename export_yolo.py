import os
import sys

def main():
    print("Exporting YOLOv8n to ONNX for Frigate NVR...")
    from ultralytics import YOLO
    
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Export to ONNX with 320x320 input size
    output_path = model.export(
        format="onnx",
        imgsz=320,
        opset=12,
        dynamic=False,
        simplify=True
    )
    print(f"Exported model successfully to: {output_path}")
    
    # Move to config/model_cache/yolov8n.onnx
    os.makedirs("config/model_cache", exist_ok=True)
    target = os.path.join("config", "model_cache", "yolov8n.onnx")
    if os.path.exists(target):
        os.remove(target)
    os.replace(output_path, target)
    print(f"Moved model to {target}")

if __name__ == "__main__":
    main()
