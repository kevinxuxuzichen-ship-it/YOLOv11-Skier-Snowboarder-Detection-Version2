from ultralytics import YOLO
import os

def main():
    # Production-grade model: Use more training epochs and early stopping mechanism
    model = YOLO('yolo11n.pt')

    # Production-level training configuration
    model.train(
        data='data.yaml',
        epochs=200,            # Increase epochs to guarantee full convergence
        patience=50,           # Trigger early stop if no performance improvement after 50 epochs
        imgsz=640,
        batch=16,
        device=0,
        project='Ski_Detection_Production',
        name='model2_final',
        optimizer='auto',      # Automatically select the optimizer
        seed=42,               # Fix random seed for reproducible training results
        save=True,
        exist_ok=True
    )
    print("Production training finished. Weights are in runs/detect/Ski_Detection_Production/model2_final/weights/best.pt")

if __name__ == '__main__':
    main()
