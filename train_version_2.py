from ultralytics import YOLO
import os

def main():
    # 生产模型：使用更大的 Epoch 和早停机制
    model = YOLO('yolo11n.pt')

    # 生产级配置
    model.train(
        data='data.yaml',
        epochs=200,            # 增加 Epoch 确保收敛
        patience=50,           # 如果 50 轮不提升则早停
        imgsz=640,
        batch=16,
        device=0,
        project='Ski_Detection_Production',
        name='model2_final',
        optimizer='auto',      # 自动选择优化器
        seed=42,               # 固定随机种子以便复现
        save=True,
        exist_ok=True
    )
    print("Production training finished. Weights are in runs/detect/Ski_Detection_Production/model2_final/weights/best.pt")

if __name__ == '__main__':
    main()
