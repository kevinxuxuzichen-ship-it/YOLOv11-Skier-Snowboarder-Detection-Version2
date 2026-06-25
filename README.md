# YOLOv11-Skier-Snowboarder-Detection-Version2

Advanced Skier & Snowboarder detection using YOLO11. Features 5-Fold Cross-Validation and K+1 production training for reliable real-world performance.

## 📊 Training Evolution Report


### **Context & Key Decisions**
*   **Training Strategy**:
    *   **Phase 1 (Baseline)**: Established a benchmark with ~95.1% mAP50 using a fixed 70/20/10 split, as shown in YOLOv11-Skier-Snowboarder-Detection (version 1).
    *   **Dataset Setup for Phase 2**: All original data (Train/Val/Test) was merged into a single pool of 213 images to allow for random, unbiased splitting during advanced validation.
    *   **Phase 2 (K-Fold)**: Conducted 5-Fold Cross-Validation with an average mAP50 of **0.8292** to ensure statistical stability.
    *   **Phase 2 (K+1 Production)**: Developed a final production model using 90% of data, achieving **0.8010 mAP50** on a 10% hold-out test set.
*   **Hardware**: Tesla T4 GPU support for extended training.
*   **Optimization**: Early Stopping (patience=50) triggered at epoch 151 for the production model.


### Scripts & Strategy
- `train.py`: Phase 1 setup (100 Epochs, 70% Train split).
- `train_version_2.py`: Phase 2 setup (200 Epochs + Early Stopping, 90% Train split).


### Metrics Comparison

| Phase | Validation Method | mAP50 |
| :--- | :--- | :--- |
| **Phase 1** (Version 1 Baseline) | Small subset validation | **0.9510** |
| **Phase 2** (K-Fold Avg) | 5-Fold Cross-Validation | **0.8292** |
| **Phase 2** (K+1 Production) | Strictly held-out test set | **0.8010** |


### Why is Phase 2 better?
1. **Higher Data Utilization**: Phase 2 leverages a larger portion of the dataset (90% in K+1 vs 70% in Phase 1), allowing the model to learn from more diverse samples.
2. **Extended Training (200 Epochs)**: Increasing the epoch limit to 200 ensures the model has sufficient time to converge and find the global minimum loss, while Early Stopping prevents late-stage overfitting.
3. **Statistical Robustness**: Phase 2 uses Cross-Validation to eliminate 'lucky' data splits. While the numerical value appears lower, the 80.1% mAP is a **more honest and reliable** representation of real-world performance on unseen data.
4. **Production Ready**: Optimized with professional training logic and exported to multiple formats for deployment.

## 🛠 Features
- **Algorithm**: YOLOv11 Nano
- **Robustness**: Validated across 5 different data folds.
- **Deployment**: Includes exported `.onnx` weights for high-performance inference.

## 📂 Repository Structure
- `results/` : Contains comprehensive training logs and visualization charts (e.g., confusion matrices, PR curves).
- `weights/` : Stores the optimized model weights (`.pt` and `.onnx`).
- `data.yaml` : Dataset configuration for skiers and snowboarders.
- `train_version_2.py` : Model training script.
