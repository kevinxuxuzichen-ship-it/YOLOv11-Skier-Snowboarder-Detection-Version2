# YOLOv11-Skier-Snowboarder-Detection-Version2

Advanced Skier & Snowboarder detection using YOLO11. Features 5-Fold Cross-Validation and K+1 production training for reliable real-world performance.

## 📊 Training Evolution Report

### Metrics Comparison

| Phase | Validation Method | mAP50 |
| :--- | :--- | :--- |
| **Phase 1** (Version 1 Baseline) | Small subset validation | **0.9510** |
| **Phase 2** (K-Fold Avg) | 5-Fold Cross-Validation | **0.8292** |
| **Phase 2** (K+1 Production) | Strictly held-out test set | **0.8010** |

### Why Phase 2 is better?

Phase 2 uses **Cross-Validation** to ensure that the model doesn't rely on "lucky" data splits. Although the numerical value is lower than the Phase 1 baseline, the final **80.1% mAP** is a realistic representation of real-world performance on unseen data. This version is more robust, less prone to overfitting, and production-ready.

## 🛠 Features
- **Algorithm**: YOLOv11 Nano
- **Robustness**: Validated across 5 different data folds.
- **Deployment**: Includes exported `.onnx` weights for high-performance inference.

## 📂 Repository Structure
- `Analysis/`: Contains confusion matrices and PR curves for each fold.
- `Models/`: Optimized `.pt` and `.onnx` weight files.
- `data.yaml`: Dataset configuration for skiers and snowboarders.
