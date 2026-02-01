# **IE 7615 - Deep Learning and Neural Networks (Spring 2026) - Object Classification| Milestone 1**

---

A deep learning project for automated single-object image classification using CNNs and transfer learning on a 35-class student-collected dataset.

---

## 👥 Team - 3 

- Rithika Sankar Rajeswari
- Sreevarshan Sathiyamurthy
- Meena Periasamy
- Sahil Mohanty

---

## 📋 Project Overview

This project implements and compares multiple CNN architectures for multi-class object classification. The goal is to build a discriminative model capable of accurately identifying individual objects from images, serving as the foundation for a multi-object detection system in Milestone 2.

### Key Results
- **Best Model:** MobileNetV2 with **98.94% test accuracy**
- **Dataset:** 3,708 images across 35 unique object classes
- **Training Time:** Under 5 minutes on Apple Silicon M3

---

## 📊 Dataset

### Dataset Statistics
- **Total Classes:** 35 (OBJ001 - OBJ787)
- **Total Images:** 3,708
- **Image Size:** 224×224 pixels (RGB)
- **Format:** JPEG

### Data Split
- **Training:** 2,963 images (80%)
- **Validation:** 366 images (10%)
- **Test:** 379 images (10%)

### Data Collection
All students contributed images of one unique object, captured under various conditions:
- Different lighting (natural, artificial, shadows)
- Multiple viewing angles (front, side, top)
- Various backgrounds (indoor, outdoor)
- Different orientations and positions

### Data Augmentation (Training Only)
- Random rotation: ±20°
- Horizontal flipping
- Width/height shifts: ±20%
- Zoom range: ±20%
- Pixel normalization: [0, 1]

---

## 🧠 Models Tested

We implemented and evaluated four CNN architectures:

### 1. Custom CNN (Baseline)
- Built from scratch with 3 convolutional blocks
- Architecture: 32→64→128 filters
- Batch normalization + dropout regularization
- **Parameters:** ~1.2M
- **Test Accuracy:** 92.61%

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Frozen base + custom classification head
- **Parameters:** ~25M
- **Test Accuracy:** 63.06%

### 3. MobileNetV2 (Transfer Learning) ⭐ **BEST**
- Lightweight, efficient architecture
- Pre-trained on ImageNet
- **Parameters:** ~3.5M
- **Test Accuracy:** 98.94%

### 4. EfficientNetB0 (Transfer Learning)
- Compound scaling approach
- Pre-trained on ImageNet
- **Parameters:** ~5.3M
- **Test Accuracy:** 3.69% (failed to converge)

---

## 📈 Results Summary

| Model | Test Acc | Val Acc | Params | Training Time | Status |
|-------|----------|---------|--------|---------------|--------|
| **MobileNetV2** | **98.94%** | **99.18%** | 3.5M | 4.9 min | ✅ Best |
| CustomCNN | 92.61% | 92.90% | 1.2M | 73.4 min | ✅ Good |
| ResNet50 | 63.06% | 65.57% | 25M | 40.1 min | ⚠️ Underperformed |
| EfficientNetB0 | 3.69% | 3.55% | 5.3M | 4.2 min | ❌ Failed |

---

## 🛠️ Project Structure

```
Discriminative_Project/
├── data/
│   └── processed/
│       ├── train/          # 2,963 training images
│       ├── val/            # 366 validation images
│       └── test/           # 379 test images
├── models/
│   ├── CustomCNN.keras
│   ├── ResNet50.keras
│   ├── MobileNetV2.keras   # Best model
│   └── EfficientNetB0.keras
├── scripts/
│   ├── 01_data_inspection.py      # Dataset analysis
│   ├── 02_split_data.py           # Train/val/test split
│   ├── 03_train_models.py         # Model training pipeline
│   └── 04_generate_report.py     # Report generation
├── results/
│   ├── model_comparison.json
│   ├── class_labels.json
│   ├── *_training_history.png
│   ├── *_confusion_matrix.png
│   └── *_classification_report.json
├── report/
│   └── Milestone1_Report.md
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
TensorFlow 2.20.0+
NumPy, Pandas, Matplotlib, Seaborn, Pillow, scikit-learn
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rithika-sr/discriminative-deep-learning-project.git
cd discriminative-deep-learning-project
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

### Usage

**1. Inspect Dataset:**
```bash
python scripts/01_data_inspection.py
```

**2. Split Data into Train/Val/Test:**
```bash
python scripts/02_split_data.py
```

**3. Train All Models:**
```bash
python scripts/03_train_models.py
```
*Note: Training all 4 models takes approximately 2 hours*

**4. Generate Report:**
```bash
python scripts/04_generate_report.py
```

---

## 📊 Training Configuration

```python
# Common settings for all models
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
OPTIMIZER = Adam(lr=0.001)
LOSS = 'categorical_crossentropy'

# Callbacks
- EarlyStopping(patience=5)
- ReduceLROnPlateau(factor=0.5, patience=3)
```

---

## 🎯 Key Findings

### ✅ What Worked
1. **MobileNetV2 Transfer Learning**
   - Best accuracy (98.94%) with minimal parameters
   - Fast convergence and excellent generalization
   - Only 4 misclassifications out of 379 test images

2. **Data Augmentation**
   - Critical for preventing overfitting
   - Improved Custom CNN from ~85% to 92.61%

3. **Transfer Learning**
   - Pre-trained ImageNet weights accelerated training
   - MobileNetV2 adapted well to object classification

### ⚠️ Challenges
1. **ResNet50 Underperformance**
   - Despite 25M parameters, achieved only 63.06%
   - Frozen base layers may not have adapted well
   - High validation variance suggests instability

2. **EfficientNetB0 Convergence Failure**
   - Model predicted only one class (3.69% accuracy)
   - Incompatibility with current dataset
   - Would require architecture modifications

### 📚 Lessons Learned
- Larger models don't always perform better
- Efficient architectures (MobileNetV2) can outperform heavier models
- Proper data augmentation is essential
- Early stopping prevents overfitting and saves training time

---

## 🔬 Detailed Performance Analysis

### MobileNetV2 (Best Model)
```
Test Accuracy:    98.94%
Precision:        99.09%
Recall:           98.91%
F1-Score:         98.93%
Training Time:    4.9 minutes
```

**Per-Class Performance:**
- 33 out of 35 classes: 100% accuracy
- OBJ009: 90% recall (1 error)
- OBJ095: 81.8% recall (2 errors)
- OBJ208: 90% recall (1 error)

**Confusion Matrix:** Nearly perfect diagonal alignment with minimal off-diagonal errors

---

## 📁 Output Files

### Models
- `models/MobileNetV2.keras` - Best performing model (recommended for deployment)
- `models/CustomCNN.keras` - Strong baseline model
- `models/ResNet50.keras` - Underperforming transfer learning model
- `models/EfficientNetB0.keras` - Failed convergence

### Results
- `results/model_comparison.json` - Complete performance metrics
- `results/class_labels.json` - Class ID to label mapping
- `results/*_training_history.png` - Training/validation curves
- `results/*_confusion_matrix.png` - Confusion matrices
- `results/*_classification_report.json` - Detailed metrics per class

---

## 🔮 Next Steps (Milestone 2)

1. **Multi-Object Image Generation**
   - Create composite images with 2-5 objects
   - Generate bounding box annotations

2. **Object Detection Implementation**
   - Implement YOLOv8 for object detection
   - Fine-tune for multi-object localization

3. **Pipeline Integration**
   - Combine YOLO detection + MobileNetV2 classification
   - End-to-end multi-object recognition system

4. **Performance Optimization**
   - Model quantization for deployment
   - Inference speed optimization
   - Mobile/edge device compatibility

---

## 💻 Hardware & Environment

- **Device:** MacBook Pro M3 chip
- **Framework:** TensorFlow 2.20.0
- **Python:** 3.13
- **Training:** CPU (Apple Silicon optimized)

---

## 📄 Report & Documentation

- **Full Report:** [Milestone1_Report.pdf](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/report/Project%20Report%20-%20Milestone%201.pdf)
- **Markdown Report:** [Milestone1_Report.md](report/Milestone1_Report.md)
- **Results Visualizations:** [results/](results/)

