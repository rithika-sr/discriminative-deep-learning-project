# Discriminative Deep Learning Project
**IE 7615 - Deep Learning and Neural Networks (Spring 2026)**

A complete deep learning system implementing single-object classification (Milestone 1) and multi-object detection and localization (Milestone 2) using CNNs, transfer learning, and YOLOv8.

---

## 👥 Team - 4 Members

- Rithika Sankar Rajeswari
- Sreevarshan Sathiyamurthy
- Meena Periasamy
- Sahil Mohanty

---

## 📋 Project Overview

This project demonstrates a complete discriminative deep learning pipeline in two phases:

**Milestone 1:** Single-object image classification achieving **98.94% accuracy**  
**Milestone 2:** Multi-object detection and localization achieving **93.52% mAP@0.5**

The system processes student-collected object images and achieves state-of-the-art performance using transfer learning and modern object detection techniques.

### 🏆 Key Achievements
- **Classification:** 98.94% accuracy (MobileNetV2)
- **Detection:** 93.52% mAP@0.5, 93.52% mAP@0.5:0.95 (YOLOv8s)
- **Dataset:** 4,108 total images across 35 object classes
- **Training Time:** ~3 hours total (both milestones)

---

# 🎯 MILESTONE 1: Single-Object Classification

## Results Summary
- **Best Model:** MobileNetV2 with **98.94% test accuracy**
- **Dataset:** 3,708 images across 35 unique object classes
- **Training Time:** 4.9 minutes on Apple Silicon M3
- **Report:** [Milestone 1 Report PDF](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/milestone1/report/Project%20Report%20-%20Milestone%201.pdf)

---

## 📊 Milestone 1 Dataset

### Statistics
- **Total Classes:** 35 (OBJ001 - OBJ787)
- **Total Images:** 3,708
- **Image Size:** 224×224 pixels (RGB)
- **Format:** JPEG
- **Split:** 80% train (2,963) / 10% val (366) / 10% test (379)

### Data Collection
Each student contributed 100+ images of one unique object with variations:
- Different lighting conditions (natural, artificial, shadows)
- Multiple viewing angles (front, side, top, rotated)
- Various backgrounds (indoor, outdoor)
- Different orientations and placements

### Preprocessing
- Stratified train/val/test splitting
- Data augmentation (rotation ±20°, flips, shifts ±20%, zoom ±20%)
- Pixel normalization to [0, 1]

---

## 🧠 Models Tested (Milestone 1)

| Model | Architecture | Test Acc | Val Acc | Params | Time |
|-------|--------------|----------|---------|--------|------|
| **MobileNetV2** ⭐ | Transfer Learning | **98.94%** | **99.18%** | 3.5M | 4.9 min |
| CustomCNN | Built from Scratch | 92.61% | 92.90% | 1.2M | 73.4 min |
| ResNet50 | Transfer Learning | 63.06% | 65.57% | 25M | 40.1 min |
| EfficientNetB0 | Transfer Learning | 3.69% | 3.55% | 5.3M | 4.2 min |

### MobileNetV2 Performance Details
```
Test Accuracy:    98.94%
Precision:        99.09%
Recall:           98.91%
F1-Score:         98.93%
Misclassifications: 4 out of 379 images
Perfect Classes:  33 out of 35 (100% accuracy)
```

---

# 🎯 MILESTONE 2: Multi-Object Detection & Localization

## Results Summary
- **Model:** YOLOv8s (custom trained)
- **mAP@0.5:** **93.52%** 🌟
- **mAP@0.5:0.95:** **93.52%** 🌟
- **Precision:** **96.19%**
- **Recall:** **87.21%**
- **Dataset:** 400 multi-object composite images
- **Training:** 73 epochs in 49 minutes
- **Detection Speed:** ~72ms per image

---

## 📊 Milestone 2 Dataset

### Statistics
- **Total Images:** 400 composite images
- **Image Size:** 640×640 pixels
- **Objects per Image:** 2-5 randomly selected
- **Total Annotations:** ~1,200 bounding boxes
- **Split:** 80% train (320) / 10% val (40) / 10% test (40)

### Image Generation Process
1. Random selection of 2-5 object classes per image
2. Smart grid-based placement ensuring clear object separation
3. Automatic YOLO format annotation generation
4. Normalized bounding boxes: `class_id x_center y_center width height`

### Data Augmentation (by YOLOv8)
- Mosaic augmentation
- Random flipping and rotation
- HSV color variation
- Random erasing

---

## 🤖 YOLOv8 Training Configuration

### Model Specifications
- **Architecture:** YOLOv8s (small variant)
- **Parameters:** 11.1M
- **Input Size:** 640×640 pixels
- **Pre-trained:** COCO dataset
- **Fine-tuned:** 35 custom classes

### Training Settings
```python
EPOCHS = 100 (early stopped at 73)
BATCH_SIZE = 16
OPTIMIZER = Adam (lr=0.001)
DEVICE = MPS (Apple Silicon GPU)
PATIENCE = 15
LOSS_WEIGHTS = box:7.5, cls:0.5, dfl:1.5
```

---

## 📈 Milestone 2: Performance Metrics

### Test Set Results
| Metric | Value | Description |
|--------|-------|-------------|
| **mAP@0.5** | **93.52%** | Mean Average Precision at IoU=0.5 |
| **mAP@0.5:0.95** | **93.52%** | COCO-style mAP across IoU thresholds |
| **Precision** | **96.19%** | Correct positive predictions |
| **Recall** | **87.21%** | Objects successfully detected |

### Training Progression
| Epoch | mAP@0.5 | mAP@0.5:0.95 | Box Loss | Cls Loss |
|-------|---------|--------------|----------|----------|
| 1 | 1.0% | 0.6% | 2.024 | 4.669 |
| 20 | 59.1% | 49.5% | 0.436 | 1.345 |
| 40 | 87.1% | 86.9% | 0.320 | 0.805 |
| **58** | **98.4%** | **98.4%** | **0.143** | **0.488** ← Best |
| 73 | 91.9% | 91.9% | 0.131 | 0.384 (stopped) |

### Per-Class Performance
- **Excellent (>95% mAP):** 29 out of 35 classes
- **Good (80-95% mAP):** 4 classes
- **Challenging (<80% mAP):** 2 classes (OBJ012: 39%, OBJ230: 0%)

---

## 🎨 Detection Pipeline

### Sample Detection Results
**Test on 10 images:**
- Total objects detected: 39
- Average objects per image: 3.9
- Average confidence: 82.86%

**Example Detections:**
```
Image 1 (4 objects):
  - OBJ031: 91.25%
  - OBJ006: 89.25%
  - OBJ004: 56.74%
  - OBJ021: 53.72%

Image 2 (5 objects):
  - OBJ009: 99.30%
  - OBJ028: 98.37%
  - OBJ108: 95.63%
  - OBJ021: 86.41%
  - OBJ019: 78.88%
```

---

## 🛠️ Project Structure

```
Discriminative_Project/
│
├── milestone1/                          # Single-Object Classification
│   ├── data/                           # 3,708 images (35 classes)
│   ├── results/                        # Training curves, confusion matrices
│   ├── report/                         # Milestone 1 Report PDF
│   ├── screenshots/                    # Training visualizations
│   └── *.keras                         # Trained models (98.94% best)
│
├── milestone2/                          # Multi-Object Detection
│   ├── data/
│   │   ├── images/                    # 400 composite images
│   │   ├── labels/                    # YOLO annotations
│   │   └── dataset.yaml               # YOLO config
│   ├── scripts/
│   │   ├── create_multi_object_dataset.py
│   │   ├── train_yolo.py
│   │   └── detect_objects.py
│   ├── models/
│   │   └── yolo_best.pt              # 93.52% mAP model
│   ├── results/                       # Detection visualizations
│   └── runs/                          # Training logs & metrics
│
└── README.md
```

---

# 🚀 Quick Start

## Installation

```bash
# Clone repository
git clone https://github.com/rithika-sr/discriminative-deep-learning-project.git
cd discriminative-deep-learning-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow ultralytics opencv-python torch numpy pandas matplotlib seaborn scikit-learn pillow
```

## Usage

### Milestone 1: Classification
```bash
python milestone1/03_train_models.py
```

### Milestone 2: Detection
```bash
# Generate multi-object dataset
python milestone2/scripts/create_multi_object_dataset.py

# Train YOLOv8
python milestone2/scripts/train_yolo.py

# Run detection
python milestone2/scripts/detect_objects.py
```

---

# 📊 Results Comparison

| Metric | Milestone 1 | Milestone 2 |
|--------|-------------|-------------|
| **Task** | Classification | Detection + Localization |
| **Dataset** | 3,708 images | 400 composite images |
| **Best Model** | MobileNetV2 | YOLOv8s |
| **Accuracy/mAP** | 98.94% | 93.52% |
| **Parameters** | 3.5M | 11.1M |
| **Training Time** | 4.9 min | 49 min |
| **Inference** | 5ms/image | 72ms/image |
| **Output** | Class label | Bboxes + labels |

---

# 🎯 Key Findings

## What Worked Well

### Transfer Learning
- Pre-trained weights (ImageNet, COCO) dramatically improved performance
- MobileNetV2: 98.94% vs CustomCNN: 92.61% (+6.33%)
- 15× faster training with transfer learning

### Model Architecture Selection
- Efficient models outperformed larger ones
- MobileNetV2 (3.5M params) > ResNet50 (25M params)
- YOLOv8s optimal for real-time detection

### Dataset Quality
- Proper train/val/test splitting prevented overfitting
- Data augmentation essential for generalization
- 400 images sufficient for robust YOLO training

## Challenges & Solutions

### Milestone 1
- **Challenge:** ResNet50 underperformed (63.06%)
- **Solution:** MobileNetV2 with frozen base proved optimal

### Milestone 2
- **Challenge:** Small objects harder to detect
- **Solution:** Multi-scale detection heads in YOLOv8
- **Challenge:** Class imbalance in test set
- **Solution:** Stratified sampling during generation

---

# 💻 Technical Implementation

## Milestone 1: MobileNetV2 Pipeline
```python
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, 'relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(35, 'softmax')
])
```

## Milestone 2: YOLOv8 Pipeline
```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='mps',
    patience=15
)
```

---

# 📈 Detailed Performance

## Milestone 1: Per-Model Metrics

| Model | Test Acc | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MobileNetV2 | 98.94% | 99.09% | 98.91% | 98.93% |
| CustomCNN | 92.61% | 93.19% | 92.43% | 92.33% |
| ResNet50 | 63.06% | 69.88% | 62.90% | 61.61% |
| EfficientNetB0 | 3.69% | 0.11% | 2.86% | 0.20% |

## Milestone 2: Training Evolution

**Loss Reduction:**
```
Epoch 1:  Total Loss = 8.5
Epoch 20: Total Loss = 2.8
Epoch 40: Total Loss = 2.0
Epoch 58: Total Loss = 1.5  (Best checkpoint)
Epoch 73: Total Loss = 1.4  (Early stopped)
```

**mAP Improvement:**
```
Start (Epoch 1):   1.0% → 0.6%
Mid (Epoch 40):   87.1% → 86.9%
Peak (Epoch 58):  98.4% → 98.4%  ✅ Best
Final (Epoch 73): 91.9% → 91.9%
```

---

# 📁 Key Outputs

## Milestone 1
- **Best Model:** `milestone1/MobileNetV2.keras` (98.94%)
- **Visualizations:** Training curves, confusion matrices
- **Report:** `milestone1/report/Project Report - Milestone 1.pdf`

## Milestone 2
- **Best Model:** `milestone2/models/yolo_best.pt` (93.52% mAP)
- **Dataset:** 400 images + annotations
- **Visualizations:** 10 detection results with bounding boxes
- **Metrics:** Training logs, confusion matrix, mAP curves

---

# 🔬 Technical Details

## Milestone 1: CNN Architecture

### Custom CNN
```
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
GlobalAvgPool → Dense(256) → Dropout(0.5) → Dense(35)
```

### Training Config
```python
IMAGE_SIZE = 224×224
BATCH_SIZE = 32
EPOCHS = 30
OPTIMIZER = Adam(lr=0.001)
CALLBACKS = EarlyStopping(patience=5), ReduceLROnPlateau(patience=3)
```

---

## Milestone 2: YOLO Implementation

### Multi-Object Dataset Generation
```python
def create_composite(num_objects):
    canvas = Image.new('RGB', (640, 640), (240, 240, 240))
    objects = random.sample(classes, num_objects)
    
    for obj_class in objects:
        img = load_random_object(obj_class)
        x, y = find_valid_position()
        canvas.paste(img, (x, y))
        
        bbox = calculate_normalized_bbox(x, y, w, h)
        save_annotation(class_id, bbox)
    
    return canvas
```

### YOLOv8 Architecture
- **Backbone:** CSPDarknet with C2f blocks
- **Neck:** PAN (Path Aggregation Network)
- **Head:** Decoupled detection head (3 scales)
- **Layers:** 130 total (73 fused)
- **GFLOPs:** 28.7

---

# 🎓 Lessons Learned

## Model Selection
✅ Transfer learning dramatically accelerates training and improves accuracy  
✅ Efficient architectures (MobileNetV2, YOLOv8s) outperform larger models  
✅ Early stopping prevents overfitting and saves compute time  

## Data Engineering
✅ Dataset size critically impacts performance (400 images optimal for YOLO)  
✅ Quality annotations essential for object detection  
✅ Proper object spacing improves detection learning  
✅ Data augmentation crucial for small datasets  

## Training Best Practices
✅ Learning rate scheduling improves convergence  
✅ Patience-based early stopping finds optimal models  
✅ Regular checkpointing prevents progress loss  
✅ Validation metrics guide model selection  

---

# 💻 Hardware & Environment

**Development Platform:**
- Device: MacBook Pro M5 chip
- RAM: 16GB+
- GPU: Apple Metal Performance Shaders (MPS)

**Software Stack:**
- Python 3.13
- TensorFlow 2.20.0 (Milestone 1)
- PyTorch 2.10.0 + Ultralytics 8.4.12 (Milestone 2)

---

# 📊 Performance Benchmarks

## Milestone 1: Classification
| Model | Accuracy | Params | Inference | Memory |
|-------|----------|--------|-----------|--------|
| MobileNetV2 | 98.94% | 3.5M | 5ms | 14MB |
| CustomCNN | 92.61% | 1.2M | 3ms | 5MB |
| ResNet50 | 63.06% | 25M | 15ms | 98MB |

## Milestone 2: Detection
| Metric | Value | Performance |
|--------|-------|-------------|
| mAP@0.5 | 93.52% | Excellent |
| mAP@0.5:0.95 | 93.52% | Outstanding |
| Precision | 96.19% | Very High |
| Recall | 87.21% | Strong |
| Inference | 72ms | Real-time capable |
| FPS | ~14 | Good for applications |

---

# 🔮 Future Enhancements

## Short-term
1. Increase dataset to 1000+ multi-object images
2. Fine-tune YOLOv8 deeper layers
3. Implement model ensemble for higher accuracy
4. Add instance segmentation (pixel-level masks)

## Long-term
1. Real-time video object detection (30+ FPS)
2. Object tracking across video frames
3. Mobile/edge device deployment
4. 3D bounding box estimation
5. Few-shot learning for new classes

---

# 📄 Documentation

## Reports
- **Milestone 1:** [Project Report - Milestone 1.pdf](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/milestone1/report/Project%20Report%20-%20Milestone%201.pdf)
- **Milestone 2:** Coming soon

## Code Documentation
All scripts include comprehensive docstrings and comments explaining:
- Function parameters and return values
- Algorithm logic and design decisions
- Configuration options

---


**⭐ Star this repository if you found it helpful!**

**🔗 [View Milestone 1 Report](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/milestone1/report/Project%20Report%20-%20Milestone%201.pdf) | [View Detection Results](https://github.com/rithika-sr/discriminative-deep-learning-project/tree/main/milestone2/results)**
