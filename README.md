# Discriminative Deep Learning Project
**IE 7615 - Deep Learning and Neural Networks (Spring 2026)**

A complete deep learning project implementing single-object classification (Milestone 1) and multi-object detection and localization (Milestone 2) using CNNs, transfer learning, and YOLOv8.

---

## 👥 Team - 4 Members

- Rithika Sankar Rajeswari
- Sreevarshan Sathiyamurthy
- Meena Periasamy
- Sahil Mohanty

---

## 📋 Project Overview

This project builds a complete discriminative deep learning system in two phases:

**Milestone 1:** Single-object image classification using **CNN architectures**  
**Milestone 2:** Multi-object detection and localization using **YOLOv8**

The system processes student-collected object images and demonstrates state-of-the-art performance using transfer learning and modern object detection techniques.

---

# 🎯 MILESTONE 1: Single-Object Classification

## Key Results
- **Best Model:** MobileNetV2 with **98.94% test accuracy**
- **Dataset:** 3,708 images across 35 unique object classes
- **Training Time:** Under 5 minutes on Apple Silicon M3
- **GitHub:** [Milestone 1 Report PDF](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/milestone1/report/Project%20Report%20-%20Milestone%201.pdf)

---

## 📊 Milestone 1 Dataset

### Dataset Statistics
- **Total Classes:** 35 (OBJ001 - OBJ787)
- **Total Images:** 3,708
- **Image Size:** 224×224 pixels (RGB)
- **Format:** JPEG
- **Source:** Student-collected images from shared Google Drive

### Data Split
- **Training:** 2,963 images (80%)
- **Validation:** 366 images (10%)
- **Test:** 379 images (10%)

### Data Collection Methodology
Each student contributed 100+ images of one unique object with variations:
- Different lighting conditions (natural, artificial, shadows)
- Multiple viewing angles (front, side, top, rotated)
- Various backgrounds (indoor, outdoor, different surfaces)
- Different orientations and placements

### Data Preprocessing
- **Stratified splitting** to maintain class distribution
- **Data augmentation** (training only):
  - Random rotation: ±20°
  - Horizontal flipping
  - Width/height shifts: ±20%
  - Zoom range: ±20%
- **Normalization:** Pixel values scaled to [0, 1]

---

## 🧠 Milestone 1: Models Tested

We implemented and compared four CNN architectures:

### 1. Custom CNN (Baseline)
- Built from scratch with 3 convolutional blocks
- Architecture: Conv(32)→Conv(64)→Conv(128)
- Batch normalization + dropout + global average pooling
- **Parameters:** ~1.2M
- **Test Accuracy:** 92.61%
- **Training Time:** 73.4 minutes

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Frozen base + custom classification head
- **Parameters:** ~25M
- **Test Accuracy:** 63.06%
- **Training Time:** 40.1 minutes

### 3. MobileNetV2 (Transfer Learning) ⭐ **BEST**
- Lightweight, efficient architecture
- Pre-trained on ImageNet, frozen base
- Custom dense layers for 35-class classification
- **Parameters:** ~3.5M
- **Test Accuracy:** 98.94%
- **Validation Accuracy:** 99.18%
- **Training Time:** 4.9 minutes
- **Epochs:** 13 (early stopping)

### 4. EfficientNetB0 (Transfer Learning)
- Compound scaling approach
- Pre-trained on ImageNet
- **Parameters:** ~5.3M
- **Test Accuracy:** 3.69% (failed to converge)
- **Training Time:** 4.2 minutes

---

## 📈 Milestone 1: Results Summary

| Model | Test Acc | Val Acc | Precision | Recall | F1-Score | Params | Time |
|-------|----------|---------|-----------|--------|----------|--------|------|
| **MobileNetV2** | **98.94%** | **99.18%** | **99.09%** | **98.91%** | **98.93%** | 3.5M | 4.9 min |
| CustomCNN | 92.61% | 92.90% | 93.19% | 92.43% | 92.33% | 1.2M | 73.4 min |
| ResNet50 | 63.06% | 65.57% | 69.88% | 62.90% | 61.61% | 25M | 40.1 min |
| EfficientNetB0 | 3.69% | 3.55% | 0.11% | 2.86% | 0.20% | 5.3M | 4.2 min |

### Key Observations
- ✅ MobileNetV2 achieved near-perfect classification
- ✅ Only 4 misclassifications out of 379 test images
- ✅ 33 out of 35 classes achieved 100% accuracy
- ⚠️ ResNet50 underperformed despite larger size
- ❌ EfficientNetB0 failed to converge (dataset incompatibility)

---

# 🎯 MILESTONE 2: Multi-Object Detection & Localization

## Key Results
- **Model:** YOLOv8s (custom trained)
- **mAP@0.5:** **92.94%** (Excellent!)
- **mAP@0.5:0.95:** **88.09%** (Outstanding!)
- **Precision:** **87.39%**
- **Recall:** **81.70%**
- **Dataset:** 400 multi-object composite images
- **Training Time:** 33 minutes (49 epochs with early stopping)
- **Detection Speed:** ~70ms per image on M3

---

## 📊 Milestone 2 Dataset

### Multi-Object Dataset Statistics
- **Total Images:** 400 composite images
- **Image Size:** 640×640 pixels (YOLO standard)
- **Objects per Image:** 2-5 randomly selected
- **Total Object Instances:** ~1,200 annotated bounding boxes
- **Format:** JPEG images + YOLO .txt annotations

### Data Split
- **Training:** 320 images (80%)
- **Validation:** 40 images (10%)
- **Test:** 40 images (10%)

### Multi-Object Image Generation Process
1. **Random Object Selection:** 2-5 classes randomly chosen per composite
2. **Object Placement:** Grid-based algorithm with controlled overlap (<30%)
3. **Background:** Light gray canvas (240, 240, 240 RGB)
4. **Object Resizing:** Maintain aspect ratio, max 200px per dimension
5. **YOLO Annotation:** Automatic bounding box generation
   - Format: `class_id x_center y_center width height` (all normalized to [0,1])

### Data Augmentation (Applied by YOLOv8 during training)
- Mosaic augmentation (combines 4 images)
- Random horizontal flipping (50% probability)
- HSV color augmentation
- Random erasing (40% probability)
- Image scaling and translation

---

## 🤖 Milestone 2: YOLOv8 Architecture & Training

### Model Specifications
- **Architecture:** YOLOv8s (small variant)
- **Input Resolution:** 640×640 pixels
- **Total Parameters:** 11.1M
- **GFLOPs:** 28.7
- **Network Depth:** 130 layers (73 after fusion)
- **Detection Heads:** 3 scales (small, medium, large objects)

### Training Configuration
```python
MODEL = 'yolov8s.pt'
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
OPTIMIZER = 'Adam'
LEARNING_RATE = 0.001
LR_FINAL = 0.01
DEVICE = 'mps'  # Apple Silicon GPU
PATIENCE = 15   # Early stopping
WORKERS = 4

# Loss weights
box_loss_gain = 7.5
cls_loss_gain = 0.5
dfl_loss_gain = 1.5
```

### Transfer Learning Strategy
- **Pre-trained on:** COCO dataset (80 classes)
- **Transferred:** 349/355 weights
- **Fine-tuned for:** 35 custom object classes
- **Frozen Layers:** DFL conv layer only
- **Trainable Parameters:** 11.1M

---

## 📈 Milestone 2: Training Results

### Training Progression

| Epoch | Box Loss | Cls Loss | mAP@0.5 | mAP@0.5:0.95 |
|-------|----------|----------|---------|--------------|
| 1 | 2.024 | 4.669 | 1.01% | 0.64% |
| 10 | 0.573 | 1.780 | 47.3% | 40.8% |
| 20 | 0.436 | 1.345 | 59.1% | 49.5% |
| 30 | 0.374 | 0.968 | 81.9% | 78.2% |
| **34** | **0.338** | **0.850** | **91.2%** | **87.2%** ← Best |
| 40 | 0.320 | 0.805 | 86.2% | 83.6% |
| 49 | 0.299 | 0.684 | 85.6% | 76.7% (stopped) |

**Early Stopping:** Training stopped at epoch 49 (best at epoch 34)

### Final Test Set Performance
```
mAP@0.5:          92.94%
mAP@0.5:0.95:     88.09%
Precision:        87.39%
Recall:           81.70%
Inference Speed:  ~125ms/image (8 FPS)
```

---

## 🎨 Milestone 2: Detection Pipeline Results

### Detection Capabilities
- ✅ **Multi-object detection:** Handles 2-6 objects per image
- ✅ **Object localization:** Accurate bounding box coordinates
- ✅ **Class identification:** 35-class recognition
- ✅ **Confidence scores:** Per-detection probability estimates

### Test Detection Performance (10 sample images)
- **Total objects detected:** 39
- **Average objects per image:** 3.9
- **Average confidence:** 82.86%
- **Detection accuracy:** 92.94% mAP@0.5

### Sample Detection Examples

**Image 1: 4 objects detected**
```
OBJ031 (Red Apple):    91.25% confidence  ✅
OBJ006 (Earphones):    89.25% confidence  ✅
OBJ004 (Book):         56.74% confidence  ✅
OBJ021:                53.72% confidence  ✅
```

**Image 2: 5 objects detected**
```
OBJ009:  99.30% confidence  ✅
OBJ028:  98.37% confidence  ✅
OBJ108:  95.63% confidence  ✅
OBJ021:  86.41% confidence  ✅
OBJ019:  78.88% confidence  ✅
```

**Image 3: 6 objects detected (maximum)**
```
OBJ022:  99.27% confidence  ✅
OBJ159:  96.12% confidence  ✅
OBJ016:  91.01% confidence  ✅
OBJ002:  79.66% confidence  ✅
OBJ029:  72.55% confidence  ✅
OBJ003:  28.80% confidence  ⚠️
```

---

## 📊 Performance Comparison

### Dataset Impact Analysis
| Configuration | Images | mAP@0.5 | mAP@0.5:0.95 | Training Time |
|---------------|--------|---------|--------------|---------------|
| Initial (110 img) | 110 | 70.2% | 65.4% | 8 min |
| **Final (400 img)** | **400** | **92.94%** | **88.09%** | **33 min** |
| **Improvement** | **+264%** | **+22.7%** | **+22.7%** | **+25 min** |

### Milestone Comparison
| Aspect | Milestone 1 | Milestone 2 |
|--------|-------------|-------------|
| **Task** | Single-object classification | Multi-object detection |
| **Dataset Size** | 3,708 images | 400 composite images |
| **Image Size** | 224×224 | 640×640 |
| **Classes** | 35 | 35 |
| **Best Metric** | 98.94% accuracy | 92.94% mAP@0.5 |
| **Model Type** | MobileNetV2 (CNN) | YOLOv8s (Detector) |
| **Parameters** | 3.5M | 11.1M |
| **Training Time** | 4.9 minutes | 33 minutes |
| **Inference Speed** | 5ms/image | 125ms/image |
| **Output** | Class label | Bounding boxes + IDs |

---

## 🛠️ Complete Project Structure

```
Discriminative_Project/
│
├── milestone1/                          # MILESTONE 1: Classification ✅
│   ├── data/
│   │   ├── train/                      # 2,963 single-object images
│   │   ├── val/                        # 366 validation images
│   │   └── test/                       # 379 test images
│   ├── results/
│   │   ├── model_comparison.json
│   │   ├── class_labels.json
│   │   ├── MobileNetV2_training_history.png
│   │   ├── MobileNetV2_confusion_matrix.png
│   │   └── *_classification_report.json
│   ├── report/
│   │   └── Project Report - Milestone 1.pdf
│   ├── screenshots/                    # 13 training screenshots
│   ├── MobileNetV2.keras              # Best model (98.94%)
│   ├── CustomCNN.keras
│   ├── ResNet50.keras
│   └── EfficientNetB0.keras
│
├── milestone2/                          # MILESTONE 2: Detection ✅
│   ├── data/
│   │   ├── images/
│   │   │   ├── train/                 # 320 composite images
│   │   │   ├── val/                   # 40 composite images
│   │   │   └── test/                  # 40 composite images
│   │   ├── labels/
│   │   │   ├── train/                 # 320 YOLO .txt annotations
│   │   │   ├── val/                   # 40 YOLO annotations
│   │   │   └── test/                  # 40 YOLO annotations
│   │   └── dataset.yaml               # YOLO config file
│   ├── scripts/
│   │   ├── create_multi_object_dataset.py
│   │   ├── train_yolo.py
│   │   └── detect_objects.py
│   ├── models/
│   │   └── yolo_best.pt              # Trained YOLOv8 (92.94% mAP)
│   ├── results/
│   │   ├── detection_01_multi_test_0023.jpg
│   │   └── ... (10 detection visualizations)
│   ├── runs/
│   │   ├── yolo_400images/           # Training logs
│   │   │   ├── weights/best.pt
│   │   │   ├── confusion_matrix.png
│   │   │   ├── results.csv
│   │   │   └── training curves
│   │   └── detect/val/               # Validation results
│   └── yolov8s.pt                    # Pre-trained weights
│
├── README.md                            # This file
├── .gitignore
└── venv/                                # Virtual environment
```

---

# 🚀 Getting Started

## Prerequisites
```bash
Python 3.9+
TensorFlow 2.20.0+
PyTorch 2.10.0+
Ultralytics YOLOv8 8.4.12
NumPy, Pandas, Matplotlib, Seaborn, Pillow, scikit-learn, OpenCV
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/rithika-sr/discriminative-deep-learning-project.git
cd discriminative-deep-learning-project
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Milestone 1 dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow

# Milestone 2 dependencies (additional)
pip install ultralytics opencv-python torch torchvision
```

---

# 📖 Usage Guide

## Milestone 1: Running Classification Pipeline

### Inspect Dataset
```bash
python milestone1/01_data_inspection.py
```
**Output:** Dataset statistics, class distribution, image format verification

### Split Dataset
```bash
python milestone1/02_split_data.py
```
**Output:** Train/val/test split (2,963/366/379 images)

### Train Models
```bash
python milestone1/03_train_models.py
```
**Time:** ~2 hours for all 4 models  
**Output:** Trained models, training curves, confusion matrices, metrics

### Generate Report
```bash
python milestone1/04_generate_report.py
```
**Output:** Markdown report with comprehensive analysis

---

## Milestone 2: Running Detection Pipeline

### Generate Multi-Object Dataset
```bash
python milestone2/scripts/create_multi_object_dataset.py
```
**Time:** ~10-15 minutes  
**Output:** 400 composite images + YOLO annotations

### Train YOLOv8
```bash
python milestone2/scripts/train_yolo.py
```
**Time:** ~30-60 minutes (depends on epochs)  
**Output:** Trained YOLO model, training logs, performance metrics

### Run Object Detection
```bash
python milestone2/scripts/detect_objects.py
```
**Time:** ~1-2 minutes for 10 images  
**Output:** Detection visualizations with bounding boxes and labels

---

# 🎯 Key Findings & Analysis

## Milestone 1: Classification Insights

### ✅ What Worked
1. **Transfer Learning Superiority**
   - MobileNetV2: 98.94% vs CustomCNN: 92.61%
   - Pre-trained ImageNet weights accelerated convergence
   - 15× faster training (4.9 min vs 73.4 min)

2. **Model Efficiency**
   - Smaller models outperformed larger ones
   - MobileNetV2 (3.5M) > ResNet50 (25M)
   - Depthwise separable convolutions proved optimal

3. **Data Augmentation Impact**
   - Essential for preventing overfitting
   - Enabled strong generalization across variations
   - No overfitting observed (train ≈ val accuracy)

### ⚠️ Challenges
1. **ResNet50 Underperformance**
   - 63.06% accuracy despite 25M parameters
   - Frozen architecture prevented optimal adaptation
   - High validation variance (±10%)

2. **EfficientNetB0 Failure**
   - Complete convergence failure (3.69%)
   - Predicted single class for all inputs
   - Architecture-dataset incompatibility

### MobileNetV2 Error Analysis
**4 total misclassifications:**
- OBJ009: 1 error (challenging viewing angle)
- OBJ095: 2 errors (visual similarity to other objects)
- OBJ208: 1 error (ambiguous lighting conditions)

---

## Milestone 2: Detection Insights

### ✅ What Worked
1. **Dataset Size Impact**
   - 400 images: 92.94% mAP vs 110 images: 70.2% mAP
   - **+22.74% improvement** from larger dataset
   - More examples = better object localization learning

2. **YOLOv8 Transfer Learning**
   - COCO pre-trained weights adapted excellently
   - Achieved high mAP with only 320 training images
   - Early stopping at epoch 49 (optimal convergence)

3. **Detection Quality**
   - **82.86% average confidence** on test detections
   - Accurate bounding box localization (IoU ~0.75)
   - Handles multi-scale objects effectively

4. **Training Efficiency**
   - Early stopping saved 51% training time
   - Automatic checkpoint saving every 10 epochs
   - Best model selection based on validation mAP

### ⚠️ Challenges
1. **Small Object Detection**
   - Objects <50px show reduced confidence (30-60%)
   - YOLO struggles with very small instances

2. **Class Imbalance**
   - Some classes only 1-2 instances in test set
   - OBJ012, OBJ108 showed lower performance (<50% mAP)

3. **Overlapping Objects**
   - Heavy overlap (>50%) reduced detection recall
   - NMS (Non-Maximum Suppression) occasionally removes valid detections

### Per-Class Analysis

**Top Performers (99.5% mAP):**
- OBJ001, OBJ002, OBJ003, OBJ006, OBJ007, OBJ009, OBJ010
- OBJ019, OBJ021, OBJ022, OBJ027, OBJ028, OBJ029, OBJ031
- OBJ061, OBJ090, OBJ107, OBJ159, OBJ208, OBJ222, OBJ229
- OBJ230, OBJ300, OBJ311, OBJ405, OBJ787

**Challenging Classes:**
- OBJ012: 24.9% mAP (requires more training data)
- OBJ108: 4.7% mAP (small size, low contrast)
- OBJ018: 83% mAP (moderate difficulty)

---

# 🔬 Technical Implementation Details

## Milestone 1: CNN Implementation

### Custom CNN Architecture
```python
Sequential([
    # Block 1
    Conv2D(32, 3×3, padding='same') → BatchNorm → ReLU
    Conv2D(32, 3×3, padding='same') → BatchNorm → ReLU
    MaxPooling2D(2×2) → Dropout(0.25)
    
    # Block 2
    Conv2D(64, 3×3, padding='same') → BatchNorm → ReLU
    Conv2D(64, 3×3, padding='same') → BatchNorm → ReLU
    MaxPooling2D(2×2) → Dropout(0.25)
    
    # Block 3
    Conv2D(128, 3×3, padding='same') → BatchNorm → ReLU
    Conv2D(128, 3×3, padding='same') → BatchNorm → ReLU
    MaxPooling2D(2×2) → Dropout(0.25)
    
    # Classifier
    GlobalAveragePooling2D()
    Dense(256, ReLU) → BatchNorm → Dropout(0.5)
    Dense(35, softmax)
])
```

### Transfer Learning Pipeline
```python
base_model = MobileNetV2(weights='imagenet', include_top=False, 
                         input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze feature extractor

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(35, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Milestone 2: YOLO Implementation

### Multi-Object Image Generator
```python
def create_composite_image(num_objects, split='train'):
    # Create 640×640 canvas
    canvas = Image.new('RGB', (640, 640), (240, 240, 240))
    bounding_boxes = []
    
    # Select random object classes
    selected_classes = random.sample(all_classes, num_objects)
    
    for class_name in selected_classes:
        # Load random object image
        obj_img = load_random_object(class_name, split)
        
        # Resize while maintaining aspect ratio
        obj_img = resize_object(obj_img, max_size=200)
        
        # Find position with <30% overlap
        x, y = find_valid_position(obj_img, used_positions)
        
        # Paste object
        canvas.paste(obj_img, (x, y))
        
        # Calculate YOLO bbox (normalized)
        bbox = {
            'class_id': class_to_id[class_name],
            'x_center': (x + width/2) / 640,
            'y_center': (y + height/2) / 640,
            'width': width / 640,
            'height': height / 640
        }
        bounding_boxes.append(bbox)
    
    return canvas, bounding_boxes
```

### YOLOv8 Training
```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # Load pre-trained

results = model.train(
    data='milestone2/data/dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='mps',
    patience=15,
    optimizer='Adam',
    lr0=0.001,
    plots=True
)
```

### Detection Inference
```python
model = YOLO('milestone2/models/yolo_best.pt')

results = model(image_path, conf=0.25, iou=0.45)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
    # Draw bounding box and label
    draw_bbox(image, (x1, y1, x2, y2), class_names[class_id], confidence)
```

---

# 💻 Hardware & Environment

## Development Setup
- **Device:** MacBook Pro M5 chip (Apple Silicon)
- **RAM:** 16GB+
- **GPU:** Apple Metal Performance Shaders (MPS)
- **OS:** macOS Sonoma

## Software Stack
### Milestone 1
- **Framework:** TensorFlow 2.20.0 (Keras)
- **Python:** 3.13
- **Training Device:** CPU (TF-optimized for Apple Silicon)

### Milestone 2
- **Framework:** PyTorch 2.10.0 + Ultralytics 8.4.12
- **Python:** 3.13
- **Training Device:** MPS (GPU acceleration)

---

# 📊 Comprehensive Results

## Milestone 1: All Models Performance

### Training Metrics
| Model | Train Acc | Val Acc | Test Acc | Best Epoch | Total Epochs |
|-------|-----------|---------|----------|------------|--------------|
| MobileNetV2 | 99.66% | 99.18% | 98.94% | 8 | 13 |
| CustomCNN | 90.52% | 92.90% | 92.61% | 18 | 23 |
| ResNet50 | 51.15% | 65.57% | 63.06% | 29 | 30 |
| EfficientNetB0 | 3.25% | 3.55% | 3.69% | 2 | 7 |

### Confusion Matrix Analysis
- **MobileNetV2:** Near-perfect diagonal, 4 off-diagonal errors
- **CustomCNN:** Strong diagonal, 28 scattered errors
- **ResNet50:** Significant confusion, 140+ errors
- **EfficientNetB0:** Single column (one-class prediction)

---

## Milestone 2: Complete YOLO Metrics

### Training Loss Progression
```
Epoch 1:  box_loss=2.024, cls_loss=4.669, dfl_loss=2.026
Epoch 10: box_loss=0.573, cls_loss=1.780, dfl_loss=0.972
Epoch 20: box_loss=0.436, cls_loss=1.345, dfl_loss=0.923
Epoch 30: box_loss=0.374, cls_loss=0.968, dfl_loss=0.908
Epoch 34: box_loss=0.338, cls_loss=0.850, dfl_loss=0.887  ← Best
Epoch 49: box_loss=0.299, cls_loss=0.684, dfl_loss=0.867
```

### Validation mAP Progression
```
Epoch 1:  mAP@0.5=0.0101,  mAP@0.5:0.95=0.0064
Epoch 10: mAP@0.5=0.473,   mAP@0.5:0.95=0.408
Epoch 20: mAP@0.5=0.591,   mAP@0.5:0.95=0.495
Epoch 30: mAP@0.5=0.819,   mAP@0.5:0.95=0.782
Epoch 34: mAP@0.5=0.912,   mAP@0.5:0.95=0.873  ← Best
Epoch 49: mAP@0.5=0.856,   mAP@0.5:0.95=0.767
```

### Test Set: Per-Class mAP@0.5
| Class Range | Classes | Average mAP | Performance |
|-------------|---------|-------------|-------------|
| 95-100% | 25 classes | 99.5% | Excellent |
| 80-95% | 6 classes | 87.3% | Good |
| 50-80% | 2 classes | 63.1% | Moderate |
| <50% | 2 classes | 14.7% | Needs improvement |

---

# 🎓 Lessons Learned

## Model Selection
✅ Efficient architectures (MobileNetV2) > Large architectures (ResNet50)  
✅ Transfer learning dramatically accelerates training  
✅ Pre-trained weights from large datasets (ImageNet, COCO) generalize well  
✅ Early stopping prevents overfitting and saves time  

## Data Engineering
✅ More data = better performance (400 vs 110 images: +22.7% mAP)  
✅ Quality annotations critical for object detection  
✅ Controlled object placement improves detection learning  
✅ Data augmentation essential for small datasets  

## Training Strategies
✅ Learning rate scheduling improves convergence  
✅ Patience-based early stopping finds optimal checkpoint  
✅ Regular checkpointing prevents loss of progress  
✅ Validation metrics > training metrics for model selection  

---

# 🔮 Future Work

## Short-term Improvements
1. **Increase dataset to 1000+ multi-object images**
2. **Fine-tune YOLOv8 deeper layers** for better adaptation
3. **Implement ensemble detection** (combine multiple YOLO models)
4. **Add instance segmentation** (pixel-level masks)

## Long-term Extensions
1. **Real-time video object detection** (30+ FPS)
2. **Object tracking across frames** (DeepSORT integration)
3. **Mobile deployment** (iOS/Android apps)
4. **Edge device optimization** (Raspberry Pi, Jetson Nano)
5. **3D bounding box estimation**
6. **Few-shot learning** for new object classes

---

# 📄 Reports & Documentation

## Milestone 1
- **PDF Report:** [Project Report - Milestone 1.pdf](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/milestone1/report/Project%20Report%20-%20Milestone%201.pdf)
- **Key Results:** 98.94% classification accuracy with MobileNetV2

## Milestone 2  
- **PDF Report:** 
- **Key Results:** 92.94% mAP@0.5 object detection with YOLOv8

---

# 🏆 Project Achievements

## Quantitative Achievements
| Metric | Value |
|--------|-------|
| **Total Dataset** | 4,108 images (3,708 single + 400 multi) |
| **Classification Accuracy** | 98.94% (Milestone 1) |
| **Detection mAP@0.5** | 92.94% (Milestone 2) |
| **Detection mAP@0.5:0.95** | 88.09% (Milestone 2) |
| **Object Classes** | 35 unique objects |
| **Total Training Time** | ~3 hours (both milestones) |
| **Code Written** | 1,500+ lines (Python) |
| **GitHub Commits** | 15+ |

## Qualitative Achievements
✅ Production-quality classification system  
✅ Robust multi-object detection pipeline  
✅ Comprehensive evaluation methodology  
✅ Professional documentation and visualization  
✅ Reproducible, well-organized codebase  
✅ Real-world applicable system  

---

# 📧 Contact & Team

## Team Members
- **Rithika Sankar Rajeswari** 
- **Sreevarshan Sathiyamurthy** 
- **Meena Periasamy** 
- **Sahil Mohanty**


## Course
- **Course:** IE 7615 - Deep Learning for AI
- **Semester:** Spring 2026
- **Institution:** Northeastern University

---




**⭐ Star this repository if you found it helpful!**

**🔗 [View Milestone 1 Report](https://github.com/rithika-sr/discriminative-deep-learning-project/blob/main/milestone1/report/Project%20Report%20-%20Milestone%201.pdf) | [View Detection Results](https://github.com/rithika-sr/discriminative-deep-learning-project/tree/main/milestone2/results)**
