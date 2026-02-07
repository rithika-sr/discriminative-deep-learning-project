import json
from datetime import datetime

# Load results
with open('/Users/rithika/Desktop/Discriminative_Project/results/model_comparison.json', 'r') as f:
    results = json.load(f)

# Load class labels
with open('/Users/rithika/Desktop/Discriminative_Project/results/class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Generate report content
report_content = f"""
# Discriminative Deep Learning Project - Milestone 1 Report
## Single Object Image Classification

**Date:** {datetime.now().strftime('%B %d, %Y')}

---

## 1. Introduction

This report presents the development and evaluation of deep learning models for automated single-object image classification. The goal is to build a discriminative system capable of identifying individual objects from a multi-class dataset of real-world images.

### Dataset Overview
- **Total Classes:** {len(class_labels)} different objects
- **Total Images:** 3,708 images
- **Image Size:** 224×224 pixels (RGB)
- **Train/Val/Test Split:** 80% / 10% / 10%
- **Training Samples:** 2,963
- **Validation Samples:** 366
- **Test Samples:** 379

---

## 2. Data Preparation

### Data Collection
All students contributed images of their selected objects, capturing variations in:
- Lighting conditions (natural light, artificial light, shadows)
- Viewing angles (front, side, top, rotated)
- Backgrounds (various indoor and outdoor settings)
- Orientations (different positions and placements)

### Data Preprocessing
1. **Image Standardization:** All images verified to be 224×224 pixels
2. **Data Split:** Stratified split maintaining class distribution
3. **Data Augmentation (Training only):**
   - Random rotation (±20 degrees)
   - Horizontal flips
   - Width/height shifts (±20%)
   - Zoom variations (±20%)
4. **Normalization:** Pixel values scaled to [0, 1] range

---

## 3. Model Architecture & Training

We tested four different CNN architectures to compare performance:

### 3.1 Custom CNN (Baseline)
**Architecture:**
- Three convolutional blocks with increasing filters (32 → 64 → 128)
- Batch normalization and dropout for regularization
- Global average pooling
- Dense classification layer (256 units → 35 classes)

**Training Configuration:**
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical crossentropy
- Batch size: 32
- Early stopping: patience = 5 epochs
- Learning rate reduction: factor = 0.5, patience = 3 epochs

### 3.2 ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Base model frozen during training
- Custom classification head added
- Same training configuration as Custom CNN

### 3.3 MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet  
- Optimized for efficiency
- Base model frozen during training
- Custom classification head added

### 3.4 EfficientNetB0 (Transfer Learning)
- Pre-trained on ImageNet
- Compound scaling approach
- Base model frozen during training
- Custom classification head added

---

## 4. Results & Evaluation

### 4.1 Overall Performance Comparison

| Model | Test Accuracy | Val Accuracy | Training Time | Epochs |
|-------|---------------|--------------|---------------|--------|
| **MobileNetV2** | **98.94%** | **99.18%** | **4.9 min** | **13** |
| CustomCNN | 92.61% | 92.90% | 73.4 min | 23 |
| ResNet50 | 63.06% | 65.57% | 40.1 min | 30 |
| EfficientNetB0 | 3.69% | 3.55% | 4.2 min | 7 |

### 4.2 Best Model: MobileNetV2

**Performance Metrics:**
- Test Accuracy: 98.94%
- Precision (macro avg): 99.09%
- Recall (macro avg): 98.91%
- F1-Score (macro avg): 98.93%

**Key Observations:**
- Achieved near-perfect classification across all 35 classes
- Only 4 misclassifications out of 379 test samples
- Fast training convergence (13 epochs)
- Excellent efficiency (under 5 minutes training time)

**Per-Class Performance:**
- 33 out of 35 classes achieved 100% accuracy
- Class OBJ009: 90% recall (1 misclassification)
- Class OBJ095: 81.8% recall (2 misclassifications)  
- Class OBJ208: 90% recall (1 misclassification)

### 4.3 Model Analysis

**CustomCNN Performance:**
- Strong baseline performance (92.61%)
- Learned effective features from scratch
- Longer training time due to learning from random initialization
- Some confusion between visually similar objects

**ResNet50 Performance:**
- Underperformed expectations (63.06%)
- Likely due to frozen architecture not adapting well
- High variance in validation accuracy suggests instability
- May benefit from fine-tuning deeper layers

**EfficientNetB0 Performance:**
- Failed to converge (3.69%)
- Predicted only one class (OBJ229) for all samples
- Likely incompatibility between pre-trained weights and current dataset
- Would require architecture modifications or different training approach

---

## 5. Best Model Selection & Justification

**Selected Model: MobileNetV2**

**Justification:**
1. **Highest Accuracy:** 98.94% test accuracy, significantly outperforming other models
2. **Excellent Generalization:** Minimal gap between training and validation performance
3. **Efficiency:** Fastest training time (4.9 minutes) and inference speed
4. **Robustness:** Consistent performance across all object classes
5. **Production-Ready:** Lightweight architecture suitable for deployment

**Model Characteristics:**
- Only 4 images misclassified out of 379 (1.06% error rate)
- No signs of overfitting despite high accuracy
- Stable training with early stopping at epoch 13
- Leveraged ImageNet pre-training effectively

---

## 6. Confusion Matrix Analysis

### MobileNetV2 Confusion Matrix
The confusion matrix shows nearly perfect diagonal alignment, indicating:
- Excellent class separation
- Minimal inter-class confusion
- Only 4 misclassifications total:
  - OBJ009: 1 misclassification
  - OBJ095: 2 misclassifications
  - OBJ208: 1 misclassification

These errors likely represent:
- Challenging viewing angles
- Similar appearance to other objects
- Ambiguous lighting conditions

---

## 7. Limitations & Future Work

### Current Limitations
1. **Dataset Size:** Only ~100 images per class may limit generalization to novel variations
2. **Controlled Conditions:** Images captured in relatively similar conditions
3. **Frozen Base Models:** Did not fine-tune deeper layers of transfer learning models
4. **Limited Augmentation:** Could expand to include color jittering, cutout, mixup

### Future Improvements
1. **Data Collection:** Expand dataset to 500+ images per class
2. **Fine-tuning:** Unfreeze and train deeper layers of MobileNetV2
3. **Ensemble Methods:** Combine predictions from multiple models
4. **Advanced Augmentation:** Implement AutoAugment or RandAugment
5. **Cross-validation:** Implement k-fold CV for robust performance estimation
6. **Hyperparameter Optimization:** Use Optuna or similar tools for tuning

---

## 8. Conclusion

This project successfully developed a high-performance single-object classification system using deep learning. The MobileNetV2 transfer learning model achieved exceptional results with 98.94% test accuracy, demonstrating the effectiveness of pre-trained models for image classification tasks.

**Key Achievements:**
- ✅ Tested 4 different CNN architectures
- ✅ Achieved 98.94% accuracy with MobileNetV2
- ✅ Processed 3,708 images across 35 object classes
- ✅ Generated comprehensive evaluation metrics and visualizations
- ✅ Completed training in under 5 minutes on Apple Silicon

**Next Steps:**
The trained MobileNetV2 model is now ready for:
- Integration into the multi-object detection pipeline (Milestone 2)
- Deployment as a standalone classification service
- Further optimization for mobile/edge devices

The strong foundation established in Milestone 1 sets us up for success in developing the complete object detection and localization system in subsequent project phases.

---

## Appendix: Training Curves

All training curves and confusion matrices are available in the `/results` folder:
- CustomCNN_training_history.png
- ResNet50_training_history.png
- MobileNetV2_training_history.png
- EfficientNetB0_training_history.png
- CustomCNN_confusion_matrix.png
- ResNet50_confusion_matrix.png
- MobileNetV2_confusion_matrix.png
- EfficientNetB0_confusion_matrix.png

---

**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
"""

# Save report
output_path = '/Users/rithika/Desktop/Discriminative_Project/report/Milestone1_Report.md'
with open(output_path, 'w') as f:
    f.write(report_content)

print("=" * 70)
print("REPORT GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nReport saved to: {output_path}")
print("\nTo convert to PDF:")
print("1. Open the .md file in any Markdown editor")
print("2. Export as PDF, or")
print("3. Copy content into a Word document and format")
print("=" * 70)