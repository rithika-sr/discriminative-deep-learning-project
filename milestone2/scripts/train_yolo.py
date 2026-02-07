import os
from ultralytics import YOLO
from pathlib import Path
import yaml

# Configuration
PROJECT_ROOT = '/Users/rithika/Documents/Discriminative_Project'
DATASET_YAML = os.path.join(PROJECT_ROOT, 'milestone2/data/dataset.yaml')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'milestone2/runs')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'milestone2/models')

# Training parameters - UPDATED FOR 400 IMAGES
EPOCHS = 100  # Increased from 50 to 100
BATCH_SIZE = 16
IMG_SIZE = 640
MODEL_SIZE = 'yolov8s.pt'  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)

print("=" * 70)
print("YOLO TRAINING SCRIPT - MILESTONE 2 (400 IMAGES)")
print("=" * 70)

# Verify dataset.yaml exists
if not os.path.exists(DATASET_YAML):
    print(f"ERROR: dataset.yaml not found at {DATASET_YAML}")
    exit(1)

print(f"\n✓ Dataset config found: {DATASET_YAML}")

# Load and display dataset info
with open(DATASET_YAML, 'r') as f:
    dataset_info = yaml.safe_load(f)
    
print(f"✓ Number of classes: {dataset_info['nc']}")
print(f"✓ Training path: {dataset_info['train']}")
print(f"✓ Validation path: {dataset_info['val']}")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"Model: {MODEL_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE}")
print(f"Training images: 320")
print(f"Validation images: 40")
print(f"Test images: 40")
print(f"Output directory: {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("LOADING YOLO MODEL")
print("=" * 70)

# Load pre-trained YOLO model
model = YOLO(MODEL_SIZE)
print(f"✓ Loaded {MODEL_SIZE} with pre-trained weights")

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("Expected training time: 1-2 hours on M3 MacBook")
print("Progress will be shown below...")
print("Training will save checkpoints every 10 epochs.")
print("=" * 70 + "\n")

# Train the model
results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='yolo_400images',  # Updated name
    project=OUTPUT_DIR,
    patience=15,  # Increased patience for 100 epochs
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
    device='mps',  # Use Apple Metal Performance Shaders (M3 GPU)
    workers=4,
    verbose=True,
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    plots=True  # Generate training plots
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

# Get the best model path
best_model_path = os.path.join(OUTPUT_DIR, 'yolo_400images/weights/best.pt')
last_model_path = os.path.join(OUTPUT_DIR, 'yolo_400images/weights/last.pt')

if os.path.exists(best_model_path):
    print(f"✓ Best model found at: {best_model_path}")
    
    # Copy best model to models folder
    import shutil
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'yolo_best.pt')
    shutil.copy(best_model_path, final_model_path)
    print(f"✓ Copied best model to: {final_model_path}")
    
    model_to_validate = best_model_path
elif os.path.exists(last_model_path):
    print(f"⚠ Best model not found, using last model")
    import shutil
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'yolo_last.pt')
    shutil.copy(last_model_path, final_model_path)
    print(f"✓ Copied last model to: {final_model_path}")
    
    model_to_validate = last_model_path
else:
    print("ERROR: No model found!")
    exit(1)

print("\n" + "=" * 70)
print("VALIDATING MODEL ON TEST SET")
print("=" * 70)

# Validate the model
model = YOLO(model_to_validate)
metrics = model.val(data=DATASET_YAML, split='test')

print("\n" + "=" * 70)
print("FINAL TEST SET RESULTS")
print("=" * 70)
print(f"mAP@0.5:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
print(f"Precision:    {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
print(f"Recall:       {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"✓ Dataset: 400 images (320 train, 40 val, 40 test)")
print(f"✓ Model trained for up to {EPOCHS} epochs")
print(f"✓ Best model saved: {final_model_path}")
print(f"✓ Training plots saved: {OUTPUT_DIR}/yolo_400images/")
print(f"✓ Final Performance:")
print(f"  - mAP@0.5: {metrics.box.map50*100:.2f}%")
print(f"  - mAP@0.5:0.95: {metrics.box.map*100:.2f}%")
print(f"  - Precision: {metrics.box.mp*100:.2f}%")
print(f"  - Recall: {metrics.box.mr*100:.2f}%")

print("\n" + "=" * 70)
print("ALL DONE! ✓")
print("=" * 70)
print("\nNext step: Build detection pipeline!")
print(f"View detailed results at: {OUTPUT_DIR}/yolo_400images/")