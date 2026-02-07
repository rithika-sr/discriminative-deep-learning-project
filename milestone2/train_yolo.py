import os
from ultralytics import YOLO
from pathlib import Path
import yaml

# Configuration
PROJECT_ROOT = '/Users/rithika/Documents/Discriminative_Project'
DATASET_YAML = os.path.join(PROJECT_ROOT, 'milestone2/data/dataset.yaml')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'milestone2/runs')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'milestone2/models')

# Training parameters
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
MODEL_SIZE = 'yolov8s.pt'  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)

print("=" * 70)
print("YOLO TRAINING SCRIPT - MILESTONE 2")
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
print(f"Output directory: {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("LOADING YOLO MODEL")
print("=" * 70)

# Load pre-trained YOLO model
model = YOLO(MODEL_SIZE)
print(f"✓ Loaded {MODEL_SIZE} with ImageNet pre-trained weights")

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("This will take 1-2 hours on M3 MacBook...")
print("You can monitor progress in the terminal.")
print("Training will automatically save checkpoints and use early stopping.")
print("=" * 70 + "\n")

# Train the model
results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='yolo_multi_object',
    project=OUTPUT_DIR,
    patience=10,  # Early stopping patience
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
best_model_path = os.path.join(OUTPUT_DIR, 'yolo_multi_object/weights/best.pt')
last_model_path = os.path.join(OUTPUT_DIR, 'yolo_multi_object/weights/last.pt')

if os.path.exists(best_model_path):
    print(f"✓ Best model saved at: {best_model_path}")
    
    # Copy best model to models folder
    import shutil
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'yolo_best.pt')
    shutil.copy(best_model_path, final_model_path)
    print(f"✓ Copied to: {final_model_path}")
else:
    print(f"⚠ Best model not found, using last model")
    if os.path.exists(last_model_path):
        import shutil
        final_model_path = os.path.join(MODEL_SAVE_PATH, 'yolo_last.pt')
        shutil.copy(last_model_path, final_model_path)
        print(f"✓ Copied to: {final_model_path}")

print("\n" + "=" * 70)
print("VALIDATING MODEL ON TEST SET")
print("=" * 70)

# Validate the model
model = YOLO(best_model_path if os.path.exists(best_model_path) else last_model_path)
metrics = model.val(data=DATASET_YAML, split='test')

print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"✓ Model trained for {EPOCHS} epochs (or until early stopping)")
print(f"✓ Best model: {final_model_path}")
print(f"✓ Training plots: {OUTPUT_DIR}/yolo_multi_object/")
print(f"✓ Results:")
print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
print("\n" + "=" * 70)
print("ALL DONE! ✓")
print("=" * 70)
print("\nNext step: Build detection pipeline to test on new images!")
print("Training curves and confusion matrix saved in:", OUTPUT_DIR)