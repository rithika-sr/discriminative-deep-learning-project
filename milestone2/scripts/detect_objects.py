import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuration
PROJECT_ROOT = '/Users/rithika/Documents/Discriminative_Project'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'milestone2/models/yolo_best.pt')
TEST_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'milestone2/data/images/test')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'milestone2/results')

# Create results directory
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 70)
print("OBJECT DETECTION PIPELINE - MILESTONE 2")
print("=" * 70)

# Load trained model
print("\nLoading trained YOLOv8 model...")
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded from: {MODEL_PATH}")

# Get class names
class_names = model.names
print(f"✓ Model can detect {len(class_names)} classes")

print("\n" + "=" * 70)
print("RUNNING DETECTION ON TEST IMAGES")
print("=" * 70)

# Get test images
test_images = [f for f in os.listdir(TEST_IMAGES_PATH) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]  # Test on first 10 images

print(f"\nTesting on {len(test_images)} sample images...")

# Process each image
detection_results = []

for idx, img_file in enumerate(test_images, 1):
    img_path = os.path.join(TEST_IMAGES_PATH, img_file)
    
    print(f"\n[{idx}/{len(test_images)}] Processing: {img_file}")
    
    # Run detection
    results = model(img_path, conf=0.25, iou=0.45)
    
    # Get results
    result = results[0]
    boxes = result.boxes
    
    # Extract detection info
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()
        
        detections.append({
            'class_id': cls_id,
            'class_name': class_names[cls_id],
            'confidence': conf,
            'bbox': bbox
        })
    
    print(f"  Detected {len(detections)} objects:")
    for det in detections:
        print(f"    - {det['class_name']}: {det['confidence']:.2%} confidence")
    
    detection_results.append({
        'image': img_file,
        'detections': detections
    })
    
    # Visualize detections
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    
    # Draw bounding boxes
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{det['class_name']}: {det['confidence']:.2%}"
        ax.text(
            x1, y1 - 5,
            label,
            color='white',
            fontsize=12,
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', pad=2)
        )
    
    ax.axis('off')
    plt.title(f"Detection Results: {img_file}", fontsize=14, pad=10)
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(RESULTS_PATH, f'detection_{idx:02d}_{img_file}')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: detection_{idx:02d}_{img_file}")

print("\n" + "=" * 70)
print("DETECTION SUMMARY")
print("=" * 70)

# Calculate statistics
total_detections = sum(len(r['detections']) for r in detection_results)
avg_detections = total_detections / len(detection_results)
avg_confidence = np.mean([d['confidence'] for r in detection_results for d in r['detections']])

print(f"\nProcessed {len(detection_results)} test images")
print(f"Total objects detected: {total_detections}")
print(f"Average objects per image: {avg_detections:.1f}")
print(f"Average confidence: {avg_confidence:.2%}")

print("\n" + "=" * 70)
print("SAMPLE DETECTION RESULTS")
print("=" * 70)

# Show first 3 images
for i, result in enumerate(detection_results[:3], 1):
    print(f"\nImage {i}: {result['image']}")
    print(f"  Objects detected: {len(result['detections'])}")
    for det in result['detections']:
        print(f"    - {det['class_name']}: {det['confidence']:.2%}")

print("\n" + "=" * 70)
print("ALL DONE! ✓")
print("=" * 70)
print(f"\nVisualization saved to: {RESULTS_PATH}")
print(f"Total detection images: {len(test_images)}")
print("\nNext step: Write Milestone 2 report!")
print("=" * 70)