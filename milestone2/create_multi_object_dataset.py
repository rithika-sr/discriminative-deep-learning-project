import os
import random
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Configuration
PROCESSED_DATA_PATH = '/Users/rithika/Documents/Discriminative_Project/data/processed'
MULTI_OBJECT_PATH = '/Users/rithika/Documents/Discriminative_Project/data/multi_object'
OUTPUT_SIZE = (640, 640)  # Standard YOLO size
MIN_OBJECTS = 2
MAX_OBJECTS = 5
IMAGES_PER_SPLIT = {
    'train': 80,
    'val': 15,
    'test': 15
}

print("=" * 70)
print("MULTI-OBJECT IMAGE GENERATOR FOR YOLO")
print("=" * 70)

# Get all class folders
class_folders = sorted([f for f in os.listdir(os.path.join(PROCESSED_DATA_PATH, 'train'))
                       if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, 'train', f))])

print(f"\nFound {len(class_folders)} classes")
print(f"Classes: {', '.join(class_folders[:10])}... (showing first 10)")

# Create class_id mapping
class_to_id = {class_name: idx for idx, class_name in enumerate(class_folders)}
id_to_class = {idx: class_name for idx, class_name in enumerate(class_folders)}

print(f"\nClass mapping created: {len(class_to_id)} classes")

def load_random_object(class_name, split='train'):
    """Load a random image from a class folder"""
    class_path = os.path.join(PROCESSED_DATA_PATH, split, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        return None, None
    
    random_image = random.choice(images)
    img_path = os.path.join(class_path, random_image)
    
    try:
        img = Image.open(img_path).convert('RGB')
        return img, class_name
    except:
        return None, None

def resize_object(img, max_size=200):
    """Resize object to fit in composite image"""
    # Keep aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def create_composite_image(num_objects, split='train'):
    """Create a composite image with multiple objects"""
    
    # Create blank canvas
    canvas = Image.new('RGB', OUTPUT_SIZE, (240, 240, 240))  # Light gray background
    
    # Storage for bounding boxes
    bounding_boxes = []
    
    # Select random classes
    selected_classes = random.sample(class_folders, num_objects)
    
    # Grid placement to avoid too much overlap
    grid_size = int(np.ceil(np.sqrt(num_objects)))
    cell_width = OUTPUT_SIZE[0] // grid_size
    cell_height = OUTPUT_SIZE[1] // grid_size
    
    used_positions = []
    
    for i, class_name in enumerate(selected_classes):
        # Load random object
        obj_img, obj_class = load_random_object(class_name, split)
        
        if obj_img is None:
            continue
        
        # Resize object
        obj_img = resize_object(obj_img, max_size=min(cell_width, cell_height) - 20)
        obj_width, obj_height = obj_img.size
        
        # Find position with some randomness but avoid complete overlap
        attempts = 0
        while attempts < 50:
            # Random position within canvas
            x = random.randint(10, OUTPUT_SIZE[0] - obj_width - 10)
            y = random.randint(10, OUTPUT_SIZE[1] - obj_height - 10)
            
            # Check if this position overlaps too much with existing objects
            overlap = False
            for (prev_x, prev_y, prev_w, prev_h) in used_positions:
                if (x < prev_x + prev_w and x + obj_width > prev_x and
                    y < prev_y + prev_h and y + obj_height > prev_y):
                    # Calculate overlap area
                    overlap_x = min(x + obj_width, prev_x + prev_w) - max(x, prev_x)
                    overlap_y = min(y + obj_height, prev_y + prev_h) - max(y, prev_y)
                    overlap_area = overlap_x * overlap_y
                    obj_area = obj_width * obj_height
                    
                    # If overlap is more than 30%, try again
                    if overlap_area > 0.3 * obj_area:
                        overlap = True
                        break
            
            if not overlap:
                break
            
            attempts += 1
        
        # Paste object onto canvas
        canvas.paste(obj_img, (x, y))
        used_positions.append((x, y, obj_width, obj_height))
        
        # Calculate YOLO format bounding box (normalized)
        x_center = (x + obj_width / 2) / OUTPUT_SIZE[0]
        y_center = (y + obj_height / 2) / OUTPUT_SIZE[1]
        width_norm = obj_width / OUTPUT_SIZE[0]
        height_norm = obj_height / OUTPUT_SIZE[1]
        
        class_id = class_to_id[class_name]
        
        bounding_boxes.append({
            'class_id': class_id,
            'class_name': class_name,
            'x_center': x_center,
            'y_center': y_center,
            'width': width_norm,
            'height': height_norm
        })
    
    return canvas, bounding_boxes

def save_yolo_annotation(bounding_boxes, output_path):
    """Save bounding boxes in YOLO format"""
    with open(output_path, 'w') as f:
        for bbox in bounding_boxes:
            # YOLO format: class_id x_center y_center width height
            line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
            f.write(line)

def generate_dataset():
    """Generate complete multi-object dataset"""
    
    print("\n" + "=" * 70)
    print("GENERATING MULTI-OBJECT DATASET")
    print("=" * 70)
    
    total_generated = 0
    
    for split, num_images in IMAGES_PER_SPLIT.items():
        print(f"\nGenerating {num_images} images for {split} set...")
        
        for i in range(num_images):
            # Random number of objects
            num_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
            
            # Create composite image
            composite_img, bboxes = create_composite_image(num_objects, split)
            
            # Skip if no objects were placed
            if len(bboxes) == 0:
                continue
            
            # Save image
            img_filename = f'multi_{split}_{i:04d}.jpg'
            img_path = os.path.join(MULTI_OBJECT_PATH, 'images', split, img_filename)
            composite_img.save(img_path, quality=95)
            
            # Save YOLO annotation
            label_filename = f'multi_{split}_{i:04d}.txt'
            label_path = os.path.join(MULTI_OBJECT_PATH, 'labels', split, label_filename)
            save_yolo_annotation(bboxes, label_path)
            
            total_generated += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_images} images ({len(bboxes)} objects in last image)")
        
        print(f"✓ Completed {split} set: {num_images} images")
    
    print("\n" + "=" * 70)
    print(f"DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total images generated: {total_generated}")
    print(f"  Train: {IMAGES_PER_SPLIT['train']}")
    print(f"  Val: {IMAGES_PER_SPLIT['val']}")
    print(f"  Test: {IMAGES_PER_SPLIT['test']}")
    
    return total_generated

def create_dataset_yaml():
    """Create YOLO dataset configuration file"""
    
    yaml_content = f"""# Multi-Object Detection Dataset
path: {MULTI_OBJECT_PATH}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(class_folders)}

# Class names
names:
"""
    
    for idx, class_name in enumerate(class_folders):
        yaml_content += f"  {idx}: {class_name}\n"
    
    yaml_path = os.path.join(MULTI_OBJECT_PATH, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created dataset.yaml at: {yaml_path}")
    print(f"  Classes: {len(class_folders)}")

if __name__ == "__main__":
    print("\nStarting multi-object dataset generation...")
    print(f"Output size: {OUTPUT_SIZE}")
    print(f"Objects per image: {MIN_OBJECTS}-{MAX_OBJECTS}")
    
    # Generate dataset
    total = generate_dataset()
    
    # Create YAML config
    create_dataset_yaml()
    
    print("\n" + "=" * 70)
    print("ALL DONE! ✓")
    print("=" * 70)
    print(f"\nDataset ready at: {MULTI_OBJECT_PATH}")
    print(f"Total images: {total}")
    print("\nNext step: Train YOLOv8 model!")