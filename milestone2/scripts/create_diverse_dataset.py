import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import json
from pathlib import Path

# Configuration
PROCESSED_DATA_PATH = '/Users/rithika/Documents/Discriminative_Project/milestone1/data'
MULTI_OBJECT_PATH = '/Users/rithika/Documents/Discriminative_Project/milestone2/data'
OUTPUT_SIZE = (640, 640)
MIN_OBJECTS = 3
MAX_OBJECTS = 6

# 700 images total: 560 train, 70 val, 70 test
TOTAL_IMAGES_PER_SPLIT = {
    'train': 560,
    'val': 70,
    'test': 70
}

# 5 batch styles
BATCH_CONFIG = [
    ('gray_separated', 0.15),      # 15% - Gray background, no overlap
    ('ta_style_collage', 0.30),    # 30% - Realistic collages 
    ('textured_collage', 0.20),    # 20% - Textured backgrounds, tight packing
    ('varied_realistic', 0.20),    # 20% - Varied backgrounds, some overlap
    ('clean_separated', 0.15),     # 15% - White/clean, well-separated
]

print("=" * 80)
print("ADVANCED DIVERSE DATASET GENERATOR - 700 IMAGES")
print("=" * 80)
print("\nGenerating 5 diverse batch styles:")
print("  Batch 1 (15%): Gray background, separated objects")
print("  Batch 2 (30%): TA-style realistic photo collages ⭐ PRIORITY")
print("  Batch 3 (20%): Textured backgrounds, collage style")
print("  Batch 4 (20%): Varied backgrounds, realistic overlap")
print("  Batch 5 (15%): Clean white background, separated")

# Get all class folders
class_folders = sorted([f for f in os.listdir(os.path.join(PROCESSED_DATA_PATH, 'train'))
                       if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, 'train', f))])

print(f"\n✓ Found {len(class_folders)} classes")
print(f"✓ Total images to generate: 700")

# Create class mapping
class_to_id = {class_name: idx for idx, class_name in enumerate(class_folders)}

def load_random_object(class_name, split='train'):
    """Load a random image from a class folder"""
    class_path = os.path.join(PROCESSED_DATA_PATH, split, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        return None
    
    random_image = random.choice(images)
    img_path = os.path.join(class_path, random_image)
    
    try:
        img = Image.open(img_path).convert('RGB')
        return img
    except:
        return None

def create_realistic_background(size):
    """Create realistic textured background like TA's images"""
    backgrounds = [
        ('wood_light', (160, 130, 100)),   # Light wood
        ('wood_dark', (101, 67, 33)),      # Dark wood
        ('granite', (180, 175, 170)),      # Granite countertop
        ('marble', (230, 230, 225)),       # Marble
        ('desk_brown', (120, 90, 70)),     # Brown desk
        ('fabric_beige', (200, 185, 170)), # Beige fabric
        ('wall_white', (240, 238, 235)),   # Off-white wall
        ('concrete', (150, 150, 145)),     # Concrete
        ('tile', (210, 205, 200)),         # Tile floor
    ]
    
    bg_type, base_color = random.choice(backgrounds)
    bg = Image.new('RGB', size, base_color)
    
    # Add realistic texture noise
    noise_intensity = random.randint(15, 40)
    noise = np.random.randint(-noise_intensity, noise_intensity, (size[1], size[0], 3), dtype=np.int16)
    bg_array = np.array(bg, dtype=np.int16)
    bg_array = np.clip(bg_array + noise, 0, 255).astype(np.uint8)
    bg = Image.fromarray(bg_array)
    
    # Add subtle blur for realism
    bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    # Random brightness/contrast variation
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(random.uniform(0.9, 1.1))
    
    return bg

def resize_object(img, max_size=220):
    """Resize object while maintaining aspect ratio"""
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def create_batch1_gray_separated(num_objects, split='train'):
    """Batch 1: Gray background, no overlap (original style)"""
    canvas = Image.new('RGB', OUTPUT_SIZE, (240, 240, 240))
    bboxes = []
    used_positions = []
    selected_classes = random.sample(class_folders, min(num_objects, len(class_folders)))
    
    padding = 30
    
    for class_name in selected_classes:
        obj_img = load_random_object(class_name, split)
        if obj_img is None:
            continue
        
        obj_img = resize_object(obj_img, max_size=180)
        obj_width, obj_height = obj_img.size
        
        for attempt in range(100):
            x = random.randint(padding, OUTPUT_SIZE[0] - obj_width - padding)
            y = random.randint(padding, OUTPUT_SIZE[1] - obj_height - padding)
            
            overlap = False
            for (px, py, pw, ph) in used_positions:
                if (x < px + pw + padding and x + obj_width + padding > px and
                    y < py + ph + padding and y + obj_height + padding > py):
                    overlap = True
                    break
            
            if not overlap:
                canvas.paste(obj_img, (x, y))
                used_positions.append((x, y, obj_width, obj_height))
                bboxes.append(create_bbox(class_name, x, y, obj_width, obj_height))
                break
    
    return canvas, bboxes

def create_batch2_ta_style_collage(num_objects, split='train'):
    """Batch 2: TA-style realistic photo collage (objects touching, real backgrounds)"""
    # Create realistic background
    canvas = create_realistic_background(OUTPUT_SIZE)
    bboxes = []
    used_positions = []
    
    # Use more objects for collage density
    num_objects = random.randint(4, 8)
    selected_classes = random.sample(class_folders, min(num_objects, len(class_folders)))
    
    for class_name in selected_classes:
        obj_img = load_random_object(class_name, split)
        if obj_img is None:
            continue
        
        # Varied sizes for realism
        size = random.randint(120, 250)
        obj_img = resize_object(obj_img, max_size=size)
        obj_width, obj_height = obj_img.size
        
        # Random rotation for natural placement
        if random.random() < 0.3:
            angle = random.randint(-15, 15)
            obj_img = obj_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            obj_width, obj_height = obj_img.size
        
        # Tight packing - objects can touch
        for attempt in range(150):
            x = random.randint(5, OUTPUT_SIZE[0] - obj_width - 5)
            y = random.randint(5, OUTPUT_SIZE[1] - obj_height - 5)
            
            # Allow objects to touch (edges can meet)
            overlap = False
            for (px, py, pw, ph) in used_positions:
                if (x < px + pw and x + obj_width > px and
                    y < py + ph and y + obj_height > py):
                    overlap_x = min(x + obj_width, px + pw) - max(x, px)
                    overlap_y = min(y + obj_height, py + ph) - max(y, py)
                    overlap_area = overlap_x * overlap_y
                    
                    # Allow edges to touch (5% overlap max)
                    if overlap_area > 0.05 * (obj_width * obj_height):
                        overlap = True
                        break
            
            if not overlap:
                # Paste with slight blending for realism
                canvas.paste(obj_img, (x, y))
                used_positions.append((x, y, obj_width, obj_height))
                bboxes.append(create_bbox(class_name, x, y, obj_width, obj_height))
                break
    
    return canvas, bboxes

def create_batch3_textured_collage(num_objects, split='train'):
    """Batch 3: Textured backgrounds, collage style"""
    canvas = create_realistic_background(OUTPUT_SIZE)
    bboxes = []
    used_positions = []
    selected_classes = random.sample(class_folders, min(num_objects, len(class_folders)))
    
    for class_name in selected_classes:
        obj_img = load_random_object(class_name, split)
        if obj_img is None:
            continue
        
        obj_img = resize_object(obj_img, max_size=200)
        obj_width, obj_height = obj_img.size
        
        # Allow moderate overlap
        for attempt in range(100):
            x = random.randint(10, OUTPUT_SIZE[0] - obj_width - 10)
            y = random.randint(10, OUTPUT_SIZE[1] - obj_height - 10)
            
            overlap = False
            for (px, py, pw, ph) in used_positions:
                if (x < px + pw and x + obj_width > px and
                    y < py + ph and y + obj_height > py):
                    overlap_x = min(x + obj_width, px + pw) - max(x, px)
                    overlap_y = min(y + obj_height, py + ph) - max(y, py)
                    overlap_area = overlap_x * overlap_y
                    
                    # Allow up to 15% overlap
                    if overlap_area > 0.15 * (obj_width * obj_height):
                        overlap = True
                        break
            
            if not overlap:
                canvas.paste(obj_img, (x, y))
                used_positions.append((x, y, obj_width, obj_height))
                bboxes.append(create_bbox(class_name, x, y, obj_width, obj_height))
                break
    
    return canvas, bboxes

def create_batch4_varied_realistic(num_objects, split='train'):
    """Batch 4: Varied backgrounds, realistic overlap"""
    # Mix of textured and solid backgrounds
    if random.random() < 0.6:
        canvas = create_realistic_background(OUTPUT_SIZE)
    else:
        color = (random.randint(180, 250), random.randint(180, 250), random.randint(180, 250))
        canvas = Image.new('RGB', OUTPUT_SIZE, color)
    
    bboxes = []
    used_positions = []
    selected_classes = random.sample(class_folders, min(num_objects, len(class_folders)))
    
    for class_name in selected_classes:
        obj_img = load_random_object(class_name, split)
        if obj_img is None:
            continue
        
        obj_img = resize_object(obj_img, max_size=random.randint(140, 220))
        obj_width, obj_height = obj_img.size
        
        # Allow some overlap
        for attempt in range(100):
            x = random.randint(10, OUTPUT_SIZE[0] - obj_width - 10)
            y = random.randint(10, OUTPUT_SIZE[1] - obj_height - 10)
            
            overlap = False
            for (px, py, pw, ph) in used_positions:
                if (x < px + pw and x + obj_width > px and
                    y < py + ph and y + obj_height > py):
                    overlap_x = min(x + obj_width, px + pw) - max(x, px)
                    overlap_y = min(y + obj_height, py + ph) - max(y, py)
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.20 * (obj_width * obj_height):
                        overlap = True
                        break
            
            if not overlap:
                canvas.paste(obj_img, (x, y))
                used_positions.append((x, y, obj_width, obj_height))
                bboxes.append(create_bbox(class_name, x, y, obj_width, obj_height))
                break
    
    return canvas, bboxes

def create_batch5_clean_separated(num_objects, split='train'):
    """Batch 5: Clean white background, well-separated"""
    canvas = Image.new('RGB', OUTPUT_SIZE, (255, 255, 255))
    bboxes = []
    used_positions = []
    selected_classes = random.sample(class_folders, min(num_objects, len(class_folders)))
    
    padding = 25
    
    for class_name in selected_classes:
        obj_img = load_random_object(class_name, split)
        if obj_img is None:
            continue
        
        obj_img = resize_object(obj_img, max_size=190)
        obj_width, obj_height = obj_img.size
        
        for attempt in range(100):
            x = random.randint(padding, OUTPUT_SIZE[0] - obj_width - padding)
            y = random.randint(padding, OUTPUT_SIZE[1] - obj_height - padding)
            
            overlap = False
            for (px, py, pw, ph) in used_positions:
                if (x < px + pw + padding and x + obj_width + padding > px and
                    y < py + ph + padding and y + obj_height + padding > py):
                    overlap = True
                    break
            
            if not overlap:
                canvas.paste(obj_img, (x, y))
                used_positions.append((x, y, obj_width, obj_height))
                bboxes.append(create_bbox(class_name, x, y, obj_width, obj_height))
                break
    
    return canvas, bboxes

def create_bbox(class_name, x, y, width, height):
    """Create YOLO format bounding box"""
    x_center = (x + width / 2) / OUTPUT_SIZE[0]
    y_center = (y + height / 2) / OUTPUT_SIZE[1]
    width_norm = width / OUTPUT_SIZE[0]
    height_norm = height / OUTPUT_SIZE[1]
    
    return {
        'class_id': class_to_id[class_name],
        'class_name': class_name,
        'x_center': x_center,
        'y_center': y_center,
        'width': width_norm,
        'height': height_norm
    }

def save_yolo_annotation(bboxes, output_path):
    """Save bounding boxes in YOLO format"""
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
            f.write(line)

def generate_dataset():
    """Generate complete diverse multi-object dataset"""
    
    print("\n" + "=" * 80)
    print("STARTING GENERATION")
    print("=" * 80)
    
    batch_functions = {
        'gray_separated': create_batch1_gray_separated,
        'ta_style_collage': create_batch2_ta_style_collage,
        'textured_collage': create_batch3_textured_collage,
        'varied_realistic': create_batch4_varied_realistic,
        'clean_separated': create_batch5_clean_separated,
    }
    
    total_generated = 0
    batch_counts = {style: 0 for style, _ in BATCH_CONFIG}
    
    for split, total_images in TOTAL_IMAGES_PER_SPLIT.items():
        print(f"\n{'='*80}")
        print(f"Generating {total_images} images for {split.upper()} set")
        print(f"{'='*80}")
        
        # Calculate images per batch for this split
        batch_allocations = []
        for style, percentage in BATCH_CONFIG:
            count = int(total_images * percentage)
            batch_allocations.append((style, count))
        
        # Adjust last batch to ensure exact total
        total_allocated = sum(count for _, count in batch_allocations)
        if total_allocated < total_images:
            batch_allocations[-1] = (batch_allocations[-1][0], batch_allocations[-1][1] + (total_images - total_allocated))
        
        current_idx = 0
        
        for batch_style, batch_size in batch_allocations:
            batch_name = batch_style.replace('_', ' ').title()
            print(f"\n  📦 {batch_name}: {batch_size} images")
            
            batch_func = batch_functions[batch_style]
            
            for i in range(batch_size):
                num_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
                
                composite_img, bboxes = batch_func(num_objects, split)
                
                if len(bboxes) < 2:
                    composite_img, bboxes = batch_func(2, split)
                
                if len(bboxes) == 0:
                    continue
                
                # Save image
                img_filename = f'multi_{split}_{current_idx:04d}.jpg'
                img_path = os.path.join(MULTI_OBJECT_PATH, 'images', split, img_filename)
                composite_img.save(img_path, quality=95)
                
                # Save YOLO annotation
                label_filename = f'multi_{split}_{current_idx:04d}.txt'
                label_path = os.path.join(MULTI_OBJECT_PATH, 'labels', split, label_filename)
                save_yolo_annotation(bboxes, label_path)
                
                total_generated += 1
                batch_counts[batch_style] += 1
                current_idx += 1
                
                if (i + 1) % 25 == 0 or (i + 1) == batch_size:
                    print(f"     Generated {i + 1}/{batch_size} ({len(bboxes)} objects in last)")
        
        print(f"\n✓ {split.upper()} set complete: {total_images} images")
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal images generated: {total_generated}")
    print(f"  Train: {TOTAL_IMAGES_PER_SPLIT['train']}")
    print(f"  Val:   {TOTAL_IMAGES_PER_SPLIT['val']}")
    print(f"  Test:  {TOTAL_IMAGES_PER_SPLIT['test']}")
    
    print("\n" + "=" * 80)
    print("BATCH DISTRIBUTION")
    print("=" * 80)
    for style, count in batch_counts.items():
        percentage = (count / total_generated) * 100
        print(f"  {style.replace('_', ' ').title():<30} {count:>3} images ({percentage:.1f}%)")
    
    return total_generated

def create_dataset_yaml():
    """Create YOLO dataset configuration file"""
    
    yaml_content = f"""# Multi-Object Detection Dataset - Diverse Batches (700 images)
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
    
    print(f"\n✓ Created dataset.yaml")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Total images: 700")
    print(f"Output size: {OUTPUT_SIZE}")
    print(f"Objects per image: {MIN_OBJECTS}-{MAX_OBJECTS}")
    print(f"collages: 30% of dataset (210 images)")
    
    # Generate dataset
    total = generate_dataset()
    
    # Create YAML config
    create_dataset_yaml()
    
    print("\n" + "=" * 80)
    print("ALL DONE! ✓")
    print("=" * 80)
    print(f"\nDiverse dataset ready at: {MULTI_OBJECT_PATH}")
    print(f"Total images: {total}")
    print("Next step: Train YOLOv8 model on diverse data!")