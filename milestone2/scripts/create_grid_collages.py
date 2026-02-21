import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration
PROCESSED_DATA_PATH = '/Users/rithika/Documents/Discriminative_Project/milestone1/data'
MULTI_OBJECT_PATH = '/Users/rithika/Documents/Discriminative_Project/milestone2/data'
OUTPUT_SIZE = (640, 640)

# Generate 200 additional grid-style images
GRID_IMAGES_PER_SPLIT = {
    'train': 160,  # Add to existing 560
    'val': 20,     # Add to existing 70
    'test': 20     # Add to existing 70
}

print("=" * 80)
print("GRID-STYLE COLLAGE GENERATOR - TA STYLE")
print("=" * 80)
print("\nGenerating photo grid collages")
print("Features:")
print("  ✓ Perfect grid layout (3×3, 4×4, 3×4, 2×3, etc.)")
print("  ✓ Objects cropped to fill grid cells")
print("  ✓ No gaps between cells")
print("  ✓ Keeps original photo backgrounds")
print("  ✓ Contact sheet / photo collage style")

# Get all class folders
class_folders = sorted([f for f in os.listdir(os.path.join(PROCESSED_DATA_PATH, 'train'))
                       if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, 'train', f))])

print(f"\n✓ Found {len(class_folders)} classes")

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

def create_grid_collage(num_objects, split='train'):
    """Create tight grid-style photo collages"""
    
    # Determine grid dimensions based on number of objects
    if num_objects <= 4:
        grid_rows, grid_cols = 2, 2
    elif num_objects <= 6:
        grid_rows, grid_cols = 2, 3
    elif num_objects <= 9:
        grid_rows, grid_cols = 3, 3
    elif num_objects <= 12:
        grid_rows, grid_cols = 3, 4
    elif num_objects <= 16:
        grid_rows, grid_cols = 4, 4
    else:
        grid_rows, grid_cols = 4, 5
    
    # Calculate cell dimensions
    cell_width = OUTPUT_SIZE[0] // grid_cols
    cell_height = OUTPUT_SIZE[1] // grid_rows
    
    # Create canvas
    canvas = Image.new('RGB', OUTPUT_SIZE, (255, 255, 255))
    
    # Select random classes
    total_cells = grid_rows * grid_cols
    num_to_place = min(num_objects, total_cells)
    selected_classes = random.sample(class_folders, num_to_place)
    
    # Shuffle which cells to fill
    all_cells = [(row, col) for row in range(grid_rows) for col in range(grid_cols)]
    random.shuffle(all_cells)
    cells_to_use = all_cells[:num_to_place]
    
    bboxes = []
    
    for idx, (row, col) in enumerate(cells_to_use):
        class_name = selected_classes[idx]
        obj_img = load_random_object(class_name, split)
        
        if obj_img is None:
            continue
        
        # Calculate cell position
        cell_x = col * cell_width
        cell_y = row * cell_height
        
        # Resize and crop object to fit cell perfectly
        # Center crop to fill the cell
        obj_img = resize_and_crop_to_cell(obj_img, cell_width, cell_height)
        
        # Paste into grid cell
        canvas.paste(obj_img, (cell_x, cell_y))
        
        # Create bounding box for this grid cell
        # YOLO format: normalized coordinates
        x_center = (cell_x + cell_width / 2) / OUTPUT_SIZE[0]
        y_center = (cell_y + cell_height / 2) / OUTPUT_SIZE[1]
        width_norm = cell_width / OUTPUT_SIZE[0]
        height_norm = cell_height / OUTPUT_SIZE[1]
        
        bboxes.append({
            'class_id': class_to_id[class_name],
            'class_name': class_name,
            'x_center': x_center,
            'y_center': y_center,
            'width': width_norm,
            'height': height_norm
        })
    
    return canvas, bboxes

def resize_and_crop_to_cell(img, target_width, target_height):
    """Resize and center-crop image to fit grid cell exactly"""
    
    # Calculate aspect ratios
    img_aspect = img.width / img.height
    target_aspect = target_width / target_height
    
    if img_aspect > target_aspect:
        # Image is wider - fit height, crop width
        new_height = target_height
        new_width = int(target_height * img_aspect)
    else:
        # Image is taller - fit width, crop height
        new_width = target_width
        new_height = int(target_width / img_aspect)
    
    # Resize
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop to exact cell size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    img = img.crop((left, top, right, bottom))
    
    return img

def save_yolo_annotation(bboxes, output_path):
    """Save bounding boxes in YOLO format"""
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
            f.write(line)

def generate_grid_collages():
    """Generate grid-style collage images"""
    
    print("\n" + "=" * 80)
    print("GENERATING GRID-STYLE COLLAGES")
    print("=" * 80)
    
    total_generated = 0
    
    # Get current max index for each split
    start_indices = {}
    for split in ['train', 'val', 'test']:
        existing_images = [f for f in os.listdir(os.path.join(MULTI_OBJECT_PATH, 'images', split))
                          if f.startswith('multi_') and f.endswith('.jpg')]
        if existing_images:
            # Extract indices from filenames
            indices = [int(f.split('_')[2].split('.')[0]) for f in existing_images]
            start_indices[split] = max(indices) + 1
        else:
            start_indices[split] = 0
    
    for split, num_images in GRID_IMAGES_PER_SPLIT.items():
        print(f"\n{'='*80}")
        print(f"Adding {num_images} grid collages to {split.upper()} set")
        print(f"  Starting from index: {start_indices[split]}")
        print(f"{'='*80}")
        
        for i in range(num_images):
            # Random number of objects for grid
            num_objects = random.choice([4, 6, 9, 12, 16])  # Perfect squares/rectangles
            
            # Create grid collage
            composite_img, bboxes = create_grid_collage(num_objects, split)
            
            if len(bboxes) == 0:
                continue
            
            # Calculate filename index
            img_idx = start_indices[split] + i
            
            # Save image
            img_filename = f'multi_{split}_{img_idx:04d}.jpg'
            img_path = os.path.join(MULTI_OBJECT_PATH, 'images', split, img_filename)
            composite_img.save(img_path, quality=95)
            
            # Save YOLO annotation
            label_filename = f'multi_{split}_{img_idx:04d}.txt'
            label_path = os.path.join(MULTI_OBJECT_PATH, 'labels', split, label_filename)
            save_yolo_annotation(bboxes, label_path)
            
            total_generated += 1
            
            if (i + 1) % 20 == 0 or (i + 1) == num_images:
                print(f"  Generated {i + 1}/{num_images} grid collages ({len(bboxes)} objects in last)")
        
        print(f"✓ Added {num_images} grid collages to {split} set")
    
    print("\n" + "=" * 80)
    print("GRID COLLAGE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal grid collages added: {total_generated}")
    print(f"  Train: +{GRID_IMAGES_PER_SPLIT['train']} (now {560 + GRID_IMAGES_PER_SPLIT['train']} total)")
    print(f"  Val:   +{GRID_IMAGES_PER_SPLIT['val']} (now {70 + GRID_IMAGES_PER_SPLIT['val']} total)")
    print(f"  Test:  +{GRID_IMAGES_PER_SPLIT['test']} (now {70 + GRID_IMAGES_PER_SPLIT['test']} total)")
    
    grand_total = 700 + total_generated
    print(f"\n🎉 GRAND TOTAL DATASET: {grand_total} images")
    print(f"   Previous diverse batches: 700")
    print(f"   New grid collages: {total_generated}")
    
    return total_generated

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Grid collages to add: 200")
    print(f"Output size: {OUTPUT_SIZE}")
    print(f"Grid sizes: 2×2, 2×3, 3×3, 3×4, 4×4")
    print(f"Style: Perfect grid layout")
    
    # Generate grid collages
    total = generate_grid_collages()
    
    print("\n" + "=" * 80)
    print("ALL DONE! ✓")
    print("=" * 80)
    print(f"\nComplete dataset ready at: {MULTI_OBJECT_PATH}")
    print(f"Grid collages added: {total}")
    