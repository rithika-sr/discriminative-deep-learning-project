import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Configuration
RAW_DATA_PATH = '/Users/rithika/Desktop/Discriminative_Project/data/raw'

def inspect_dataset():
    """Inspect the complete dataset structure and quality"""
    
    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    
    # Get all class folders
    class_folders = sorted([f for f in os.listdir(RAW_DATA_PATH) 
                           if f.startswith('images_OBJ')])
    
    print(f"\n1. TOTAL CLASSES: {len(class_folders)}")
    print(f"   Class folders: {', '.join(class_folders[:5])}... (showing first 5)")
    
    # Analyze each class
    class_stats = {}
    all_sizes = []
    all_formats = []
    
    print("\n2. PER-CLASS ANALYSIS:")
    print("-" * 60)
    
    for folder in class_folders:
        folder_path = os.path.join(RAW_DATA_PATH, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        class_id = folder.replace('images_', '')
        class_stats[class_id] = len(images)
        
        # Sample first image to check dimensions
        if images:
            sample_img_path = os.path.join(folder_path, images[0])
            try:
                img = Image.open(sample_img_path)
                all_sizes.append(img.size)
                all_formats.append(img.format)
                img.close()
            except Exception as e:
                print(f"   Error reading {folder}: {e}")
        
        print(f"   {class_id}: {len(images)} images")
    
    # Summary statistics
    image_counts = list(class_stats.values())
    
    print("\n3. DATASET STATISTICS:")
    print("-" * 60)
    print(f"   Total images: {sum(image_counts)}")
    print(f"   Average images per class: {np.mean(image_counts):.1f}")
    print(f"   Min images in a class: {min(image_counts)}")
    print(f"   Max images in a class: {max(image_counts)}")
    print(f"   Standard deviation: {np.std(image_counts):.1f}")
    
    # Image dimensions
    size_counter = Counter(all_sizes)
    print(f"\n4. IMAGE DIMENSIONS:")
    print("-" * 60)
    for size, count in size_counter.most_common(5):
        print(f"   {size[0]}x{size[1]}: {count} images")
    
    # Image formats
    format_counter = Counter(all_formats)
    print(f"\n5. IMAGE FORMATS:")
    print("-" * 60)
    for fmt, count in format_counter.items():
        print(f"   {fmt}: {count} images")
    
    # Check for 224x224
    correct_size = sum(1 for size in all_sizes if size == (224, 224))
    print(f"\n6. SIZE VERIFICATION:")
    print("-" * 60)
    print(f"   Images with correct size (224x224): {correct_size}/{len(all_sizes)}")
    if correct_size < len(all_sizes):
        print(f"     WARNING: {len(all_sizes) - correct_size} images need resizing!")
    else:
        print(f"    All images are correctly sized!")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    
    return class_stats, class_folders

if __name__ == "__main__":
    class_stats, class_folders = inspect_dataset()