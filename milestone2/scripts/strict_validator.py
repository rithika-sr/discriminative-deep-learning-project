import os
import numpy as np
from PIL import Image

MULTI_OBJECT_PATH = '/Users/rithika/Documents/Discriminative_Project/milestone2/data'

print("=" * 80)
print("STRICT ANNOTATION VALIDATOR - CATCH ALL ISSUES")
print("=" * 80)

def strict_validate():
    total_checked = 0
    total_fixed = 0
    total_removed = 0
    
    for split in ['train', 'val', 'test']:
        print(f"\nValidating {split} set...")
        
        images_dir = os.path.join(MULTI_OBJECT_PATH, 'images', split)
        labels_dir = os.path.join(MULTI_OBJECT_PATH, 'labels', split)
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        for img_file in image_files:
            label_file = img_file.replace('.jpg', '.txt')
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, label_file)
            
            total_checked += 1
            
            # Check if label exists
            if not os.path.exists(label_path):
                print(f"  ❌ Missing label: {img_file}")
                os.remove(img_path)
                total_removed += 1
                continue
            
            # Verify image can be opened
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path)  # Reopen after verify
                assert img.size == (640, 640), f"Wrong size: {img.size}"
            except Exception as e:
                print(f"  ❌ Bad image: {img_file} - {e}")
                os.remove(img_path)
                os.remove(label_path)
                total_removed += 1
                continue
            
            # Read and validate annotations
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                valid_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Must have exactly 5 values
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Strict validation
                    if not (0 <= class_id < 35):
                        continue
                    
                    if not (0.001 <= x_center <= 0.999):
                        continue
                    
                    if not (0.001 <= y_center <= 0.999):
                        continue
                    
                    if not (0.01 <= width <= 0.98):
                        continue
                    
                    if not (0.01 <= height <= 0.98):
                        continue
                    
                    # Check box doesn't go outside image
                    x_min = x_center - width/2
                    x_max = x_center + width/2
                    y_min = y_center - height/2
                    y_max = y_center + height/2
                    
                    if not (0 <= x_min and x_max <= 1 and 0 <= y_min and y_max <= 1):
                        continue
                    
                    # Valid annotation
                    valid_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Must have at least 2 valid objects
                if len(valid_lines) >= 2:
                    # Rewrite with valid lines only
                    with open(label_path, 'w') as f:
                        f.writelines(valid_lines)
                    
                    if len(valid_lines) < len(lines):
                        total_fixed += 1
                else:
                    # Remove image and label
                    os.remove(img_path)
                    os.remove(label_path)
                    total_removed += 1
            
            except Exception as e:
                print(f"  ❌ Bad annotation: {label_file} - {e}")
                os.remove(img_path)
                os.remove(label_path)
                total_removed += 1
        
        # Count remaining
        remaining = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        print(f"  ✓ {split}: {remaining} valid images remaining")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Checked: {total_checked} images")
    print(f"Fixed: {total_fixed} annotations")
    print(f"Removed: {total_removed} invalid pairs")
    
    # Final count
    print("\n" + "=" * 80)
    print("FINAL DATASET SIZE")
    print("=" * 80)
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(MULTI_OBJECT_PATH, 'images', split)
        count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        print(f"  {split.capitalize()}: {count} images")

if __name__ == "__main__":
    strict_validate()
    print("\n✅ Dataset is now clean and ready for training!")