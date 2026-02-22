import os

MULTI_OBJECT_PATH = '/Users/rithika/Documents/Discriminative_Project/milestone2/data'

print("=" * 70)
print("ANNOTATION VALIDATOR & FIXER")
print("=" * 70)

def validate_and_fix_annotations():
    fixed_count = 0
    invalid_count = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(MULTI_OBJECT_PATH, 'labels', split)
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            valid_lines = []
            had_invalid = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    had_invalid = True
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate ranges (must be 0-1)
                    if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 < width <= 1 and 0 < height <= 1):
                        
                        # Also check if box goes outside boundaries
                        x_min = x_center - width/2
                        x_max = x_center + width/2
                        y_min = y_center - height/2
                        y_max = y_center + height/2
                        
                        if (x_min >= 0 and x_max <= 1 and y_min >= 0 and y_max <= 1):
                            valid_lines.append(line)
                        else:
                            had_invalid = True
                    else:
                        had_invalid = True
                
                except:
                    had_invalid = True
                    continue
            
            # Only keep files with at least 2 valid objects
            if len(valid_lines) >= 2:
                if had_invalid:
                    # Rewrite file with only valid lines
                    with open(label_path, 'w') as f:
                        f.writelines(valid_lines)
                    fixed_count += 1
            else:
                # Delete invalid annotation and corresponding image
                os.remove(label_path)
                img_file = label_file.replace('.txt', '.jpg')
                img_path = os.path.join(MULTI_OBJECT_PATH, 'images', split, img_file)
                if os.path.exists(img_path):
                    os.remove(img_path)
                invalid_count += 1
    
    print(f"\n✓ Fixed {fixed_count} annotation files")
    print(f"✓ Removed {invalid_count} invalid image/label pairs")
    print("\nAll annotations now valid!")

if __name__ == "__main__":
    validate_and_fix_annotations()
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE! ✓")
    print("=" * 70)