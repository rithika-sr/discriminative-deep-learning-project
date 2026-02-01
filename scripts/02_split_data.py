import os
import shutil
import random
from pathlib import Path

# Configuration
RAW_DATA_PATH = '/Users/rithika/Desktop/Discriminative_Project/data/raw'
PROCESSED_DATA_PATH = '/Users/rithika/Desktop/Discriminative_Project/data/processed'

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def split_dataset():
    """Split dataset into train/val/test sets"""
    
    print("=" * 60)
    print("SPLITTING DATASET INTO TRAIN/VAL/TEST")
    print("=" * 60)
    
    # Get all class folders
    class_folders = sorted([f for f in os.listdir(RAW_DATA_PATH) 
                           if f.startswith('images_OBJ')])
    
    print(f"\nFound {len(class_folders)} classes")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for folder in class_folders:
        class_name = folder.replace('images_', '')
        source_path = os.path.join(RAW_DATA_PATH, folder)
        
        # Get all images
        images = [f for f in os.listdir(source_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle images
        random.seed(42)  # For reproducibility
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * TRAIN_RATIO)
        n_val = int(n_images * VAL_RATIO)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create directories
        train_dir = os.path.join(PROCESSED_DATA_PATH, 'train', class_name)
        val_dir = os.path.join(PROCESSED_DATA_PATH, 'val', class_name)
        test_dir = os.path.join(PROCESSED_DATA_PATH, 'test', class_name)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy2(
                os.path.join(source_path, img),
                os.path.join(train_dir, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(source_path, img),
                os.path.join(val_dir, img)
            )
        
        for img in test_images:
            shutil.copy2(
                os.path.join(source_path, img),
                os.path.join(test_dir, img)
            )
        
        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
        
        print(f"   {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"Total images - Train: {total_train}, Val: {total_val}, Test: {total_test}")
    print(f"Percentages - Train: {total_train/(total_train+total_val+total_test)*100:.1f}%, "
          f"Val: {total_val/(total_train+total_val+total_test)*100:.1f}%, "
          f"Test: {total_test/(total_train+total_val+total_test)*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    split_dataset()