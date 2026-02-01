import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
PROCESSED_DATA_PATH = '/Users/rithika/Desktop/Discriminative_Project/data/processed'
MODELS_PATH = '/Users/rithika/Desktop/Discriminative_Project/models'
RESULTS_PATH = '/Users/rithika/Desktop/Discriminative_Project/results'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 35

# Create directories
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 70)
print("CNN MODEL TRAINING PIPELINE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Number of classes: {NUM_CLASSES}")

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_PATH, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nData loaded:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(RESULTS_PATH, 'class_labels.json'), 'w') as f:
    json.dump(class_labels, f, indent=2)

def build_custom_cnn():
    """Build a custom CNN from scratch"""
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='CustomCNN')
    
    return model

def build_transfer_model(base_model_name):
    """Build transfer learning model"""
    
    if base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom top
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=base_model_name)
    
    return model

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, model_name):
    """Evaluate model and generate metrics"""
    
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=list(class_labels.values()), output_dict=True)
    
    # Save report
    with open(os.path.join(RESULTS_PATH, f'{model_name}_classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Test accuracy
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(class_labels.values()),
                yticklabels=list(class_labels.values()),
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_acc, test_loss, report

def train_and_evaluate_model(model, model_name):
    """Complete training and evaluation pipeline"""
    
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Train
    start_time = datetime.now()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot history
    plot_training_history(history, model_name)
    
    # Evaluate
    test_acc, test_loss, report = evaluate_model(model, model_name)
    
    # Save model
    model.save(os.path.join(MODELS_PATH, f'{model_name}.keras'))
    print(f"Model saved: {model_name}.keras")
    
    return {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_time': training_time,
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }

# Train all models
results = []

print("\n\n" + "="*70)
print("STARTING MODEL TRAINING")
print("="*70)

# 1. Custom CNN
model1 = build_custom_cnn()
result1 = train_and_evaluate_model(model1, 'CustomCNN')
results.append(result1)

# 2. ResNet50
model2 = build_transfer_model('ResNet50')
result2 = train_and_evaluate_model(model2, 'ResNet50')
results.append(result2)

# 3. MobileNetV2
model3 = build_transfer_model('MobileNetV2')
result3 = train_and_evaluate_model(model3, 'MobileNetV2')
results.append(result3)

# 4. EfficientNetB0
model4 = build_transfer_model('EfficientNetB0')
result4 = train_and_evaluate_model(model4, 'EfficientNetB0')
results.append(result4)

# Save comparison results
with open(os.path.join(RESULTS_PATH, 'model_comparison.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Print final comparison
print("\n\n" + "="*70)
print("FINAL MODEL COMPARISON")
print("="*70)
print(f"\n{'Model':<20} {'Test Acc':<12} {'Val Acc':<12} {'Time (s)':<12}")
print("-"*70)
for r in results:
    print(f"{r['model_name']:<20} {r['test_accuracy']:<12.4f} {r['best_val_accuracy']:<12.4f} {r['training_time']:<12.1f}")

best_model = max(results, key=lambda x: x['test_accuracy'])
print(f"\n🏆 BEST MODEL: {best_model['model_name']} with {best_model['test_accuracy']:.4f} test accuracy")
print("="*70)