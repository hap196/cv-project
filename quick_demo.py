import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import random

from models import FCN
from dataset_preparation import ADE20KDataset, get_transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Config
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_SAMPLES = 21970
IMG_SIZE = 512
NUM_CLASSES = 15

def get_tiny_dataset(root_dir='ADE', split='train', num_samples=NUM_SAMPLES):
    """Create a tiny subset of the dataset for demo purposes."""
    # Find all image files
    img_pattern = os.path.join(root_dir, 'images', 'ADE', split, '**', '*.jpg')
    all_img_files = glob.glob(img_pattern, recursive=True)
    print(f"Found {len(all_img_files)} total images in {split} set")
    
    # Sample a subset
    img_files = random.sample(all_img_files, min(num_samples, len(all_img_files)))
    
    # Find corresponding segmentation files
    seg_files = [img_path.replace('.jpg', '_seg.png') for img_path in img_files]
    
    # Verify segmentation files exist
    valid_pairs = []
    for img_path, seg_path in zip(img_files, seg_files):
        if os.path.exists(seg_path):
            valid_pairs.append((img_path, seg_path))
    
    print(f"Using {len(valid_pairs)} image pairs for {split} set")
    return valid_pairs

def calculate_metrics(outputs, targets, num_classes, ignore_index=255):
    """Calculate pixel accuracy and mean IoU."""
    # Convert outputs to predictions
    _, preds = torch.max(outputs, 1)
    
    # Create mask for valid pixels (not ignore_index)
    valid_mask = targets != ignore_index
    
    # Calculate pixel accuracy
    correct = torch.sum((preds == targets) & valid_mask).item()
    total = torch.sum(valid_mask).item()
    pixel_acc = correct / total if total > 0 else 0
    
    # Calculate per-class IoU
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds == cls) & valid_mask
        target_mask = (targets == cls) & valid_mask
        
        intersection = torch.sum(pred_mask & target_mask).item()
        union = torch.sum(pred_mask | target_mask).item()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    mean_iou = sum(ious) / len(ious)
    
    return pixel_acc, mean_iou, ious

def train_quick_demo():
    # Get transforms
    transform, target_transform = get_transforms('train', img_size=IMG_SIZE)
    val_transform, val_target_transform = get_transforms('val', img_size=IMG_SIZE)
    
    # Get tiny datasets
    train_dataset = ADE20KDataset(
        root_dir='ADE',
        split='train',
        transform=transform,
        target_transform=target_transform,
        top_k_classes=NUM_CLASSES
    )
    
    val_dataset = ADE20KDataset(
        root_dir='ADE',
        split='val',
        transform=val_transform,
        target_transform=val_target_transform,
        top_k_classes=NUM_CLASSES
    )
    
    # Limit dataset size for demo
    train_dataset.valid_pairs = train_dataset.valid_pairs[:NUM_SAMPLES]
    val_dataset.valid_pairs = val_dataset.valid_pairs[:min(10, len(val_dataset.valid_pairs))]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Use 0 for faster startup
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
        # Create model
    model = FCN(
        num_classes=NUM_CLASSES,
        backbone="resnet50",
        pretrained=True
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    print("Starting quick demo training...")
    model.train()
    # best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'num_classes': NUM_CLASSES,
            'class_mapping': train_dataset.class_mapping
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Optionally evaluate after each epoch (you can move validation here if preferred)
        # and update best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    
    # Save model
    checkpoint_path = 'demo_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': NUM_CLASSES,
        'class_mapping': train_dataset.class_mapping
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    model.eval()
    val_loss = 0
    val_pixel_acc = 0
    val_mean_iou = 0
    val_samples = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate metrics
            pixel_acc, mean_iou, _ = calculate_metrics(outputs, masks, NUM_CLASSES)
            val_pixel_acc += pixel_acc * images.size(0)
            val_mean_iou += mean_iou * images.size(0)
            val_samples += images.size(0)
    
    # Calculate average metrics
    val_loss /= len(val_loader)
    val_pixel_acc /= val_samples
    val_mean_iou /= val_samples
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Pixel Accuracy: {val_pixel_acc:.4f} ({val_pixel_acc*100:.2f}%)")
    print(f"Mean IoU: {val_mean_iou:.4f} ({val_mean_iou*100:.2f}%)")
    
    # Visualize predictions on validation set
    visualize_predictions(model, val_loader, val_dataset.class_mapping)
    
    return model, train_dataset.class_mapping

def visualize_predictions(model, dataloader, class_mapping, num_samples=3):
    """Visualize model predictions on a few samples."""
    model.eval()
    os.makedirs('demo_output', exist_ok=True)
    
    # Create color map for visualization
    color_map = np.zeros((len(class_mapping), 3), dtype=np.uint8)
    for i in range(len(class_mapping)):
        # Generate a diverse set of colors
        r = (i * 11) % 255
        g = (i * 41) % 255
        b = (i * 97) % 255
        color_map[i] = [r, g, b]
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Calculate per-image metrics
            for j in range(images.size(0)):
                img_idx = i * dataloader.batch_size + j
                
                # Calculate metrics for this image
                img_output = outputs[j:j+1]
                img_mask = masks[j:j+1]
                pixel_acc, mean_iou, class_ious = calculate_metrics(img_output, img_mask, len(class_mapping))
                
                # Denormalize image
                image = images[j].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = image.permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)
                
                # Get mask and prediction
                mask = masks[j].cpu().numpy()
                pred = preds[j].cpu().numpy()
                
                # Create colored masks
                mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                
                for cls_idx in range(len(class_mapping)):
                    mask_colored[mask == cls_idx] = color_map[cls_idx]
                    pred_colored[pred == cls_idx] = color_map[cls_idx]
                
                # Plot
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask_colored)
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_colored)
                plt.title(f'Prediction\nPixel Acc: {pixel_acc:.2f}, mIoU: {mean_iou:.2f}')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'demo_output/sample_{img_idx}.png')
                plt.close()
                
                print(f"Visualization saved for sample {img_idx} - Pixel Acc: {pixel_acc:.4f}, mIoU: {mean_iou:.4f}")
                
                if img_idx >= num_samples - 1:
                    break

if __name__ == "__main__":
    model, class_mapping = train_quick_demo()
    print("Demo completed!") 