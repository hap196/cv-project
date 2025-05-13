import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse
import torch.serialization

# Add argparse.Namespace to the safe globals for unpickling
torch.serialization.add_safe_globals([argparse.Namespace])

from models import FCN
from dataset_preparation import ADE20KDataset, get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize segmentation predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-path', type=str, help='Path to a single image (optional)')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--save-dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    return parser.parse_args()


def get_color_map(num_classes=150):
    """Create a color map for visualization."""
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        # Generate a diverse set of colors
        r = (i * 11) % 255
        g = (i * 41) % 255
        b = (i * 97) % 255
        color_map[i] = [r, g, b]
    return color_map


def visualize_prediction(image, target, prediction, class_mapping, save_path=None):
    """Visualize the prediction alongside the ground truth."""
    # Reverse normalization to get back to original image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # Get color map
    color_map = get_color_map(len(class_mapping))
    
    # Convert target and prediction to colored masks
    target_colored = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    for cls_id, cls_idx in class_mapping.items():
        target_colored[target == cls_idx] = color_map[cls_idx]
        pred_colored[prediction == cls_idx] = color_map[cls_idx]
    
    # Create the figure
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(target_colored)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_colored)
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main(args):
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load the checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        # Try loading with weights_only=False
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating a dummy checkpoint for testing")
        # Create a dummy checkpoint for testing
        checkpoint = {
            'args': argparse.Namespace(num_classes=150, backbone='resnet18'),
            'model_state_dict': None,
            'best_miou': 0.0
        }
    
    # Get the arguments used for training
    model_args = checkpoint['args']
    print(f"Model was trained with {model_args.backbone} backbone for {model_args.num_classes} classes")
    
    # Create and load the model
    model = FCN(
        num_classes=model_args.num_classes,
        backbone=model_args.backbone,
        pretrained=False
    ).to(args.device)
    
    if checkpoint['model_state_dict'] is not None:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Continuing with random weights for testing")
    else:
        print("Using model with random weights for testing")
    
    model.eval()
    
    # Process a single image if provided
    if args.image_path:
        # Transform for a single image
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and process the image
        image = Image.open(args.image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(args.device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
        
        # TODO: You need a class mapping for visualization
        # For now, let's just visualize the raw prediction
        pred_np = pred[0].cpu().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_np, cmap='viridis')
        plt.title('Segmentation Prediction')
        plt.colorbar()
        plt.axis('off')
        
        save_path = os.path.join(args.save_dir, 'single_image_pred.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Visualization saved to {save_path}")
        
    else:
        # Get the validation dataset
        _, val_transform = get_transforms('val')
        
        try:
            val_dataset = ADE20KDataset(
                root_dir='ADE',
                split='val',
                transform=val_transform
            )
            
            # Get class mapping from the dataset
            class_mapping = val_dataset.class_mapping
            
            # Visualize predictions for a few images
            num_images = min(args.num_images, len(val_dataset))
            
            for i in range(num_images):
                # Get image and target
                image, target = val_dataset[i]
                
                # Add batch dimension and move to device
                image_batch = image.unsqueeze(0).to(args.device)
                
                # Get prediction
                with torch.no_grad():
                    output = model(image_batch)
                    _, pred = torch.max(output, 1)
                
                # Remove batch dimension
                pred = pred[0].cpu().numpy()
                target_np = target.cpu().numpy()
                
                # Save visualization
                save_path = os.path.join(args.save_dir, f'pred_{i}.png')
                visualize_prediction(
                    image, target_np, pred, class_mapping, save_path
                )
                
                print(f"Visualization {i+1}/{num_images} saved to {save_path}")
        except Exception as e:
            print(f"Error visualizing dataset images: {e}")
            print("Creating a dummy visualization")
            
            # Create a dummy image and mask
            dummy_image = torch.rand(3, 512, 512)
            dummy_target = torch.zeros(512, 512, dtype=torch.long)
            dummy_pred = torch.zeros(512, 512, dtype=torch.long)
            
            # Create a dummy class mapping
            dummy_class_mapping = {0: 0, 1: 1, 2: 2}
            
            # Save visualization
            save_path = os.path.join(args.save_dir, 'dummy_pred.png')
            visualize_prediction(
                dummy_image, dummy_target.numpy(), dummy_pred.numpy(), 
                dummy_class_mapping, save_path
            )
            
            print(f"Dummy visualization saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 