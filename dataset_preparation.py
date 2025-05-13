import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from collections import Counter
import glob

class ADE20KDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, top_k_classes=150):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on input images.
            target_transform (callable, optional): Optional transform to be applied on segmentation masks.
            top_k_classes (int): Number of top classes to keep, sorted by frequency.
        """
        self.root_dir = root_dir
        self.split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.target_transform = target_transform
        self.top_k_classes = top_k_classes
        
        # Find all image files directly using glob pattern
        img_pattern = os.path.join(root_dir, 'images', 'ADE', self.split, '**', '*.jpg')
        self.img_files = glob.glob(img_pattern, recursive=True)
        print(f"Found {len(self.img_files)} images in {img_pattern}")
        
        # Find corresponding segmentation files
        self.seg_files = [img_path.replace('.jpg', '_seg.png') for img_path in self.img_files]
        
        # Verify segmentation files exist
        valid_pairs = []
        for img_path, seg_path in zip(self.img_files, self.seg_files):
            if os.path.exists(seg_path):
                valid_pairs.append((img_path, seg_path))
        
        self.valid_pairs = valid_pairs
        print(f"Found {len(self.valid_pairs)} valid image-segmentation pairs")
        
        if len(self.valid_pairs) == 0:
            print(f"Warning: No valid image-segmentation pairs found for {split} split. Using dummy data.")
            self.dummy_mode = True
            self.class_mapping = {i: i for i in range(top_k_classes)}
            self.ignore_index = 255
        else:
            self.dummy_mode = False
            # Build class mapping for the top-k most frequent classes
            self.build_class_mapping()
            
        print(f"Loaded dataset for {split} set")
        
    def build_class_mapping(self):
        """Build a mapping from original class IDs to a consecutive range [0, top_k_classes]"""
        # Count class frequencies
        class_counter = Counter()
        
        # Sample a subset of segmentation files to analyze (for efficiency)
        sample_size = min(1000, len(self.valid_pairs))
        sample_indices = np.random.choice(len(self.valid_pairs), sample_size, replace=False)
        
        for idx in sample_indices:
            _, seg_path = self.valid_pairs[idx]
            mask = np.array(Image.open(seg_path))
            # For grayscale images, we want the values directly
            if mask.ndim == 3 and mask.shape[2] == 3:
                # Convert RGB to single-channel if needed (use first channel)
                mask = mask[:, :, 0]
            unique_classes = np.unique(mask)
            for cls in unique_classes:
                class_counter[cls] += 1
        
        # Sort classes by frequency and keep top-k
        self.class_list = [cls for cls, _ in class_counter.most_common(self.top_k_classes)]
        
        # Add class 0 (usually background) if not already in the list
        if 0 not in self.class_list:
            self.class_list = [0] + self.class_list[:-1]
            
        # Create mapping from original classes to new indices
        self.class_mapping = {cls: i for i, cls in enumerate(self.class_list)}
        
        # Map all other classes to the "ignore" index (255 is commonly used)
        self.ignore_index = 255
        
        print(f"Created mapping for top {len(self.class_list)} classes")
        
    def __len__(self):
        if self.dummy_mode:
            # Return a small number in dummy mode
            return 10
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        if self.dummy_mode:
            # Create dummy data for testing
            dummy_image = torch.randn(3, 512, 512)
            dummy_mask = torch.zeros(512, 512, dtype=torch.long)
            return dummy_image, dummy_mask
            
        # Load image and segmentation
        img_path, seg_path = self.valid_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(seg_path)
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        
        # Process the mask
        if self.target_transform:
            # Apply transform first (typically resize)
            mask = self.target_transform(mask)
        
        # Convert mask to numpy array
        mask_np = np.array(mask)
        
        # For RGB masks, use only the first channel
        if mask_np.ndim == 3 and mask_np.shape[2] == 3:
            mask_np = mask_np[:, :, 0]
        
        # Map classes to new indices or ignore_index
        new_mask = np.ones_like(mask_np) * self.ignore_index
        for orig_cls, new_cls in self.class_mapping.items():
            new_mask[mask_np == orig_cls] = new_cls
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(new_mask).long()
        
        return image, mask_tensor

# Define transformations
def get_transforms(split, img_size=512):
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # We don't apply normalization to the mask, only resize
    target_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    
    return transform, target_transform

def get_dataloaders(root_dir='ADE', batch_size=4):
    try:
        # Create datasets
        train_transform, train_target_transform = get_transforms('train')
        val_transform, val_target_transform = get_transforms('val')
        
        train_dataset = ADE20KDataset(
            root_dir=root_dir,
            split='train',
            transform=train_transform,
            target_transform=train_target_transform
        )
        
        val_dataset = ADE20KDataset(
            root_dir=root_dir,
            split='val',
            transform=val_transform,
            target_transform=val_target_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Creating dummy dataloaders for testing purposes")
        
        class DummyDataset(Dataset):
            def __init__(self, size=100, img_size=512, num_classes=150):
                self.size = size
                self.img_size = img_size
                self.num_classes = num_classes
                self.class_mapping = {i: i for i in range(num_classes)}
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Create more interesting dummy data with circles and rectangles
                # Create a blank image
                image = torch.zeros(3, self.img_size, self.img_size)
                mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
                
                # Add random background color
                image[0] = torch.rand(1) * 0.5  # R channel
                image[1] = torch.rand(1) * 0.5  # G channel
                image[2] = torch.rand(1) * 0.5  # B channel
                
                # Add some random shapes
                for i in range(5):  # Add 5 random shapes
                    # Random shape type (0: circle, 1: rectangle)
                    shape_type = torch.randint(0, 2, (1,)).item()
                    
                    # Random position
                    x = torch.randint(50, self.img_size-50, (1,)).item()
                    y = torch.randint(50, self.img_size-50, (1,)).item()
                    
                    # Random size
                    size = torch.randint(30, 100, (1,)).item()
                    
                    # Random color
                    color_r = torch.rand(1).item() * 0.5 + 0.5  # Brighter colors
                    color_g = torch.rand(1).item() * 0.5 + 0.5
                    color_b = torch.rand(1).item() * 0.5 + 0.5
                    
                    # Random class (1-5)
                    class_id = torch.randint(1, 6, (1,)).item()
                    
                    if shape_type == 0:  # Circle
                        for ii in range(self.img_size):
                            for jj in range(self.img_size):
                                if (ii - x)**2 + (jj - y)**2 < size**2:
                                    image[0, ii, jj] = color_r
                                    image[1, ii, jj] = color_g
                                    image[2, ii, jj] = color_b
                                    mask[ii, jj] = class_id
                    else:  # Rectangle
                        x_min = max(0, x - size//2)
                        x_max = min(self.img_size, x + size//2)
                        y_min = max(0, y - size//2)
                        y_max = min(self.img_size, y + size//2)
                        
                        image[0, x_min:x_max, y_min:y_max] = color_r
                        image[1, x_min:x_max, y_min:y_max] = color_g
                        image[2, x_min:x_max, y_min:y_max] = color_b
                        mask[x_min:x_max, y_min:y_max] = class_id
                
                return image, mask
        
        train_dataset = DummyDataset(size=100)
        val_dataset = DummyDataset(size=50)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(root_dir='ADE', batch_size=4)
    
    # Visualize class mapping
    try:
        dataset = train_loader.dataset
        print(f"Top class mapping: {list(dataset.class_mapping.items())[:10]}")
        
        # Get a batch
        images, masks = next(iter(train_loader))
        print(f"Batch shapes: {images.shape}, {masks.shape}")
        
        # Calculate class distribution in the first batch
        unique_classes = torch.unique(masks)
        print(f"Classes in batch: {unique_classes}")
    except Exception as e:
        print(f"Error in visualization: {e}") 