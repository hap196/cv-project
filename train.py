import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataset_preparation import get_dataloaders
from models import FCN

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network on ADE20K')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'vgg16'], 
                        help='Backbone architecture')
    parser.add_argument('--num-classes', type=int, default=150, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Create data loaders
        self.train_loader, self.val_loader = get_dataloaders(
            root_dir='ADE',
            batch_size=args.batch_size
        )
        
        # Create model
        self.model = FCN(
            num_classes=args.num_classes,
            backbone=args.backbone,
            pretrained=True
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Optimizer (different LR for backbone and classifier)
        self.optimizer = optim.AdamW([
            {'params': self.model.get_backbone_params(), 'lr': args.lr / 10},
            {'params': self.model.get_classifier_params(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr / 100
        )
        
        # Track metrics
        self.best_miou = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        
        pbar = tqdm(self.train_loader)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        
        avg_loss = train_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
        return avg_loss
        
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        
        # Setup metrics
        n_classes = self.args.num_classes
        # Initialize confusion matrix
        confusion_matrix = torch.zeros(n_classes, n_classes).to(self.device)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Compute confusion matrix
                _, preds = torch.max(outputs, 1)
                
                # Mask out ignored pixels (255)
                mask = targets != 255
                targets_masked = targets[mask]
                preds_masked = preds[mask]
                
                for t, p in zip(targets_masked.view(-1), preds_masked.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        
        # Calculate per-class IoU
        intersection = torch.diag(confusion_matrix)
        union = (confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection)
        iou = intersection / (union + 1e-6)
        mean_iou = iou.mean().item()
        
        # Calculate pixel accuracy
        accuracy = intersection.sum() / confusion_matrix.sum()
        
        # Print metrics
        avg_loss = val_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | mIoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f}")
        
        # Save model if it has the best mIoU
        if mean_iou > self.best_miou:
            self.best_miou = mean_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_miou': self.best_miou,
                'args': self.args,
            }, os.path.join(self.args.save_dir, f'model_best.pth'))
            print(f"New best model saved with mIoU: {mean_iou:.4f}")
        
        return avg_loss, mean_iou, accuracy
    
    def train(self):
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, miou, accuracy = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'miou': miou,
                    'accuracy': accuracy,
                    'args': self.args,
                }, os.path.join(self.args.save_dir, f'checkpoint_epoch{epoch+1}.pth'))
        
        print(f"Training finished. Best mIoU: {self.best_miou:.4f}")


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train() 