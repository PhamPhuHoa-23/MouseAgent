import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
import json
import os
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import argparse


class DynamicMaskingProcess:
    """Dynamic masking for robustness during training"""
    def __init__(self, patch_size=8):
        self.patch_size = patch_size
        
    def __call__(self, image):
        """
        Randomly mask patches in image
        Args:
            image: (C, H, W) tensor
        Returns:
            masked_image: Image with random patches set to 0
        """
        if len(image.shape) == 4:  # Batch dimension
            return torch.stack([self._mask_single(img) for img in image])
        else:
            return self._mask_single(image)
    
    def _mask_single(self, image):
        C, H, W = image.shape
        
        # Random mask ratio [0, 0.8] - không mask quá nhiều
        mask_ratio = random.uniform(0, 0.8)
        
        # Calculate patches to mask
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        total_patches = patch_h * patch_w
        num_masked = int(total_patches * mask_ratio)
        
        if num_masked == 0:
            return image
            
        # Select random patches
        masked_patches = random.sample(range(total_patches), num_masked)
        
        # Apply masking
        masked_image = image.clone()
        for patch_id in masked_patches:
            row = (patch_id // patch_w) * self.patch_size
            col = (patch_id % patch_w) * self.patch_size
            masked_image[:, row:row+self.patch_size, col:col+self.patch_size] = 0
            
        return masked_image


class CNNEncoder(nn.Module):
    """ResNet34-based backbone encoder"""
    def __init__(self, hidden_dim=512, pretrained=True):
        super().__init__()
        # Load pretrained ResNet34
        backbone = resnet34(pretrained=pretrained)
        
        # Remove final layers
        self.early_layers = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Get feature dimension (ResNet34 has 512 features)
        backbone_dim = 512
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        features = self.early_layers(x)  # (B, 512, H', W')
        pooled = self.adaptive_pool(features).flatten(1)  # (B, 512)
        projected = self.projection(pooled)  # (B, hidden_dim)
        return projected


class RobustForagingAgent(nn.Module):
    """Main agent with multi-task self-supervised learning"""
    def __init__(self, hidden_dim=512, action_dim=3, pretrained=True):
        super().__init__()
        
        # Shared backbone
        self.backbone = CNNEncoder(hidden_dim, pretrained)
        
        # Task 1: Future State Prediction
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.state_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task 2: Action Prediction
        self.action_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Robustness component
        self.dmp = DynamicMaskingProcess(patch_size=8)
        
    def future_state_loss(self, img1, action, img2):
        """Future state prediction task"""
        # Apply DMP for robustness
        img1_masked = self.dmp(img1)
        img2_masked = self.dmp(img2)
        
        # Extract features
        feat1 = self.backbone(img1_masked)
        feat2 = self.backbone(img2_masked)
        
        # Predict future state
        action_emb = self.action_encoder(action)
        combined = torch.cat([feat1, action_emb], dim=-1)
        feat2_pred = self.state_predictor(combined)
        
        return F.mse_loss(feat2_pred, feat2)
    
    def action_prediction_loss(self, img1, action, img2):
        """Action prediction task"""
        img1_masked = self.dmp(img1)
        img2_masked = self.dmp(img2)
        
        feat1 = self.backbone(img1_masked)
        feat2 = self.backbone(img2_masked)
        
        # Predict action from state transition
        combined = torch.cat([feat1, feat2], dim=-1)
        action_pred = self.action_predictor(combined)
        
        return F.mse_loss(action_pred, action)
    
    def cycle_consistency_loss(self, img1, action, img2):
        """Cycle consistency for coherent understanding"""
        # Forward: img1 + action -> feat2_pred
        feat1 = self.backbone(self.dmp(img1))
        action_emb = self.action_encoder(action)
        feat2_pred = self.state_predictor(torch.cat([feat1, action_emb], -1))
        
        # Backward: feat1 + feat2_pred -> action_pred
        action_pred = self.action_predictor(torch.cat([feat1, feat2_pred], -1))
        
        return F.mse_loss(action_pred, action)
    
    def masking_consistency_loss(self, img1, action, img2):
        """Feature consistency across different maskings"""
        # Original features
        feat1_orig = self.backbone(self.dmp(img1))
        
        # Different masking
        feat1_alt = self.backbone(self.dmp(img1))
        
        # L2 consistency
        return F.mse_loss(feat1_orig, feat1_alt)


class ForagingDataset(Dataset):
    """Dataset for loading triplet data"""
    def __init__(self, json_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((86, 155)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Load triplets from JSON lines file
        self.triplets = []
        with open(json_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.triplets.append(data)
        
        print(f"Loaded {len(self.triplets)} triplets")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Load images
        img1_path = os.path.join(self.images_dir, triplet['image_t'])
        img2_path = os.path.join(self.images_dir, triplet['image_t1'])
        
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Get action (continuous part only)
        action = torch.tensor(triplet['action']['continuous'], dtype=torch.float32)
        
        return img1, action, img2


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Training for one epoch"""
    model.train()
    total_loss = 0
    future_losses = 0
    action_losses = 0
    cycle_losses = 0
    consistency_losses = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (img1, action, img2) in enumerate(progress_bar):
        img1, action, img2 = img1.to(device), action.to(device), img2.to(device)
        
        optimizer.zero_grad()
        
        # Multi-task losses
        future_loss = model.future_state_loss(img1, action, img2)
        action_loss = model.action_prediction_loss(img1, action, img2)
        cycle_loss = model.cycle_consistency_loss(img1, action, img2)
        consistency_loss = model.masking_consistency_loss(img1, action, img2)
        
        # Weighted total loss
        total_batch_loss = (
            1.0 * future_loss + 
            1.0 * action_loss + 
            0.3 * cycle_loss + 
            0.2 * consistency_loss
        )
        
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update running averages
        total_loss += total_batch_loss.item()
        future_losses += future_loss.item()
        action_losses += action_loss.item()
        cycle_losses += cycle_loss.item()
        consistency_losses += consistency_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Total': f'{total_batch_loss.item():.4f}',
            'Future': f'{future_loss.item():.4f}',
            'Action': f'{action_loss.item():.4f}',
            'Cycle': f'{cycle_loss.item():.4f}',
            'Consist': f'{consistency_loss.item():.4f}'
        })
    
    avg_total = total_loss / len(dataloader)
    avg_future = future_losses / len(dataloader)
    avg_action = action_losses / len(dataloader)
    avg_cycle = cycle_losses / len(dataloader)
    avg_consistency = consistency_losses / len(dataloader)
    
    return avg_total, avg_future, avg_action, avg_cycle, avg_consistency


def main():
    parser = argparse.ArgumentParser(description='Train Robust Foraging Agent')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup data
    json_path = os.path.join(args.data_dir, 'metadata', 'triplets.json')
    images_dir = os.path.join(args.data_dir, 'images')
    
    dataset = ForagingDataset(json_path, images_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup model
    model = RobustForagingAgent(
        hidden_dim=args.hidden_dim,
        action_dim=3,  # Từ dữ liệu của bạn
        pretrained=True
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Training batches: {len(dataloader)}')
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, future_loss, action_loss, cycle_loss, consistency_loss = train_epoch(
            model, dataloader, optimizer, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log results
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Total Loss: {train_loss:.6f}')
        print(f'  Future: {future_loss:.6f}, Action: {action_loss:.6f}')
        print(f'  Cycle: {cycle_loss:.6f}, Consistency: {consistency_loss:.6f}')
        print(f'  Learning Rate: {current_lr:.2e}')
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  → Best model saved (loss: {best_loss:.6f})')
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print('\nTraining completed!')
    print(f'Best loss: {best_loss:.6f}')


if __name__ == '__main__':
    main()