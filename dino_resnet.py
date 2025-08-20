# #!/usr/bin/env python3
# """
# DINOv2 with ResNet Backbone (GPU Memory Friendly)
# Uses ResNet + ImageNet pretrained weights instead of Vision Transformer
# """

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from PIL import Image
# import numpy as np
# import argparse
# from pathlib import Path
# import yaml
# from tqdm import tqdm
# # Optional wandb import
# try:
#     import wandb
#     WANDB_AVAILABLE = True
# except ImportError:
#     WANDB_AVAILABLE = False
#     print("⚠️  wandb not available, skipping logging")
# import math


# class MouseVisionDataset(Dataset):
#     """Dataset for mouse vision self-supervised learning"""
    
#     def __init__(self, data_dir: str, transform=None, load_in_memory: bool = False):
#         self.data_dir = Path(data_dir)
#         self.transform = transform
#         self.load_in_memory = load_in_memory
        
#         # Find all image files
#         self.image_files = list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg"))
        
#         if not self.image_files:
#             raise ValueError(f"No images found in {data_dir}")
        
#         print(f"📊 Found {len(self.image_files)} images")
        
#         # Load all images in memory if requested (not recommended for large datasets)
#         if self.load_in_memory and len(self.image_files) < 10000:
#             print("💾 Loading images into memory...")
#             self.images = []
#             for img_path in tqdm(self.image_files):
#                 img = Image.open(img_path).convert('RGB')
#                 self.images.append(img)
#         else:
#             self.images = None
    
#     def __len__(self):
#         return len(self.image_files)
    
#     def __getitem__(self, idx):
#         if self.images is not None:
#             img = self.images[idx]
#         else:
#             img_path = self.image_files[idx]
#             img = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             # Apply two different augmentations for contrastive learning
#             img1 = self.transform(img)
#             img2 = self.transform(img)
#             return img1, img2
        
#         return img, img


# class ResNetBackbone(nn.Module):
#     """ResNet backbone with ImageNet pretrained weights"""
    
#     def __init__(self, arch='resnet18', pretrained=True, output_dim=512):
#         super().__init__()
        
#         # Load pretrained ResNet
#         if arch == 'resnet18':
#             self.backbone = models.resnet18(pretrained=pretrained)
#             backbone_dim = 512
#         elif arch == 'resnet34':
#             self.backbone = models.resnet34(pretrained=pretrained)
#             backbone_dim = 512
#         elif arch == 'resnet50':
#             self.backbone = models.resnet50(pretrained=pretrained)
#             backbone_dim = 2048
#         else:
#             raise ValueError(f"Unsupported architecture: {arch}")
        
#         # Remove the final classification layer
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
#         # Add projection layer
#         self.projection = nn.Sequential(
#             nn.Linear(backbone_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim)
#         )
        
#         self.output_dim = output_dim
#         print(f"🧠 Created {arch} backbone (pretrained={pretrained}) -> {output_dim}D features")
    
#     def forward(self, x):
#         # Extract features using ResNet backbone
#         features = self.backbone(x)  # [B, backbone_dim, 1, 1]
#         features = features.flatten(1)  # [B, backbone_dim]
        
#         # Project to output dimension
#         projected = self.projection(features)  # [B, output_dim]
        
#         return projected


# class DINOHead(nn.Module):
#     """DINO projection head (lightweight version)"""
    
#     def __init__(self, in_dim, out_dim=8192, bottleneck_dim=256):  # Smaller output for GPU memory
#         super().__init__()
        
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, bottleneck_dim),
#             nn.GELU(),
#             nn.Linear(bottleneck_dim, bottleneck_dim),
#             nn.GELU(),
#             nn.Linear(bottleneck_dim, out_dim)
#         )
        
#         self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
#         self.last_layer.weight_g.data.fill_(1)
#         self.last_layer.weight_g.requires_grad = False
    
#     def forward(self, x):
#         x = self.mlp(x)
#         x = F.normalize(x, dim=-1, p=2)
#         x = self.last_layer(x)
#         return x


# class DINOLoss(nn.Module):
#     """DINO loss with teacher-student framework"""
    
#     def __init__(self, out_dim=8192, teacher_temp=0.07, student_temp=0.1):
#         super().__init__()
#         self.teacher_temp = teacher_temp
#         self.student_temp = student_temp
#         self.center = nn.Parameter(torch.zeros(1, out_dim))
        
#     def forward(self, student_output, teacher_output):
#         """Cross-entropy between softmax outputs of teacher and student networks."""
#         student_out = student_output / self.student_temp
#         student_out = student_out.chunk(2)
        
#         # Teacher centering and sharpening
#         temp = self.teacher_temp
#         teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
#         teacher_out = teacher_out.detach().chunk(2)
        
#         total_loss = 0
#         n_loss_terms = 0
#         for iq, q in enumerate(teacher_out):
#             for v in range(len(student_out)):
#                 if v == iq:
#                     continue
#                 loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
#                 total_loss += loss.mean()
#                 n_loss_terms += 1
#         total_loss /= n_loss_terms
        
#         # Update center with momentum
#         self.update_center(teacher_output)
        
#         return total_loss
    
#     @torch.no_grad()
#     def update_center(self, teacher_output):
#         """Update center used for teacher output centering"""
#         batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
#         batch_center = batch_center / len(teacher_output)
        
#         # EMA update (use .data for in-place update)
#         self.center.data = self.center.data * 0.9 + batch_center * 0.1


# class DINOResNetTrainer:
#     """DINO trainer with ResNet backbone"""
    
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Print GPU info
#         if torch.cuda.is_available():
#             gpu_name = torch.cuda.get_device_name(0)
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
#             print(f"🔥 Using GPU: {gpu_name} ({gpu_memory}GB)")
        
#         # Create ResNet backbones
#         backbone_config = config['backbone']
#         self.student_backbone = ResNetBackbone(**backbone_config).to(self.device)
#         self.teacher_backbone = ResNetBackbone(**backbone_config).to(self.device)
        
#         # Teacher is EMA of student
#         for p in self.teacher_backbone.parameters():
#             p.requires_grad = False
        
#         # DINO heads
#         output_dim = backbone_config['output_dim']
#         head_config = config['head']
#         self.student_head = DINOHead(output_dim, **head_config).to(self.device)
#         self.teacher_head = DINOHead(output_dim, **head_config).to(self.device)
        
#         # Teacher head is EMA of student head
#         for p in self.teacher_head.parameters():
#             p.requires_grad = False
        
#         # Loss function
#         self.criterion = DINOLoss(**config['loss']).to(self.device)
        
#         # Optimizer (different LR for pretrained vs new layers)
#         backbone_params = list(self.student_backbone.backbone.parameters())
#         new_params = list(self.student_backbone.projection.parameters()) + list(self.student_head.parameters())
        
#         self.optimizer = torch.optim.AdamW([
#             {'params': backbone_params, 'lr': config['optimizer']['backbone_lr']},  # Lower LR for pretrained
#             {'params': new_params, 'lr': config['optimizer']['lr']}  # Higher LR for new layers
#         ], weight_decay=config['optimizer']['weight_decay'])
        
#         # Learning rate scheduler
#         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             self.optimizer, T_max=config['training']['epochs']
#         )
        
#         total_params = sum(p.numel() for p in self.student_backbone.parameters()) + sum(p.numel() for p in self.student_head.parameters())
#         print(f"🧠 Model created: {total_params/1e6:.1f}M parameters")
#         print(f"💾 Estimated GPU memory: ~{total_params * 4 / 1024**3:.1f}GB")
    
#     def train_epoch(self, dataloader, epoch):
#         """Train one epoch"""
#         self.student_backbone.train()
#         self.teacher_backbone.train()
        
#         total_loss = 0
#         for batch_idx, (img1, img2) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
#             img1, img2 = img1.to(self.device), img2.to(self.device)
            
#             # Student forward pass
#             student_features1 = self.student_backbone(img1)
#             student_features2 = self.student_backbone(img2)
#             student_out1 = self.student_head(student_features1)
#             student_out2 = self.student_head(student_features2)
#             student_output = torch.cat([student_out1, student_out2], dim=0)
            
#             # Teacher forward pass
#             with torch.no_grad():
#                 teacher_features1 = self.teacher_backbone(img1)
#                 teacher_features2 = self.teacher_backbone(img2)
#                 teacher_out1 = self.teacher_head(teacher_features1)
#                 teacher_out2 = self.teacher_head(teacher_features2)
#                 teacher_output = torch.cat([teacher_out1, teacher_out2], dim=0)
            
#             # Compute loss
#             loss = self.criterion(student_output, teacher_output)
            
#             # Backward pass
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             # Update teacher (EMA)
#             self.update_teacher()
            
#             total_loss += loss.item()
            
#             # Log progress
#             if batch_idx % 50 == 0:
#                 gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
#                 print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, GPU: {gpu_memory:.1f}GB")
                
#             # Clear cache periodically
#             if batch_idx % 100 == 0 and torch.cuda.is_available():
#                 torch.cuda.empty_cache()
        
#         avg_loss = total_loss / len(dataloader)
#         self.lr_scheduler.step()
        
#         return avg_loss
    
#     @torch.no_grad()
#     def update_teacher(self):
#         """Exponential moving average update of teacher"""
#         momentum = self.config['training']['teacher_momentum']
        
#         # Update teacher backbone
#         for student_param, teacher_param in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
#             teacher_param.data.mul_(momentum).add_((1 - momentum) * student_param.data)
        
#         # Update teacher head
#         for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
#             teacher_param.data.mul_(momentum).add_((1 - momentum) * student_param.data)
    
#     def save_checkpoint(self, epoch, loss, filepath):
#         """Save training checkpoint"""
#         torch.save({
#             'epoch': epoch,
#             'student_backbone_state_dict': self.student_backbone.state_dict(),
#             'teacher_backbone_state_dict': self.teacher_backbone.state_dict(),
#             'student_head_state_dict': self.student_head.state_dict(),
#             'teacher_head_state_dict': self.teacher_head.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
#             'loss': loss,
#             'config': self.config
#         }, filepath)
#         print(f"💾 Checkpoint saved: {filepath}")


# def get_transforms():
#     """Get data augmentation transforms for DINO"""
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])


# def main():
#     parser = argparse.ArgumentParser(description="DINO with ResNet backbone")
#     parser.add_argument("--data", type=str, required=True,
#                         help="Path to training data directory")
#     parser.add_argument("--config", type=str, default="dino_resnet_config.yaml",
#                         help="Path to training config")
#     parser.add_argument("--output", type=str, default="./dino_resnet_checkpoints",
#                         help="Output directory for checkpoints")
#     parser.add_argument("--wandb", action="store_true",
#                         help="Use Weights & Biases logging")
    
#     args = parser.parse_args()
    
#     # GPU-friendly config for RTX 4060 8GB
#     default_config = {
#         'backbone': {
#             'arch': 'resnet18',        # Lightweight ResNet
#             'pretrained': True,        # Use ImageNet pretrained
#             'output_dim': 512         # Feature dimension
#         },
#         'head': {
#             'out_dim': 8192,          # Smaller than ViT version
#             'bottleneck_dim': 256
#         },
#         'loss': {
#             'out_dim': 8192,
#             'teacher_temp': 0.07,
#             'student_temp': 0.1
#         },
#         'optimizer': {
#             'lr': 1e-3,               # Higher LR for new layers
#             'backbone_lr': 1e-4,      # Lower LR for pretrained backbone
#             'weight_decay': 1e-4
#         },
#         'training': {
#             'epochs': 50,
#             'batch_size': 64,         # Larger batch size possible with ResNet
#             'teacher_momentum': 0.996
#         }
#     }
    
#     # Load or create config
#     if Path(args.config).exists():
#         with open(args.config, 'r') as f:
#             config = yaml.safe_load(f)
#     else:
#         config = default_config
#         with open(args.config, 'w') as f:
#             yaml.dump(config, f, default_flow_style=False)
#         print(f"📝 Created GPU-friendly ResNet config: {args.config}")
    
#     # Setup output directory
#     output_dir = Path(args.output)
#     output_dir.mkdir(exist_ok=True, parents=True)
    
#     # Initialize wandb if requested
#     if args.wandb and WANDB_AVAILABLE:
#         wandb.init(project="mouse-dino-resnet", config=config)
    
#     # Create dataset and dataloader
#     transform = get_transforms()
#     dataset = MouseVisionDataset(args.data, transform=transform, load_in_memory=False)
    
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=config['training']['batch_size'],
#         shuffle=True, 
#         num_workers=2,  # Reduced for stability
#         pin_memory=True,
#         drop_last=True
#     )
    
#     # Create trainer
#     trainer = DINOResNetTrainer(config)
    
#     print(f"🚀 Starting DINO ResNet pre-training...")
#     print(f"   Dataset: {len(dataset)} images")
#     print(f"   Epochs: {config['training']['epochs']}")
#     print(f"   Batch size: {config['training']['batch_size']}")
#     print(f"   Backbone: {config['backbone']['arch']} (pretrained={config['backbone']['pretrained']})")
#     print(f"   Device: {trainer.device}")
    
#     # Training loop
#     best_loss = float('inf')
    
#     for epoch in range(config['training']['epochs']):
#         loss = trainer.train_epoch(dataloader, epoch)
        
#         print(f"✅ Epoch {epoch}: Loss = {loss:.4f}")
        
#         # Clear GPU cache
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         # Log to wandb
#         if args.wandb and WANDB_AVAILABLE:
#             wandb.log({'epoch': epoch, 'loss': loss})
        
#         # Save checkpoint
#         if loss < best_loss:
#             best_loss = loss
#             trainer.save_checkpoint(
#                 epoch, loss, 
#                 output_dir / f"best_resnet_checkpoint.pth"
#             )
        
#         # Save regular checkpoint
#         if epoch % 10 == 0:
#             trainer.save_checkpoint(
#                 epoch, loss,
#                 output_dir / f"resnet_checkpoint_epoch_{epoch}.pth"
#             )
    
#     print(f"🎉 Training complete! Best loss: {best_loss:.4f}")
#     print(f"📁 Checkpoints saved to: {output_dir}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
DINOv2 with ResNet Backbone - Improved Version
Enhanced with multi-crop strategy for better object-centric attention
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import math
import random

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available, skipping logging")


class MultiCropDataset(Dataset):
    """Dataset with multi-crop strategy for DINO training"""
    
    def __init__(self, data_dir: str, global_transform=None, local_transform=None, 
                 global_crops_number=2, local_crops_number=6):
        self.data_dir = Path(data_dir)
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        
        # Find all image files
        self.image_files = list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg"))
        
        if not self.image_files:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"📊 Found {len(self.image_files)} images")
        print(f"🌾 Multi-crop: {global_crops_number} global + {local_crops_number} local crops")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        crops = []
        
        # Generate global crops (large, covers full context)
        for _ in range(self.global_crops_number):
            if self.global_transform:
                crops.append(self.global_transform(img))
        
        # Generate local crops (small, focuses on object parts)
        for _ in range(self.local_crops_number):
            if self.local_transform:
                crops.append(self.local_transform(img))
        
        return crops


class ResNetBackbone(nn.Module):
    """ResNet backbone with improved architecture"""
    
    def __init__(self, arch='resnet50', pretrained=True, output_dim=2048):
        super().__init__()
        
        # Load pretrained ResNet
        if arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif arch == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Remove the final classification layer and avgpool
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])  # Keep conv layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection layer with BatchNorm
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
        self.output_dim = output_dim
        self.backbone_dim = backbone_dim
        print(f"🧠 Created {arch} backbone (pretrained={pretrained}) -> {output_dim}D features")
    
    def forward(self, x):
        # Extract features using ResNet backbone
        features = self.features(x)  # [B, backbone_dim, H, W]
        pooled = self.avgpool(features)  # [B, backbone_dim, 1, 1]
        pooled = pooled.flatten(1)  # [B, backbone_dim]
        
        # Project to output dimension
        projected = self.projection(pooled)  # [B, output_dim]
        
        return projected


class DINOHead(nn.Module):
    """Enhanced DINO projection head with LayerNorm"""
    
    def __init__(self, in_dim, out_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, bottleneck_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
            layers.append(nn.LayerNorm(bottleneck_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(bottleneck_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Weight normalized final layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        
        print(f"🎯 DINO Head: {in_dim} -> {bottleneck_dim} -> {out_dim}")
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """Enhanced DINO loss with better centering"""
    
    def __init__(self, out_dim, ncrops, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=0, nepochs=100, student_temp=0.1, 
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Temperature schedule for teacher
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global crops for teacher
        
        total_loss = 0
        n_loss_terms = 0
        
        # Cross-entropy loss between all crops
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # Skip same crop
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output centering with momentum"""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOResNetTrainer:
    """Enhanced DINO trainer with multi-crop strategy"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Print GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            print(f"🔥 Using GPU: {gpu_name} ({gpu_memory}GB)")
        
        # Create ResNet backbones
        backbone_config = config['backbone']
        self.student = ResNetBackbone(**backbone_config).to(self.device)
        self.teacher = ResNetBackbone(**backbone_config).to(self.device)
        
        # Teacher is EMA of student
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # DINO heads
        output_dim = backbone_config['output_dim']
        head_config = config['head']
        self.student_head = DINOHead(output_dim, **head_config).to(self.device)
        self.teacher_head = DINOHead(output_dim, **head_config).to(self.device)
        
        # Teacher head is EMA of student head
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        
        # Calculate total crops
        crop_config = config['crops']
        self.ncrops = crop_config['global_crops_number'] + crop_config['local_crops_number']
        
        # Loss function with enhanced configuration
        loss_config = config['loss'].copy()
        loss_config.update({
            'ncrops': self.ncrops,
            'nepochs': config['training']['epochs']
        })
        self.criterion = DINOLoss(**loss_config).to(self.device)
        
        # Optimizer with layer-wise learning rates
        self.setup_optimizer()
        
        # Learning rate scheduler with warmup
        self.setup_scheduler()
        
        # Initialize teacher with student weights
        self.copy_student_to_teacher()
        
        total_params = sum(p.numel() for p in self.student.parameters()) + sum(p.numel() for p in self.student_head.parameters())
        print(f"🧠 Model created: {total_params/1e6:.1f}M parameters")
        print(f"💾 Estimated GPU memory: ~{total_params * 4 / 1024**3:.1f}GB")
    
    def setup_optimizer(self):
        """Setup optimizer with different learning rates for different components"""
        opt_config = self.config['optimizer']
        
        # Different learning rates for different parts
        backbone_params = list(self.student.features.parameters())
        projection_params = list(self.student.projection.parameters())
        head_params = list(self.student_head.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': opt_config['backbone_lr'], 'name': 'backbone'},
            {'params': projection_params, 'lr': opt_config['lr'], 'name': 'projection'},
            {'params': head_params, 'lr': opt_config['lr'], 'name': 'head'}
        ], weight_decay=opt_config['weight_decay'])
    
    def setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        training_config = self.config['training']
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < training_config.get('warmup_epochs', 10):
                return epoch / training_config.get('warmup_epochs', 10)
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - training_config.get('warmup_epochs', 10)) / 
                                         (training_config['epochs'] - training_config.get('warmup_epochs', 10))))
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def copy_student_to_teacher(self):
        """Initialize teacher with student weights"""
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_param.copy_(student_param)
            for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                teacher_param.copy_(student_param)
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch with multi-crop strategy"""
        self.student.train()
        self.teacher.train()
        
        total_loss = 0
        for batch_idx, crops in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # Move all crops to device
            crops = [crop.to(self.device) for crop in crops]
            
            # Student forward pass on all crops
            student_outputs = []
            for crop in crops:
                features = self.student(crop)
                output = self.student_head(features)
                student_outputs.append(output)
            student_output = torch.cat(student_outputs, dim=0)
            
            # Teacher forward pass only on global crops (first 2)
            with torch.no_grad():
                teacher_outputs = []
                for crop in crops[:2]:  # Only global crops
                    features = self.teacher(crop)
                    output = self.teacher_head(features)
                    teacher_outputs.append(output)
                teacher_output = torch.cat(teacher_outputs, dim=0)
            
            # Compute loss
            loss = self.criterion(student_output, teacher_output, epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=3.0)
            torch.nn.utils.clip_grad_norm_(self.student_head.parameters(), max_norm=3.0)
            
            self.optimizer.step()
            
            # Update teacher (EMA)
            self.update_teacher(epoch)
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 20 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}, GPU: {gpu_memory:.1f}GB")
                
            # Clear cache periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        self.lr_scheduler.step()
        
        return avg_loss
    
    @torch.no_grad()
    def update_teacher(self, epoch):
        """Exponential moving average update of teacher with momentum schedule"""
        # Momentum schedule (starts low, increases to final value)
        base_momentum = self.config['training']['teacher_momentum']
        final_momentum = 1 - (1 - base_momentum) * (math.cos(math.pi * epoch / self.config['training']['epochs']) + 1) / 2
        
        # Update teacher backbone
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(final_momentum).add_((1 - final_momentum) * student_param.data)
        
        # Update teacher head
        for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_param.data.mul_(final_momentum).add_((1 - final_momentum) * student_param.data)
    
    def save_checkpoint(self, epoch, loss, filepath):
        """Save training checkpoint"""
        torch.save({
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'student_head_state_dict': self.student_head.state_dict(),
            'teacher_head_state_dict': self.teacher_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }, filepath)
        print(f"💾 Checkpoint saved: {filepath}")


def get_transforms(config):
    """Get multi-crop data augmentation transforms"""
    crop_config = config['crops']
    
    # Global crops transform (large scale, covers context)
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            crop_config['global_size'], 
            scale=crop_config['global_scale'],
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Local crops transform (small scale, focuses on object parts)
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            crop_config['local_size'], 
            scale=crop_config['local_scale'],
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return global_transform, local_transform


def main():
    parser = argparse.ArgumentParser(description="Enhanced DINO with ResNet backbone")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data directory")
    parser.add_argument("--config", type=str, default="dino_resnet_config.yaml",
                        help="Path to training config")
    parser.add_argument("--output", type=str, default="./dino_resnet_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Enhanced config with multi-crop strategy
    default_config = {
        'backbone': {
            'arch': 'resnet18',        # Use ResNet50 for better features
            'pretrained': True,        # Use ImageNet pretrained
            'output_dim': 1024         # Match ResNet50 feature dimension
        },
        'head': {
            'out_dim': 1024,          # Output dimension
            'bottleneck_dim': 256,    # Bottleneck dimension
            'nlayers': 3              # Number of layers in head
        },
        'crops': {
            'global_crops_number': 2,  # Number of global crops
            'local_crops_number': 6,   # Number of local crops
            'global_size': 224,        # Global crop size
            'local_size': 96,          # Local crop size
            'global_scale': [0.6, 1.0], # Global crop scale range
            'local_scale': [0.15, 0.4]  # Local crop scale range (focuses on objects)
        },
        'loss': {
            'out_dim': 1024,
            'warmup_teacher_temp': 0.04,
            'teacher_temp': 0.04,
            'warmup_teacher_temp_epochs': 30,
            'student_temp': 0.1,
            'center_momentum': 0.9
        },
        'optimizer': {
            'lr': 0.001,               # Learning rate for new layers
            'backbone_lr': 0.0005,     # Lower LR for pretrained backbone
            'weight_decay': 0.0001
        },
        'training': {
            'epochs': 30,             # More epochs for convergence
            'batch_size': 16,          # Adjust based on GPU memory
            'teacher_momentum': 0.996,
            'warmup_epochs': 10
        }
    }
    
    # Load or create config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = default_config
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"📝 Created enhanced DINO ResNet config: {args.config}")
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project="mouse-dino-resnet-enhanced", config=config)
    
    # Create transforms
    global_transform, local_transform = get_transforms(config)
    
    # Create dataset and dataloader
    dataset = MultiCropDataset(
        args.data, 
        global_transform=global_transform,
        local_transform=local_transform,
        global_crops_number=config['crops']['global_crops_number'],
        local_crops_number=config['crops']['local_crops_number']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Create trainer
    trainer = DINOResNetTrainer(config)
    
    print(f"🚀 Starting Enhanced DINO ResNet pre-training...")
    print(f"   Dataset: {len(dataset)} images")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Backbone: {config['backbone']['arch']} (pretrained={config['backbone']['pretrained']})")
    print(f"   Multi-crop: {config['crops']['global_crops_number']} global + {config['crops']['local_crops_number']} local")
    print(f"   Device: {trainer.device}")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        loss = trainer.train_epoch(dataloader, epoch)
        
        print(f"✅ Epoch {epoch}: Loss = {loss:.4f}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({'epoch': epoch, 'loss': loss, 'lr': trainer.optimizer.param_groups[0]['lr']})
        
        # Save checkpoint
        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(
                epoch, loss, 
                output_dir / f"best_dino_resnet_checkpoint.pth"
            )
        
        # Save regular checkpoint
        if epoch % 20 == 0:
            trainer.save_checkpoint(
                epoch, loss,
                output_dir / f"dino_resnet_checkpoint_epoch_{epoch}.pth"
            )
    
    print(f"🎉 Training complete! Best loss: {best_loss:.4f}")
    print(f"📁 Checkpoints saved to: {output_dir}")
    
    # Save final model for inference
    final_model = {
        'backbone': trainer.student.state_dict(),
        'head': trainer.student_head.state_dict(),
        'config': config
    }
    torch.save(final_model, output_dir / "dino_resnet_final.pth")
    print(f"🎯 Final model saved for inference: {output_dir}/dino_resnet_final.pth")


if __name__ == "__main__":
    main()