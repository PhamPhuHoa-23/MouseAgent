"""
DINO ResNet50 Encoder for RL Fine-tuning
Load pre-trained DINO ResNet50 weights for RL training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

# ML-Agents imports - compatible version
try:
    from mlagents.trainers.torch.model_serialization import exporting_to_onnx
    ML_AGENTS_AVAILABLE = True
except ImportError:
    try:
        from mlagents.trainers.torch import exporting_to_onnx
        ML_AGENTS_AVAILABLE = True
    except ImportError:
        ML_AGENTS_AVAILABLE = False
        print("⚠️ ML-Agents not available, using fallback")

# Fallback class for ONNX export
class DummyExporting:
    @staticmethod
    def is_exporting():
        return False

if not ML_AGENTS_AVAILABLE:
    exporting_to_onnx = DummyExporting()


class NatureVisualEncoder(nn.Module):
    """DINO ResNet50 encoder for RL fine-tuning"""
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        
        self.output_size = output_size
        self.height = height
        self.width = width
        
        print(f"🔥 [DINO ResNet50] Building RL encoder: {height}x{width}x{initial_channels} -> {output_size}")
        
        # ResNet50 backbone (same as pre-training)
        self.resnet = models.resnet50(pretrained=False)  # We'll load DINO weights
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool + fc
        
        # ResNet50 outputs 2048 channels
        backbone_dim = 2048
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 4x4 spatial (smaller than ImageNet)
        
        # Calculate flattened dimension
        flattened_dim = backbone_dim * 4 * 4  # 2048 * 16 = 32,768
        
        # RL head (simpler than classification)
        self.rl_head = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, output_size)
        )
        
        # Load DINO pre-trained weights
        self.load_dino_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"🔥 [DINO ResNet50] Total params: {total_params/1e6:.1f}M")
        print(f"🔥 [DINO ResNet50] Trainable: {trainable_params/1e6:.1f}M")
        print(f"🔥 [DINO ResNet50] Ready for RL fine-tuning!")
    
    def load_dino_weights(self):
        """Load pre-trained DINO weights"""
        checkpoint_paths = [
            "./dino_resnet_checkpoints/best_resnet_checkpoint.pth",
            "./dino_resnet_checkpoints/resnet_checkpoint_epoch_0.pth"
        ]
        
        for checkpoint_path in checkpoint_paths:
            if Path(checkpoint_path).exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    # Extract backbone state dict from DINO checkpoint
                    if 'student_backbone_state_dict' in checkpoint:
                        backbone_state = checkpoint['student_backbone_state_dict']
                    elif 'student_backbone' in checkpoint:
                        backbone_state = checkpoint['student_backbone']
                    elif 'model' in checkpoint:
                        backbone_state = checkpoint['model']
                    else:
                        backbone_state = checkpoint
                    
                    # Remove 'backbone.' prefix from DINO checkpoint if present
                    cleaned_state = {}
                    for key, value in backbone_state.items():
                        if key.startswith('backbone.'):
                            # Remove 'backbone.' prefix
                            new_key = key[9:]  # Remove first 9 chars: 'backbone.'
                            cleaned_state[new_key] = value
                        else:
                            cleaned_state[key] = value
                    
                    print(f"🔧 [DINO ResNet50] Cleaned {len(cleaned_state)} keys (removed backbone prefix)")
                    
                    # Load backbone weights
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state, strict=False)
                    
                    print(f"✅ [DINO ResNet50] Loaded pre-trained weights from: {checkpoint_path}")
                    
                    # Calculate parameter coverage
                    loaded_params = len(cleaned_state) - len(unexpected_keys)
                    total_params = len(self.backbone.state_dict())
                    coverage = loaded_params / total_params * 100 if total_params > 0 else 0
                    
                    if coverage >= 99.5:
                        print(f"🎉 [DINO ResNet50] Perfect loading! Coverage: {coverage:.1f}%")
                    else:
                        if missing_keys:
                            print(f"⚠️  Missing keys: {len(missing_keys)}")
                        if unexpected_keys:
                            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                        print(f"📈 Parameter coverage: {coverage:.1f}%")
                    
                    return
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {checkpoint_path}: {e}")
                    continue
        
        print(f"⚠️  [DINO ResNet50] No pre-trained weights found - using ImageNet initialization")
        # Load ImageNet weights as fallback
        self.resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Handle ML-Agents input format
        if not (ML_AGENTS_AVAILABLE and exporting_to_onnx.is_exporting()):
            images = visual_obs.permute([0, 3, 1, 2])  # BHWC -> BCHW
        else:
            images = visual_obs  # Already BCHW during ONNX export
            
        # Convert grayscale -> RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
            
        # Normalize to [0,1] range  
        images = images.float() / 255.0
        
        # ImageNet normalization (DINO pre-training used this)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # Resize to 224x224 (DINO pre-training size)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # DINO ResNet50 feature extraction
        features = self.backbone(images)  # [B, 2048, 7, 7]
        
        # Adaptive pooling for RL (smaller than ImageNet)
        pooled = self.adaptive_pool(features)  # [B, 2048, 4, 4]
        
        # Flatten (use reshape for safety)
        flattened = pooled.reshape(pooled.size(0), -1)  # [B, 32768]
        
        # RL head
        output = self.rl_head(flattened)  # [B, output_size]
        
        return output
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
        if freeze:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"🧊 [DINO ResNet50] Backbone frozen - {trainable/1e6:.1f}M trainable params")
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"🔥 [DINO ResNet50] Backbone unfrozen - {trainable/1e6:.1f}M trainable params")


# For testing
if __name__ == "__main__":
    print("🧪 Testing DINO ResNet50 Encoder...")
    
    encoder = NatureVisualEncoder(86, 155, 3, 256)
    
    # Test forward pass
    dummy_input = torch.randint(0, 255, (2, 86, 155, 3), dtype=torch.uint8)
    with torch.no_grad():
        output = encoder(dummy_input)
        
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ DINO ResNet50 encoder ready for RL training!")
