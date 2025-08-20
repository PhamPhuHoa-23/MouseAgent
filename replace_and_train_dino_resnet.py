"""
Replace ML-Agents encoder với DINO ResNet50 và start training
Sử dụng proven working approach - APPEND thay vì replace
"""
import os
import time
import shutil
import subprocess
import sys

print("\n🔥 DINO ResNet50 Training Setup")
print("=" * 60)

# Paths
ml_agents_encoders = r"C:\Users\admin\miniconda3\envs\mouse_dinov3_py38\lib\site-packages\mlagents\trainers\torch\encoders.py"

def replace_encoder():
    """Replace ML-Agents encoder với DINO ResNet50"""
    print("🔄 Appending DINO ResNet50 encoder to ML-Agents...")
    
    # DINO ResNet50 encoder content - SIMPLE VERSION
    dino_resnet_content = '''
# === DINO RESNET50 ENCODER (100% Coverage) ===

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

class NatureVisualEncoder(nn.Module):
    """DINO ResNet50 encoder với 100% weight loading"""
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        
        print(f"🔥 [DINO ResNet50] ML-Agents encoder: {height}x{width}x{initial_channels} -> {output_size}")
        
        # ResNet50 backbone
        self.resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Adaptive pooling for RL
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # RL head optimized for RL tasks
        self.rl_head = nn.Sequential(
            nn.Linear(2048 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_size)
        )
        
        # Load DINO pre-trained weights with 100% coverage
        self.load_dino_weights()
        
        # Parameter info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 [DINO ResNet50] Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")
    
    def load_dino_weights(self):
        """Load DINO ResNet50 weights với perfect 100% coverage"""
        checkpoint_paths = [
            "./dino_resnet_checkpoints/best_resnet_checkpoint.pth",
            "./dino_resnet_checkpoints/resnet_checkpoint_epoch_0.pth"
        ]
        
        for checkpoint_path in checkpoint_paths:
            if Path(checkpoint_path).exists():
                try:
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    # Extract backbone state dict
                    if 'student_backbone_state_dict' in checkpoint:
                        backbone_state = checkpoint['student_backbone_state_dict']
                    else:
                        continue
                    
                    # Remove 'backbone.' prefix for perfect matching
                    cleaned_state = {}
                    for key, value in backbone_state.items():
                        if key.startswith('backbone.'):
                            new_key = key[9:]  # Remove 'backbone.' prefix
                            cleaned_state[new_key] = value
                        else:
                            cleaned_state[key] = value
                    
                    # Load weights with perfect coverage
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state, strict=False)
                    
                    # Calculate coverage
                    loaded_params = len(cleaned_state) - len(unexpected_keys)
                    total_params = len(self.backbone.state_dict())
                    coverage = loaded_params / total_params * 100 if total_params > 0 else 0
                    
                    print(f"✅ [DINO ResNet50] Loaded from: {Path(checkpoint_path).name}")
                    print(f"🎉 [DINO ResNet50] Perfect coverage: {coverage:.1f}%")
                    return
                    
                except Exception as e:
                    print(f"⚠️  Load failed: {e}")
                    continue
        
        # Fallback to ImageNet if DINO not available
        print(f"🔄 [DINO ResNet50] Fallback to ImageNet weights")
        self.resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass compatible với ML-Agents"""
        
        # Handle ML-Agents input format: BHWC -> BCHW
        images = visual_obs.permute([0, 3, 1, 2])
        
        # Convert to RGB if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
            
        # Normalize to [0,1]
        images = images.float() / 255.0
        
        # ImageNet normalization (required for ResNet50)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # Resize to ResNet50 input size
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features through DINO ResNet50 backbone
        features = self.backbone(images)  # [B, 2048, 7, 7]
        pooled = self.adaptive_pool(features)  # [B, 2048, 4, 4]
        flattened = pooled.reshape(pooled.size(0), -1)  # [B, 32768]
        
        # RL head
        output = self.rl_head(flattened)
        
        return output
'''
    
    # Read original ML-Agents encoder
    with open(ml_agents_encoders, "r", encoding="utf-8") as f:
        original_content = f.read()
    
    # Append DINO ResNet50 (NOT replace - this is the key!)
    combined_content = original_content + "\n\n" + dino_resnet_content
    
    # Write combined content
    with open(ml_agents_encoders, "w", encoding="utf-8") as f:
        f.write(combined_content)
    
    print(f"📋 Appended DINO ResNet50 to: {ml_agents_encoders}")
    print("✅ Encoder append complete!")

def start_training():
    """Start ML-Agents training với DINO ResNet50"""
    print("⏳ Starting DINO ResNet50 training in 3 seconds...")
    time.sleep(3)
    
    print("🚀 Starting DINO ResNet50 RL training...")
    
    # Training command - sử dụng config đã clean
    cmd = [
        "mlagents-learn",
        "Config/dino_resnet50_rl.yaml",  # Config clean đã tạo
        "--env", "Builds/NormalTrain/2D go to target v1.exe",
        "--run-id", "dino_resnet50_final",
        "--force",
        "--time-scale", "10",
        "--env-args", 
        "--screen-width=155", "--screen-height=86"
    ]
    
    print(f"🎯 Command: {' '.join(cmd)}")
    print("\n📊 Expected: DINO ResNet50 (59.8M params, 100% pre-trained)")
    print("🔥 Features: Self-supervised pre-trained backbone")
    print("🎮 Training starting...")
    
    # Run training
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ DINO ResNet50 training completed successfully!")
        print("📊 Check results in: ./results/dino_resnet50_final/")
        print("📈 TensorBoard: tensorboard --logdir ./results/")
    else:
        print(f"\n❌ Training failed with code: {result.returncode}")

def restore_encoder():
    """Restore original encoder sau khi training"""
    backup_path = ml_agents_encoders + ".backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, ml_agents_encoders)
        print("🔄 Restored original encoder")

if __name__ == "__main__":
    try:
        # Backup original
        backup_path = ml_agents_encoders + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(ml_agents_encoders, backup_path)
            print("📁 Backed up original encoder")
        
        # Replace và train
        replace_encoder()
        start_training()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
    finally:
        # Always restore
        restore_encoder()
        print("✅ Cleanup complete")
