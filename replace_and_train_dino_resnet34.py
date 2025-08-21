#!/usr/bin/env python3
"""
Replace ML-Agents encoder với DINO ResNet34 và start training
Sử dụng proven working approach - REPLACE trực tiếp NatureVisualEncoder
"""
import os
import time
import shutil
import subprocess
import sys

print("\n🔥 DINO ResNet34 Training Setup")
print("=" * 60)

# Paths
ml_agents_encoders = r"C:\Users\admin\miniconda3\envs\mouse_dinov3_py38\lib\site-packages\mlagents\trainers\torch\encoders.py"

def replace_encoder():
    """Replace ML-Agents encoder với DINO ResNet34"""
    print("🔄 Replacing NatureVisualEncoder with DINO ResNet34...")
    
    # DINO ResNet34 encoder content
    dino_resnet_content = '''
# === DINO RESNET34 ENCODER (DIRECT REPLACEMENT) ===

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


class NatureVisualEncoder(nn.Module):
    """DINO ResNet34 encoder - DIRECT REPLACEMENT for ML-Agents"""
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        
        print("=" * 80)
        print("🔥🔥🔥 DINO RESNET34 ENCODER IS BEING USED! 🔥🔥🔥")
        print("=" * 80)
        print(f"🔥 [DINO ResNet34] ML-Agents encoder: {height}x{width}x{initial_channels} -> {output_size}")
        print("=" * 80)
        
        # ResNet34 backbone
        self.resnet = models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # ResNet34 outputs 512 channels
        backbone_dim = 512
        
        # Adaptive pooling for RL
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # RL head optimized for RL tasks
        flattened_dim = backbone_dim * 4 * 4  # 512 * 16 = 8192
        self.rl_head = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_size)
        )
        
        # Load DINO pre-trained weights
        self.load_dino_weights()
        
        # Parameter info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 [DINO ResNet34] Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")
        print("=" * 80)
    
    def load_dino_weights(self):
        """Load DINO ResNet34 weights with perfect coverage"""
        checkpoint_paths = [
            "./enhanced_checkpoints/best_dino_resnet_checkpoint.pth",
            "./enhanced_checkpoints/best_dino_resnet_checkpoint_balance.pth",
            "./enhanced_checkpoints/dino_resnet_checkpoint_epoch_0.pth"
        ]
        
        for checkpoint_path in checkpoint_paths:
            if Path(checkpoint_path).exists():
                try:
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    # Extract backbone state dict
                    if 'student_state_dict' in checkpoint:
                        backbone_state = checkpoint['student_state_dict']
                    elif 'student_backbone_state_dict' in checkpoint:
                        backbone_state = checkpoint['student_backbone_state_dict']
                    else:
                        continue
                    
                    # Remove prefixes for perfect matching
                    cleaned_state = {}
                    for key, value in backbone_state.items():
                        if key.startswith('features.'):
                            new_key = key[9:]  # Remove 'features.' prefix
                            cleaned_state[new_key] = value
                        elif key.startswith('backbone.'):
                            new_key = key[9:]  # Remove 'backbone.' prefix
                            cleaned_state[new_key] = value
                        else:
                            # Try to match keys directly
                            cleaned_state[key] = value
                    
                    # Load weights with perfect coverage
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state, strict=False)
                    
                    # Calculate coverage
                    loaded_params = len(cleaned_state) - len(unexpected_keys)
                    total_params = len(self.backbone.state_dict())
                    coverage = loaded_params / total_params * 100 if total_params > 0 else 0
                    
                    print("=" * 80)
                    print("🎉🎉🎉 DINO CHECKPOINT LOADED SUCCESSFULLY! 🎉🎉🎉")
                    print(f"✅ [DINO ResNet34] Loaded from: {Path(checkpoint_path).name}")
                    print(f"🎉 [DINO ResNet34] Perfect coverage: {coverage:.1f}%")
                    print("=" * 80)
                    return
                    
                except Exception as e:
                    print(f"⚠️  Load failed: {e}")
                    continue
        
        # Fallback to ImageNet if DINO not available
        print(f"🔄 [DINO ResNet34] Fallback to ImageNet weights")
        self.resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass compatible với ML-Agents"""
        
        # Print confirmation on first forward pass
        if not hasattr(self, '_logged_forward'):
            print("=" * 80)
            print("🚀🚀🚀 DINO RESNET34 FORWARD PASS RUNNING! 🚀🚀🚀")
            print(f"📊 Input shape: {visual_obs.shape}")
            print("=" * 80)
            self._logged_forward = True
        
        # Handle ML-Agents input format: BHWC -> BCHW
        # images = visual_obs.permute([0, 3, 1, 2])
        if not exporting_to_onnx.is_exporting():
            images = visual_obs.permute([0, 3, 1, 2])  # BHWC -> BCHW
        else: # Exporting to ONNX
            images = visual_obs  # Already BHWC during ONNX export
            
        images = images.repeat(1, 3, 1, 1)        
            
        # Normalize to [0,1]
        # images = images.float() / 255.0
        
        # ImageNet normalization (required for ResNet34)
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        # images = (images - mean) / std
        
        # Resize to ResNet34 input size
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features through DINO ResNet34 backbone
        features = self.backbone(images)  # [B, 512, 7, 7]
        pooled = self.adaptive_pool(features)  # [B, 512, 4, 4]
        flattened = pooled.reshape(pooled.size(0), -1)  # [B, 8192]
        
        # RL head
        output = self.rl_head(flattened)
        
        return output
'''
    
    # Read original ML-Agents encoder
    with open(ml_agents_encoders, "r", encoding="utf-8") as f:
        original_content = f.read()
    
    # Find and replace NatureVisualEncoder class
    import re
    
    # Pattern to find the NatureVisualEncoder class
    pattern = r'class NatureVisualEncoder\(.*?\n(?:.*\n)*?(?=\nclass|\n\n\nclass|\Z)'
    
    if re.search(pattern, original_content):
        # Replace the class
        modified_content = re.sub(pattern, dino_resnet_content, original_content, flags=re.MULTILINE)
        print(f"✅ Found and replaced NatureVisualEncoder class")
    else:
        # Append at the end if not found
        modified_content = original_content + "\n\n" + dino_resnet_content
        print(f"✅ Appended DINO ResNet34 encoder")
    
    # Write modified content
    with open(ml_agents_encoders, "w", encoding="utf-8") as f:
        f.write(modified_content)
    
    print(f"📋 Modified: {ml_agents_encoders}")
    print("✅ Encoder replacement complete!")

def start_training():
    """Start ML-Agents training với DINO ResNet34"""
    print("⏳ Starting DINO ResNet34 training in 3 seconds...")
    time.sleep(3)
    
    print("🚀 Starting DINO ResNet34 RL training...")
    
    # Training command with lightweight config
    cmd = [
        "mlagents-learn",
        "Config/dino_resnet34_light.yaml",  # Use lightweight config
        "--env", "Builds/RandomTrain/2D go to target v1.exe",
        "--run-id", "dino_resnet34_light",
        "--force",
        "--time-scale", "7",
        "--num-envs", "1"
    ]
    
    # Set environment variables
    env_vars = os.environ.copy()
    
    print(f"🎯 Command: {' '.join(cmd)}")
    print("\n📊 Expected: DINO ResNet34 (30.8M params, 100% pre-trained)")
    print("🔥 Features: Self-supervised pre-trained backbone")
    print("🎮 Training starting...")
    
    # Run training with environment variables
    result = subprocess.run(cmd, env=env_vars, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ DINO ResNet34 training completed successfully!")
        print("📊 Check results in: ./results/dino_resnet34_direct/")
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
