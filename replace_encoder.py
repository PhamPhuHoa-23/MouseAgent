#!/usr/bin/env python3
"""
Script để replace encoder trong ML-Agents
"""

import shutil
import os
from pathlib import Path

def find_mlagents_encoders():
    """Tìm path của mlagents encoders.py"""
    import mlagents
    mlagents_path = Path(mlagents.__file__).parent
    encoders_path = mlagents_path / "trainers" / "torch" / "encoders.py"
    
    if encoders_path.exists():
        return str(encoders_path)
    else:
        # Fallback search
        possible_paths = [
            Path("C:/Users/admin/miniconda3/envs/mouse_dinov3_basic/lib/site-packages/mlagents/trainers/torch/encoders.py"),
            Path("C:/Users/admin/miniconda3/lib/site-packages/mlagents/trainers/torch/encoders.py"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
    
    return None

def backup_original_encoder(encoders_path):
    """Backup file gốc"""
    backup_path = encoders_path + ".backup"
    if not Path(backup_path).exists():
        print(f"📁 Backing up original encoders.py to {backup_path}")
        shutil.copy2(encoders_path, backup_path)
    else:
        print(f"📁 Backup already exists: {backup_path}")

def create_ml_agents_compatible_encoder():
    """Tạo encoder tương thích với ML-Agents"""
    
    encoder_code = '''"""
DINOv3 ViT-S Encoder for ML-Agents (Compatible Version)
Replaces NatureVisualEncoder in ML-Agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ML-Agents imports
from mlagents.trainers.torch.layers import linear_layer
from mlagents.trainers.settings import Initialization  
from mlagents.trainers.torch import exporting_to_onnx

# Try to import transformers, fallback if not available
try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using fallback encoder")


class DINOv3VisualEncoder(nn.Module):
    """DINOv3 ViT-S Visual Encoder compatible with ML-Agents"""
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.height = height
        self.width = width
        self.initial_channels = initial_channels
        
        if TRANSFORMERS_AVAILABLE:
            self._init_dinov3()
        else:
            self._init_fallback()
        
        self._init_architecture()
    
    def _init_dinov3(self):
        """Initialize DINOv3 model"""
        try:
            self.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            print(f"🚀 Loading DINOv3 ViT-S for ML-Agents...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.dinov3_model = AutoModel.from_pretrained(self.model_name)
            
            # Freeze backbone
            for param in self.dinov3_model.parameters():
                param.requires_grad = False
                
            self.embed_dim = self.dinov3_model.config.hidden_size  # 384
            self.use_dinov3 = True
            
            print(f"✅ DINOv3 loaded! Embedding dim: {self.embed_dim}")
            
        except Exception as e:
            print(f"⚠️ DINOv3 loading failed: {e}")
            print("🔄 Falling back to simple encoder...")
            self._init_fallback()
    
    def _init_fallback(self):
        """Fallback to simple CNN if DINOv3 fails"""
        self.embed_dim = 384
        self.use_dinov3 = False
        
        # Simple CNN fallback
        from mlagents.trainers.torch.utils import conv_output_shape
        
        conv_1_hw = conv_output_shape((self.height, self.width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2) 
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        final_flat = conv_3_hw[0] * conv_3_hw[1] * 64
        
        self.fallback_cnn = nn.Sequential(
            nn.Conv2d(self.initial_channels, 32, [8, 8], [4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, [4, 4], [2, 2]), 
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, [3, 3], [1, 1]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(final_flat, self.embed_dim),
            nn.LeakyReLU()
        )
    
    def _init_architecture(self):
        """Initialize user's specified architecture"""
        # CLS → Linear → Softmax → 2 Linear+ReLU → Linear cuối
        reduced_dim = max(128, self.h_size * 4)
        hidden_dim = max(64, self.h_size * 2)
        
        self.dim_reduction = nn.Linear(self.embed_dim, reduced_dim)
        
        self.feature_layers = nn.Sequential(
            nn.Linear(reduced_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output_layer = nn.Sequential(
            linear_layer(
                hidden_dim,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )
    
    def _convert_to_pil_images(self, visual_obs: torch.Tensor) -> list:
        """Convert ML-Agents tensor to PIL images"""
        batch_size = visual_obs.shape[0]
        images = []
        
        for i in range(batch_size):
            img = visual_obs[i]  # HWC format after permutation
            img_np = img.cpu().numpy()
            
            # Handle grayscale -> RGB
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            
            # Scale to 0-255
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            images.append(Image.fromarray(img_np))
        
        return images
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Handle ML-Agents format conversion
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])  # BHWC -> BCHW
        
        if self.use_dinov3:
            try:
                # Convert back to BHWC for PIL processing
                bhwc_obs = visual_obs.permute([0, 2, 3, 1])  # BCHW -> BHWC
                images = self._convert_to_pil_images(bhwc_obs)
                
                # Process through DINOv3
                inputs = self.processor(images, return_tensors="pt")
                inputs = {k: v.to(visual_obs.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.dinov3_model(**inputs)
                    cls_features = outputs.last_hidden_state[:, 0]  # CLS token
                    
            except Exception as e:
                print(f"⚠️ DINOv3 forward error: {e}")
                # Fallback to CNN
                hidden = self.fallback_cnn(visual_obs)
                cls_features = hidden
        else:
            # Use fallback CNN
            cls_features = self.fallback_cnn(visual_obs)
        
        # User's architecture
        reduced = self.dim_reduction(cls_features)
        soft_features = F.softmax(reduced, dim=-1)
        processed = self.feature_layers(soft_features)
        output = self.output_layer(processed)
        
        return output


# Replace NatureVisualEncoder với DINOv3 version
NatureVisualEncoder = DINOv3VisualEncoder
'''
    
    return encoder_code

def replace_nature_visual_encoder(target_encoders_path, custom_encoder_path=None):
    """Replace NatureVisualEncoder trong ML-Agents"""
    
    print(f"🔧 Replacing NatureVisualEncoder in ML-Agents...")
    print(f"   Target: {target_encoders_path}")
    
    # Backup original
    backup_original_encoder(target_encoders_path)
    
    if custom_encoder_path and Path(custom_encoder_path).exists():
        # Use custom encoder file
        print(f"📁 Using custom encoder: {custom_encoder_path}")
        with open(custom_encoder_path, 'r') as f:
            encoder_code = f.read()
    else:
        # Use built-in DINOv3 encoder
        encoder_code = create_ml_agents_compatible_encoder()
    
    # Read original encoders.py
    with open(target_encoders_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Find NatureVisualEncoder class and replace it
    import re
    
    # Pattern to find the class definition
    pattern = r'class NatureVisualEncoder\(.*?\n(?:.*\n)*?(?=\n\nclass|\nclass|\Z)'
    
    if re.search(pattern, original_content):
        # Replace the class
        modified_content = re.sub(pattern, encoder_code, original_content)
    else:
        # Append at the end if not found
        modified_content = original_content + "\n\n" + encoder_code
    
    # Write modified content
    with open(target_encoders_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ NatureVisualEncoder replaced successfully!")
    print(f"💡 Backup available at: {target_encoders_path}.backup")

if __name__ == "__main__":
    print("🔧 ML-Agents Encoder Replacement Tool")
    
    # Find encoders.py
    encoders_path = find_mlagents_encoders()
    
    if encoders_path:
        print(f"📍 Found ML-Agents encoders.py: {encoders_path}")
        
        # Replace encoder
        replace_nature_visual_encoder(encoders_path)
        
        print(f"🎯 Ready for training with DINOv3!")
        
    else:
        print(f"❌ Could not find ML-Agents encoders.py")
        print(f"💡 Please check your ML-Agents installation")
