#!/usr/bin/env python3
"""
VANP Trainer với Synthesized Data
Train VANP model sử dụng data từ trained models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet34
import json
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


class VANPDataset(Dataset):
    """Dataset for VANP training"""
    
    def __init__(self, dataset_file, episodes_dir, transform=None, max_samples=None):
        self.episodes_dir = Path(episodes_dir)
        self.transform = transform
        
        # Load VANP samples
        self.samples = []
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line.strip())
                self.samples.append(sample)
        
        print(f"📊 Loaded {len(self.samples)} VANP samples")
        
        # Statistics by model
        model_counts = {}
        for sample in self.samples:
            model = sample['architecture']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        print("   Distribution by architecture:")
        for model, count in model_counts.items():
            print(f"     {model}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, episode_id, model_name, frame_id):
        """Load image từ episode directory"""
        episode_dir = self.episodes_dir / f"{model_name}_{episode_id:04d}"
        img_path = episode_dir / f"frame_{frame_id:06d}.png"
        
        if not img_path.exists():
            # Return dummy image if not found
            return Image.new('L', (155, 86), color='black')
        
        img = Image.open(img_path).convert('L')
        return img
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        episode_id = sample['episode_id']
        model_name = sample['model_name']
        
        # Load visual history (past frames)
        visual_history = []
        for frame_info in sample['visual_history']:
            frame_id = frame_info['frame_id']
            img = self.load_image(episode_id, model_name, frame_id)
            
            if self.transform:
                img = self.transform(img)
            
            visual_history.append(img)
        
        # Stack visual history
        visual_history = torch.stack(visual_history)  # (tau_p, C, H, W)
        
        # Load goal image
        goal_frame_id = sample['goal_frame']['frame_id']
        goal_img = self.load_image(episode_id, model_name, goal_frame_id)
        if self.transform:
            goal_img = self.transform(goal_img)
        
        # Process future actions
        future_actions = sample['future_actions']
        
        # Pad or truncate to fixed length
        max_actions = 12  # tau_f
        if len(future_actions) < max_actions:
            # Pad với zeros
            padding = [[0.0, 0.0, 0.0]] * (max_actions - len(future_actions))
            future_actions.extend(padding)
        else:
            future_actions = future_actions[:max_actions]
        # print(future_actions)
        future_actions = torch.tensor(future_actions, dtype=torch.float32)
        # if future_actions.size(0) == 1: 
        #     future_actions = future_actions.unsqueeze(0)
        return {
            'visual_history': visual_history,     # (tau_p, C, H, W)
            'future_actions': future_actions,     # (tau_f, action_dim)
            'goal_image': goal_img,               # (C, H, W)
            'model_name': sample['model_name'],
            'architecture': sample['architecture']
        }


# class VANPEncoder(nn.Module):
#     """VANP Visual Encoder sử dụng ResNet34"""
    
#     def __init__(self, output_dim=512):
#         super().__init__()
        
#         # ResNet34 backbone
#         resnet = resnet34(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
#         # Adaptive pooling
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Projection head
#         self.projection = nn.Sequential(
#             nn.Linear(512, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim)
#         )
        
#         self.output_dim = output_dim
    
#     def forward(self, x):
#         """
#         Args:
#             x: (B, C, H, W) hoặc (B, T, C, H, W)
#         Returns:
#             features: (B, output_dim) hoặc (B, T, output_dim)
#         """
#         original_shape = x.shape
        
#         # Reshape để process qua ResNet
#         if len(x.shape) == 5:  # (B, T, C, H, W)
#             B, T, C, H, W = x.shape
#             x = x.view(B * T, C, H, W)
        
#         # Extract features
#         features = self.backbone(x)  # (B*T, 512, h, w)
#         features = self.adaptive_pool(features)  # (B*T, 512, 1, 1)
#         features = features.flatten(1)  # (B*T, 512)
        
#         # Project
#         features = self.projection(features)  # (B*T, output_dim)
        
#         # Reshape back
#         if len(original_shape) == 5:
#             features = features.view(B, T, -1)
        
#         return features
# import torch
# import torch.nn as nn
# import torch.nn.init as init
# from mlagents.trainers.torch.model_serialization import exporting_to_onnx
# import os
# import random
# import numpy as np
# from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
# from pathlib import Path

# class LightWeatherSimulation:
#     def __init__(self):
#         self.weather_types = ['light_dim', 'very_light_fog']
    
#     def apply_light_dim_effect(self, image):
#         """Áp dụng hiệu ứng làm tối nhẹ thay vì night effect mạnh"""
#         enhancer = ImageEnhance.Brightness(image)
#         dimmed_image = enhancer.enhance(0.85)  # Nhẹ hơn: 0.85 thay vì 0.6
#         return dimmed_image
    
#     def apply_very_light_fog_effect(self, image):
#         """Fog effect rất nhẹ, chỉ ở các góc"""
#         img_array = np.array(image).astype(np.float32)
#         H, W = img_array.shape
        
#         center_x, center_y = W//2, H//2
#         y_coords, x_coords = np.ogrid[:H, :W]
        
#         distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
#         max_distance = np.sqrt(center_x**2 + center_y**2)
        
#         # Fog intensity nhẹ hơn nhiều: 0.15 max thay vì 0.4
#         fog_intensity = (distances / max_distance) * 0.1 + 0.02
        
#         fog_color = 190
#         fogged = img_array * (1 - fog_intensity) + fog_color * fog_intensity
#         fogged = np.clip(fogged, 0, 255)
        
#         return Image.fromarray(fogged.astype(np.uint8))

# class LightColorJitterAugmentation:
#     def __init__(self, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1)):
#         """Giảm range từ (0.7, 1.4) xuống (0.9, 1.1)"""
#         self.brightness_range = brightness_range
#         self.contrast_range = contrast_range
    
#     def apply_jitter(self, image):
#         jittered = image.copy()
        
#         brightness_factor = random.uniform(*self.brightness_range)
#         enhancer = ImageEnhance.Brightness(jittered)
#         jittered = enhancer.enhance(brightness_factor)
        
#         contrast_factor = random.uniform(*self.contrast_range)
#         enhancer = ImageEnhance.Contrast(jittered)
#         jittered = enhancer.enhance(contrast_factor)
        
#         return jittered, f"B:{brightness_factor:.2f},C:{contrast_factor:.2f}"

# class LightAugmentationPipeline:
#     def __init__(self):
#         self.weather_sim = LightWeatherSimulation()
#         self.color_jitter = LightColorJitterAugmentation()
        
#     def apply_light_augmentation(self, image):
#         """Augmentation pipeline nhẹ cho navigation task"""
#         augmented = image.copy()
#         aug_info = []
        
#         # Giảm probability của weather effects: 10% thay vì 25%
#         weather_chance = random.random()
#         if weather_chance < 0.05:  # 5% thay vì 25%
#             if random.random() < 0.5:
#                 augmented = self.weather_sim.apply_light_dim_effect(augmented)
#                 aug_info.append("Light dim")
#             else:
#                 augmented = self.weather_sim.apply_very_light_fog_effect(augmented)
#                 aug_info.append("Very light fog")
        
#         # Tăng probability của color jittering nhẹ: 50% thay vì 30-40%
#         if random.random() < 0.5:
#             augmented, jitter_info = self.color_jitter.apply_jitter(augmented)
#             aug_info.append("Light jitter")
            
#         return augmented, " + ".join(aug_info) if aug_info else "No augmentation"

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

# def pool_out_shape(input_shape, kernel_size, stride=2, padding=0):
#     h, w = input_shape
#     h_out = (h - kernel_size + 2 * padding) // stride + 1
#     w_out = (w - kernel_size + 2 * padding) // stride + 1
#     return h_out, w_out

# class ResNetBlock(nn.Module):
#     def __init__(self, channel: int):
#         super().__init__()
#         self.layers = nn.Sequential(
#             Swish(),
#             nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
#             Swish(),
#             nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
#             # Swish(),
#             # nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
#             # Swish(),
#             # nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
#             # Swish(),
#             # nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
#         )

#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         return input_tensor + self.layers(input_tensor)

# class VANPEncoder(nn.Module):
#     def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
#         super().__init__()
#         self.embed_dim = output_size
#         self.use_augmentation = True
#         self.height = height
#         self.width = width
        
#         # Sử dụng light augmentation pipeline
#         self.augmentation_pipeline = LightAugmentationPipeline()
        
#         n_channels = [16, 32, 32]
#         n_blocks = 2
#         layers = []
#         last_channel = initial_channels
#         current_height, current_width = height, width

#         for channel in n_channels:
#             layers.append(nn.Conv2d(last_channel, channel, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
#             current_height, current_width = pool_out_shape((current_height, current_width), kernel_size=3)
#             for _ in range(n_blocks):
#                 layers.append(ResNetBlock(channel))
#             last_channel = channel

#         layers.append(Swish())
#         self.final_flat_size = n_channels[-1] * current_height * current_width

#         self.dense = nn.Linear(self.final_flat_size, output_size)
#         init.kaiming_normal_(self.dense.weight, mode='fan_out', nonlinearity='relu', a=1.41)
#         if self.dense.bias is not None:
#             init.zeros_(self.dense.bias)

#         self.sequential = nn.Sequential(*layers)
        
#         # Load checkpoint if available
#         self._load_checkpoint_if_exists()
    
#     def _load_checkpoint_if_exists(self):
#         """Try to load checkpoint from common paths"""
#         pass
#         # possible_paths = [
#         #     "./resnet_aug/best_robust_agent.pth",
#         #     "./resnet_aug/best_robust_agent_naturevisualencoder.pth",
#         #     "C:/Users/admin/Neurips2025/MouseVsAI/mouse_vs_ai_windows/resnet_aug/best_robust_agent_naturevisualencoder.pth"
#         # ]
        
#         # for checkpoint_path in possible_paths:
#         #     if Path(checkpoint_path).exists():
#         #         try:
#         #             checkpoint = torch.load(checkpoint_path, map_location='cpu')
#         #             pretrained_state_dict = checkpoint['model_state_dict']
                    
#         #             backbone_weights = {}
#         #             for key, value in pretrained_state_dict.items():
#         #                 if key.startswith('backbone.'):
#         #                     new_key = key[9:]
#         #                     backbone_weights[new_key] = value
                    
#         #             if backbone_weights:
#         #                 missing_keys, unexpected_keys = self.load_state_dict(backbone_weights, strict=False)
#         #                 print(f"✅ Loaded checkpoint: {checkpoint_path}")
#         #                 return
#         #         except Exception as e:
#         #             print(f"Failed to load {checkpoint_path}: {e}")
#         #             continue

#     def _apply_light_augmentation_to_batch(self, visual_obs):
#         """Áp dụng augmentation nhẹ cho batch"""
#         if not self.training or not self.use_augmentation:
#             return visual_obs
        
#         batch_size = visual_obs.shape[0]
        
#         if visual_obs.shape[-1] == 1:
#             np_batch = visual_obs.cpu().numpy()
#             augmented_batch = []
            
#             for i in range(batch_size):
#                 img_array = np_batch[i, :, :, 0]
#                 img_pil = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                
#                 # Giảm probability từ 40% xuống 25%
#                 if random.random() < 0.25:
#                     augmented_pil, _ = self.augmentation_pipeline.apply_light_augmentation(img_pil)
#                 else:
#                     augmented_pil = img_pil
                
#                 augmented_array = np.array(augmented_pil).astype(np.float32) / 255.0
#                 augmented_array = np.expand_dims(augmented_array, axis=-1)
#                 augmented_batch.append(augmented_array)
            
#             augmented_batch = np.stack(augmented_batch, axis=0)
#             return torch.from_numpy(augmented_batch).to(visual_obs.device)
        
#         else:
#             if visual_obs.shape[1] == 1:
#                 np_batch = visual_obs.cpu().numpy()
#                 augmented_batch = []
                
#                 for i in range(batch_size):
#                     img_array = np_batch[i, 0, :, :]
#                     img_pil = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                    
#                     # Giảm probability từ 40% xuống 25%
#                     if random.random() < 0.25:
#                         augmented_pil, _ = self.augmentation_pipeline.apply_light_augmentation(img_pil)
#                     else:
#                         augmented_pil = img_pil
                    
#                     augmented_array = np.array(augmented_pil).astype(np.float32) / 255.0
#                     augmented_array = np.expand_dims(augmented_array, axis=0)
#                     augmented_batch.append(augmented_array)
                
#                 augmented_batch = np.stack(augmented_batch, axis=0)
#                 return torch.from_numpy(augmented_batch).to(visual_obs.device)
        
#         return visual_obs

#     def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
#         # Bỏ comment để sử dụng augmentation trong training
#         # if self.training and self.use_augmentation:
#         #     visual_obs = self._apply_light_augmentation_to_batch(visual_obs)
#         # print(visual_obs.shape)
#         # if not exporting_to_onnx.is_exporting():
#         #     visual_obs = visual_obs.permute(0, 3, 1, 2)
#         original_shape = visual_obs.shape
#         # print(visual_obs.max())

#         if len(visual_obs.shape) == 5:  # (B, T, C, H, W)
#             B, T, C, H, W = visual_obs.shape
#             visual_obs = visual_obs.view(B * T, C, H, W)
#         hidden = self.sequential(visual_obs)
#         before_out = hidden.reshape(-1, self.final_flat_size)
#         features = torch.relu(self.dense(before_out))
#         if len(original_shape) == 5:
#             features = features.view(B, T, -1)
        
#         return features
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# FiLM module for conditioning RNN with CNN features
class FiLMLayer(nn.Module):
    def __init__(self, cnn_channels: int, rnn_channels: int):
        super().__init__()
        self.gamma_proj = nn.Conv2d(cnn_channels, rnn_channels, 1)  # Scale
        self.beta_proj = nn.Conv2d(cnn_channels, rnn_channels, 1)   # Bias
        
    def forward(self, cnn_features: torch.Tensor, rnn_features: torch.Tensor) -> torch.Tensor:
        # Resize CNN features to match RNN spatial dimensions if needed
        if cnn_features.shape[-2:] != rnn_features.shape[-2:]:
            cnn_features = F.interpolate(cnn_features, size=rnn_features.shape[-2:], mode='bilinear', align_corners=False)
        
        gamma = self.gamma_proj(cnn_features)
        beta = self.beta_proj(cnn_features)
        return gamma * rnn_features + beta

# ConvLSTM Cell (ONNX compatible)
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Input-to-hidden convolutions
        self.W_xi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_xf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_xo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_xg = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        
        # Hidden-to-hidden convolutions
        self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hg = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden
        
        # Gates computation
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h_prev))  # Input gate
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h_prev))  # Forget gate
        o = torch.sigmoid(self.W_xo(x) + self.W_ho(h_prev))  # Output gate
        g = torch.tanh(self.W_xg(x) + self.W_hg(h_prev))     # New info
        
        # Cell state update
        c_new = f * c_prev + i * g
        
        # Hidden state update
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)

# ConvGRU Cell (Alternative, lighter than LSTM)
class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Reset gate
        self.W_xr = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hr = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        
        # Update gate
        self.W_xz = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hz = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        
        # New state
        self.W_xh = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hh = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        h_prev = hidden
        
        # Gates
        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h_prev))  # Reset gate
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h_prev))  # Update gate
        
        # New hidden state candidate
        h_tilde = torch.tanh(self.W_xh(x) + self.W_hh(r * h_prev))
        
        # Final hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        
        return h_new

# ResNet block for MLAgents style
class ResNetBlock(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1)
        self.conv2 = nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return F.relu(out)

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

# from mlagents.torch_utils import linear_layer
# from mlagents.trainers.models.utils import conv_output_shape
from mlagents.trainers.torch.layers import linear_layer
from mlagents.trainers.torch.encoders import conv_output_shape, pool_out_shape

# Simple Visual Encoder with CNN-RNN parallel branches
# class VANPEncoder(nn.Module):
#     def __init__(
#         self, height: int, width: int, initial_channels: int, output_size: int
#     ):
#         super().__init__()
#         self.h_size = output_size
        
#         # Calculate output shapes for CNN branch
#         conv_1_hw = conv_output_shape((height, width), 8, 4)
#         conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
#         cnn_final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        
#         # CNN Branch
#         self.cnn_conv1 = nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
#         self.cnn_conv2 = nn.Conv2d(16, 32, [4, 4], [2, 2])
        
#         # RNN Branch (using ConvGRU for simplicity)
#         self.rnn_downsample1 = nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
#         self.rnn_gru1 = ConvGRUCell(16, 16, kernel_size=3)
        
#         self.rnn_downsample2 = nn.Conv2d(16, 32, [4, 4], [2, 2])
#         self.rnn_gru2 = ConvGRUCell(32, 32, kernel_size=3)
        
#         # FiLM conditioning layers
#         self.film1 = FiLMLayer(16, 16)
#         self.film2 = FiLMLayer(32, 32)
        
#         # Initialize hidden states (will be computed dynamically)
#         self.hidden1 = None
#         self.hidden2 = None
        
#         # Final fusion and output
#         total_flat = cnn_final_flat + conv_2_hw[0] * conv_2_hw[1] * 32  # RNN same size as CNN
#         self.dense = nn.Sequential(
#             linear_layer(
#                 total_flat,
#                 self.h_size,
#                 kernel_gain=1.41,
#             ),
#             nn.LeakyReLU(),
#         )

#     def _init_hidden(self, x: torch.Tensor, channels: int, h: int, w: int) -> torch.Tensor:
#         """Initialize hidden state for GRU"""
#         return torch.zeros(x.size(0), channels, h, w, device=x.device, dtype=x.dtype)

#     def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
#         # if not exporting_to_onnx.is_exporting():
#         #     visual_obs = visual_obs.permute([0, 3, 1, 2])
#         original_shape = visual_obs.shape
 
#         if len(visual_obs.shape) == 5:  # (B, T, C, H, W)
#             B, T, C, H, W = visual_obs.shape
#             visual_obs = visual_obs.view(B * T, C, H, W)
        
#         # CNN Branch
#         cnn_1 = F.leaky_relu(self.cnn_conv1(visual_obs))
#         cnn_2 = F.leaky_relu(self.cnn_conv2(cnn_1))
        
#         # RNN Branch
#         rnn_down1 = F.leaky_relu(self.rnn_downsample1(visual_obs))
        
#         # Initialize hidden state for first GRU layer
#         if self.hidden1 is None or self.hidden1.size(0) != rnn_down1.size(0):
#             self.hidden1 = self._init_hidden(rnn_down1, 16, rnn_down1.size(2), rnn_down1.size(3))
        
#         rnn_1 = self.rnn_gru1(rnn_down1, self.hidden1)
#         self.hidden1 = rnn_1.detach()  # Detach to prevent gradient flow through time
        
#         # Apply FiLM conditioning
#         rnn_1_refined = self.film1(cnn_1, rnn_1)
        
#         rnn_down2 = F.leaky_relu(self.rnn_downsample2(rnn_1_refined))
        
#         # Initialize hidden state for second GRU layer
#         if self.hidden2 is None or self.hidden2.size(0) != rnn_down2.size(0):
#             self.hidden2 = self._init_hidden(rnn_down2, 32, rnn_down2.size(2), rnn_down2.size(3))
        
#         rnn_2 = self.rnn_gru2(rnn_down2, self.hidden2)
#         self.hidden2 = rnn_2.detach()  # Detach to prevent gradient flow through time
        
#         # Apply FiLM conditioning
#         rnn_2_refined = self.film2(cnn_2, rnn_2)
        
#         # Flatten and concatenate
#         cnn_flat = cnn_2.reshape(cnn_2.size(0), -1)
#         rnn_flat = rnn_2_refined.reshape(rnn_2_refined.size(0), -1)
#         combined = torch.cat([cnn_flat, rnn_flat], dim=1)
        
#         if len(original_shape) == 5:
#             combined = combined.view(B, T, -1)
            
#         return self.dense(combined)

class VANPEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        
        # Calculate dimensions through the network
        conv1_hw = conv_output_shape((height, width), 3, 1, 1)
        pool1_hw = pool_out_shape(conv1_hw, 3)
        
        conv2_hw = conv_output_shape(pool1_hw, 3, 1, 1)
        pool2_hw = pool_out_shape(conv2_hw, 3)
        
        conv3_hw = conv_output_shape(pool2_hw, 3, 1, 1)
        pool3_hw = pool_out_shape(conv3_hw, 3)
        
        final_flat_size = 32 * pool3_hw[0] * pool3_hw[1]
        
        # CNN Branch - Block 1
        self.cnn_conv1 = nn.Conv2d(initial_channels, 16, [3, 3], [1, 1], padding=1)
        self.cnn_pool1 = nn.MaxPool2d([3, 3], [2, 2])
        self.cnn_res1_1 = ResNetBlock(16)
        self.cnn_res1_2 = ResNetBlock(16)
        
        # CNN Branch - Block 2
        self.cnn_conv2 = nn.Conv2d(16, 32, [3, 3], [1, 1], padding=1)
        self.cnn_pool2 = nn.MaxPool2d([3, 3], [2, 2])
        self.cnn_res2_1 = ResNetBlock(32)
        self.cnn_res2_2 = ResNetBlock(32)
        
        # CNN Branch - Block 3
        self.cnn_conv3 = nn.Conv2d(32, 32, [3, 3], [1, 1], padding=1)
        self.cnn_pool3 = nn.MaxPool2d([3, 3], [2, 2])
        self.cnn_res3_1 = ResNetBlock(32)
        self.cnn_res3_2 = ResNetBlock(32)
        self.cnn_swish = Swish()
        
        # RNN Branch - Block 1
        self.rnn_conv1 = nn.Conv2d(initial_channels, 16, [3, 3], [1, 1], padding=1)
        self.rnn_pool1 = nn.MaxPool2d([3, 3], [2, 2])
        self.rnn_lstm1 = ConvLSTMCell(16, 16, kernel_size=3)
        self.rnn_res1_1 = ResNetBlock(16)
        self.rnn_res1_2 = ResNetBlock(16)
        
        # RNN Branch - Block 2
        self.rnn_conv2 = nn.Conv2d(16, 32, [3, 3], [1, 1], padding=1)
        self.rnn_pool2 = nn.MaxPool2d([3, 3], [2, 2])
        self.rnn_lstm2 = ConvLSTMCell(32, 32, kernel_size=3)
        self.rnn_res2_1 = ResNetBlock(32)
        self.rnn_res2_2 = ResNetBlock(32)
        
        # RNN Branch - Block 3
        self.rnn_conv3 = nn.Conv2d(32, 32, [3, 3], [1, 1], padding=1)
        self.rnn_pool3 = nn.MaxPool2d([3, 3], [2, 2])
        self.rnn_lstm3 = ConvLSTMCell(32, 32, kernel_size=3)
        self.rnn_res3_1 = ResNetBlock(32)
        self.rnn_res3_2 = ResNetBlock(32)
        self.rnn_swish = Swish()
        
        # FiLM conditioning modules
        self.film1 = FiLMLayer(16, 16)
        self.film2 = FiLMLayer(32, 32)
        self.film3 = FiLMLayer(32, 32)
        
        # LSTM hidden states
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        
        # Final dense layer
        total_features = final_flat_size * 2  # CNN + RNN
        self.dense = linear_layer(
            total_features,
            output_size,
            kernel_gain=1.41,
        )

    def _init_lstm_hidden(self, x: torch.Tensor, channels: int, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states for LSTM"""
        hidden = torch.zeros(x.size(0), channels, h, w, device=x.device, dtype=x.dtype)
        cell = torch.zeros(x.size(0), channels, h, w, device=x.device, dtype=x.dtype)
        return (hidden, cell)

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # if not exporting_to_onnx.is_exporting():
        #     visual_obs = visual_obs.permute([0, 3, 1, 2])
        original_shape = visual_obs.shape
 
        if len(visual_obs.shape) == 5:  # (B, T, C, H, W)
            B, T, C, H, W = visual_obs.shape
            visual_obs = visual_obs.view(B * T, C, H, W)
        # Block 1 - CNN
        cnn_x = self.cnn_conv1(visual_obs)
        cnn_x = self.cnn_pool1(cnn_x)
        cnn_x = self.cnn_res1_1(cnn_x)
        cnn_x = self.cnn_res1_2(cnn_x)
        cnn_block1 = cnn_x
        
        # Block 1 - RNN
        rnn_x = self.rnn_conv1(visual_obs)
        rnn_x = self.rnn_pool1(rnn_x)
        
        if self.hidden1 is None or self.hidden1[0].size(0) != rnn_x.size(0):
            self.hidden1 = self._init_lstm_hidden(rnn_x, 16, rnn_x.size(2), rnn_x.size(3))
        
        rnn_x, self.hidden1 = self.rnn_lstm1(rnn_x, self.hidden1)
        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        
        rnn_x = self.film1(cnn_block1, rnn_x)  # FiLM conditioning
        rnn_x = self.rnn_res1_1(rnn_x)
        rnn_x = self.rnn_res1_2(rnn_x)
        
        # Block 2 - CNN
        cnn_x = self.cnn_conv2(cnn_x)
        cnn_x = self.cnn_pool2(cnn_x)
        cnn_x = self.cnn_res2_1(cnn_x)
        cnn_x = self.cnn_res2_2(cnn_x)
        cnn_block2 = cnn_x
        
        # Block 2 - RNN
        rnn_x = self.rnn_conv2(rnn_x)
        rnn_x = self.rnn_pool2(rnn_x)
        
        if self.hidden2 is None or self.hidden2[0].size(0) != rnn_x.size(0):
            self.hidden2 = self._init_lstm_hidden(rnn_x, 32, rnn_x.size(2), rnn_x.size(3))
        
        rnn_x, self.hidden2 = self.rnn_lstm2(rnn_x, self.hidden2)
        self.hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())
        
        rnn_x = self.film2(cnn_block2, rnn_x)  # FiLM conditioning
        rnn_x = self.rnn_res2_1(rnn_x)
        rnn_x = self.rnn_res2_2(rnn_x)
        
        # Block 3 - CNN
        cnn_x = self.cnn_conv3(cnn_x)
        cnn_x = self.cnn_pool3(cnn_x)
        cnn_x = self.cnn_res3_1(cnn_x)
        cnn_x = self.cnn_res3_2(cnn_x)
        cnn_x = self.cnn_swish(cnn_x)
        cnn_block3 = cnn_x
        
        # Block 3 - RNN
        rnn_x = self.rnn_conv3(rnn_x)
        rnn_x = self.rnn_pool3(rnn_x)
        
        if self.hidden3 is None or self.hidden3[0].size(0) != rnn_x.size(0):
            self.hidden3 = self._init_lstm_hidden(rnn_x, 32, rnn_x.size(2), rnn_x.size(3))
        
        rnn_x, self.hidden3 = self.rnn_lstm3(rnn_x, self.hidden3)
        self.hidden3 = (self.hidden3[0].detach(), self.hidden3[1].detach())
        
        rnn_x = self.film3(cnn_block3, rnn_x)  # FiLM conditioning
        rnn_x = self.rnn_res3_1(rnn_x)
        rnn_x = self.rnn_res3_2(rnn_x)
        rnn_x = self.rnn_swish(rnn_x)
        
        # Flatten and combine
        cnn_flat = cnn_x.reshape(-1, cnn_x.size(1) * cnn_x.size(2) * cnn_x.size(3))
        rnn_flat = rnn_x.reshape(-1, rnn_x.size(1) * rnn_x.size(2) * rnn_x.size(3))
        combined = torch.cat([cnn_flat, rnn_flat], dim=1)
        
        if len(original_shape) == 5:
            combined = combined.view(B, T, -1)
            
        return torch.relu(self.dense(combined))

class VANPModel(nn.Module):
    """Complete VANP model"""
    
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, action_dim=3, tau_p=3, tau_f=12):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.tau_p = tau_p
        self.tau_f = tau_f
        
        # Visual encoder
        self.visual_encoder = VANPEncoder(
            height=155,
            width=86,
            initial_channels=1,
            output_size=embed_dim
        )
        
        # Temporal transformer for visual history
        self.visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Context token for visual transformer
        self.visual_context_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Action transformer
        self.action_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Context token for action transformer
        self.action_context_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Projection heads for VICReg
        self.visual_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.action_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.goal_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, visual_history, future_actions, goal_image):
        """
        Args:
            visual_history: (B, tau_p, C, H, W)
            future_actions: (B, tau_f, action_dim)
            goal_image: (B, C, H, W)
        Returns:
            embeddings: Dict with visual, action, goal embeddings
        """
        B = visual_history.shape[0]
        
        # Encode visual history
        visual_features = self.visual_encoder(visual_history)  # (B, tau_p, embed_dim)
        
        # Add context token
        visual_context = self.visual_context_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        visual_input = torch.cat([visual_context, visual_features], dim=1)  # (B, tau_p+1, embed_dim)
        
        # Apply transformer
        visual_output = self.visual_transformer(visual_input)
        visual_embedding = visual_output[:, 0]  # Context token (B, embed_dim)
        
        # Encode future actions
        action_features = self.action_encoder(future_actions)  # (B, tau_f, embed_dim)
        
        # Add context token
        action_context = self.action_context_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        # print("action_context:", action_context.shape)
        # print("action_features:", action_features.shape)
        action_features = action_features.squeeze(2)   # (16, 20, 512)

        action_input = torch.cat([action_context, action_features], dim=1)  # (B, tau_f+1, embed_dim)
        
        # Apply transformer
        action_output = self.action_transformer(action_input)
        action_embedding = action_output[:, 0]  # Context token (B, embed_dim)
        
        # Encode goal image
        goal_features = self.visual_encoder(goal_image)  # (B, embed_dim)
        
        # Project embeddings
        visual_proj = self.visual_proj(visual_embedding)
        action_proj = self.action_proj(action_embedding)
        goal_proj = self.goal_proj(goal_features)
        
        return {
            'visual': visual_proj,
            'action': action_proj,
            'goal': goal_proj
        }


class VICRegLoss(nn.Module):
    """VICReg Loss cho VANP"""
    
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    
    def forward(self, z1, z2):
        """
        VICReg loss giữa 2 embeddings
        """
        B, D = z1.shape
        
        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)
        
        # Variance loss (prevent collapse)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # Covariance loss (decorrelation)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)
        
        cov_loss = self.off_diagonal(cov_z1).pow(2).sum() / D
        cov_loss += self.off_diagonal(cov_z2).pow(2).sum() / D
        
        # Total loss
        loss = (
            self.sim_coeff * sim_loss +
            self.std_coeff * std_loss +
            self.cov_coeff * cov_loss
        )
        
        return loss, {
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item()
        }
    
    def off_diagonal(self, x):
        """Remove diagonal elements"""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VANPTrainer:
    """VANP Trainer class"""
    
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # Loss function
        self.vicreg_loss = VICRegLoss()
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'visual_goal_loss': [],
            'visual_action_loss': [],
            'lr': []
        }

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint để resume training"""
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint không tồn tại: {checkpoint_path}")
            return 0
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            
            start_epoch = checkpoint['epoch'] + 1
            last_loss = checkpoint['loss']
            
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            print(f"   Resume từ epoch: {start_epoch}")
            print(f"   Last loss: {last_loss:.4f}")
            
            return start_epoch
            
        except Exception as e:
            print(f"❌ Lỗi khi load checkpoint: {e}")
            return 0
        
    def train_epoch(self, dataloader, epoch, lambda_weight=0.5):
        """Train một epoch"""
        self.model.train()
        
        total_loss = 0
        visual_goal_losses = []
        visual_action_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            visual_history = batch['visual_history'].to(self.device)
            future_actions = batch['future_actions'].to(self.device)
            goal_image = batch['goal_image'].to(self.device)
            
            # Forward pass
            embeddings = self.model(visual_history, future_actions, goal_image)
            
            # VICReg losses
            visual_goal_loss, vg_components = self.vicreg_loss(
                embeddings['visual'], embeddings['goal']
            )
            
            visual_action_loss, va_components = self.vicreg_loss(
                embeddings['visual'], embeddings['action']
            )
            
            # Combined loss
            total_batch_loss = (
                lambda_weight * visual_goal_loss + 
                (1 - lambda_weight) * visual_action_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            visual_goal_losses.append(visual_goal_loss.item())
            visual_action_losses.append(visual_action_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'VG': f'{visual_goal_loss.item():.4f}',
                'VA': f'{visual_action_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Update scheduler
        self.scheduler.step()
        
        # Record history
        avg_total_loss = total_loss / len(dataloader)
        avg_vg_loss = np.mean(visual_goal_losses)
        avg_va_loss = np.mean(visual_action_losses)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.history['epoch'].append(epoch)
        self.history['total_loss'].append(avg_total_loss)
        self.history['visual_goal_loss'].append(avg_vg_loss)
        self.history['visual_action_loss'].append(avg_va_loss)
        self.history['lr'].append(current_lr)

        # Trong train_epoch(), sau forward pass:
        embeddings = self.model(visual_history, future_actions, goal_image)

        # Check embedding statistics:
        visual_emb = embeddings['visual']
        action_emb = embeddings['action'] 
        goal_emb = embeddings['goal']

        print(f"Visual: mean={visual_emb.mean():.3f}, std={visual_emb.std():.3f}")
        print(f"Action: mean={action_emb.mean():.3f}, std={action_emb.std():.3f}")  
        print(f"Goal: mean={goal_emb.mean():.3f}, std={goal_emb.std():.3f}")

        
        return avg_total_loss, avg_vg_loss, avg_va_loss
    
    def save_checkpoint(self, epoch, loss, filepath):
        """Save training checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'history': self.history
        }, filepath)
    
    def plot_training_history(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['epoch'], self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Visual-Goal loss
        axes[0, 1].plot(self.history['epoch'], self.history['visual_goal_loss'])
        axes[0, 1].set_title('Visual-Goal Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Visual-Action loss
        axes[1, 0].plot(self.history['epoch'], self.history['visual_action_loss'])
        axes[1, 0].set_title('Visual-Action Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['epoch'], self.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def get_transforms():
    """Get data transforms"""
    return transforms.Compose([
        transforms.Resize((86, 155)),  # Unity environment resolution
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def main():
    parser = argparse.ArgumentParser(description="Train VANP với Synthesized Data")
    parser.add_argument("--dataset-file", required=True,
                       help="Path to VANP samples JSONL file")
    parser.add_argument("--episodes-dir", required=True,
                       help="Directory chứa episode images")
    parser.add_argument("--output-dir", default="./vanp_checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=512,
                       help="Embedding dimension")
    parser.add_argument("--lambda-weight", type=float, default=0.5,
                       help="Weight for visual-goal vs visual-action loss")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to load (for testing)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume training from checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                       help="Path to checkpoint file")

    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Create dataset and dataloader
    print("📊 Loading dataset...")
    transform = get_transforms()
    dataset = VANPDataset(
        args.dataset_file,
        args.episodes_dir, 
        transform=transform,
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator(device="cuda")  # Thêm dòng này

    )
    
    # Create model
    print("🧠 Creating VANP model...")
    model = VANPModel(
        embed_dim=args.embed_dim,
        num_heads=8,
        num_layers=4,
        action_dim=3,
        tau_p=3,
        tau_f=12
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    
    # Create trainer
    trainer = VANPTrainer(model, device, lr=args.lr)
    # LOAD CHECKPOINT NẾU CÓ
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.checkpoint_path)
    # # Training loop
    # print(f"🚀 Starting VANP training...")
    # print(f"   Dataset: {len(dataset)} samples")
    # print(f"   Batch size: {args.batch_size}")
    # print(f"   Epochs: {args.epochs}")
    # print(f"   Lambda: {args.lambda_weight}")
    
    # best_loss = float('inf')
    
    # for epoch in range(args.epochs):
    #     # Train
    #     total_loss, vg_loss, va_loss = trainer.train_epoch(
    #         dataloader, epoch, args.lambda_weight
    #     )
        
    #     print(f"\n✅ Epoch {epoch}: Total={total_loss:.4f}, "
    #           f"VG={vg_loss:.4f}, VA={va_loss:.4f}")
        
    #     # Save checkpoint
    #     if total_loss < best_loss:
    #         best_loss = total_loss
    #         trainer.save_checkpoint(
    #             epoch, total_loss,
    #             Path(args.output_dir) / "best_vanp_checkpoint.pt"
    #         )
    #         print(f"💾 Best checkpoint saved: {best_loss:.4f}")
        
    #     # Save regular checkpoint
    #     if epoch % 20 == 0:
    #         trainer.save_checkpoint(
    #             epoch, total_loss,
    #             Path(args.output_dir) / f"vanp_checkpoint_epoch_{epoch}.pt"
    #         )
    # Training loop - SỬA DỤNG start_epoch
    print(f"🚀 Starting VANP training from epoch {start_epoch}...")
    print(f"   Dataset: {len(dataset)} samples")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Total epochs: {args.epochs}")
    print(f"   Lambda: {args.lambda_weight}")
    
    best_loss = float('inf')
    
    # Tìm best loss từ history nếu đã có
    if trainer.history['total_loss']:
        best_loss = min(trainer.history['total_loss'])
        print(f"   Current best loss: {best_loss:.4f}")
    
    for epoch in range(start_epoch, args.epochs):  # BẮT ĐẦU TỪ start_epoch
        # Train
        total_loss, vg_loss, va_loss = trainer.train_epoch(
            dataloader, epoch, args.lambda_weight
        )
        
        print(f"\n✅ Epoch {epoch}: Total={total_loss:.4f}, "
              f"VG={vg_loss:.4f}, VA={va_loss:.4f}")
        
        # Save checkpoint
        if total_loss < best_loss:
            best_loss = total_loss
            trainer.save_checkpoint(
                epoch, total_loss,
                Path(args.output_dir) / "best_vanp_checkpoint.pt"
            )
            print(f"💾 Best checkpoint saved: {best_loss:.4f}")
        
        # Save regular checkpoint
        if epoch % 20 == 0:
            trainer.save_checkpoint(
                epoch, total_loss,
                Path(args.output_dir) / f"vanp_checkpoint_epoch_{epoch}.pt"
            )
    # Plot training history
    print("\n📈 Plotting training history...")
    trainer.plot_training_history(
        Path(args.output_dir) / "training_history.png"
    )
    
    # Save final model
    torch.save(
        model.state_dict(),
        Path(args.output_dir) / "vanp_final_model.pt"
    )
    
    print(f"\n🎉 VANP training completed!")
    print(f"📁 Checkpoints saved to: {args.output_dir}")
    print(f"📊 Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()