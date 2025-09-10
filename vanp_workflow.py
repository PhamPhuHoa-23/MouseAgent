#!/usr/bin/env python3
"""
Complete VANP Workflow
1. Collect data từ trained models
2. Train VANP với synthesized data
3. Use VANP cho Robust Foraging Challenge
"""

import subprocess
import argparse
import time
from pathlib import Path
import yaml


def run_command(cmd, description, check=True):
    """Run command với description"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    result = subprocess.run(cmd, check=check)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS ({elapsed:.1f}s)")
    else:
        print(f"❌ {description} - FAILED ({elapsed:.1f}s)")
        if check:
            raise RuntimeError(f"Command failed: {cmd}")
    
    return result.returncode == 0


class VANPWorkflow:
    def __init__(self, config_file="vanp_workflow_config.yaml"):
        """Initialize workflow with config"""
        self.config_file = config_file
        self.config = self.load_or_create_config()
        
        print(f"🎯 VANP Workflow Configuration:")
        print(f"   Data collection: {self.config['data_collection']['episodes_per_model']} episodes per model")
        print(f"   VANP training: {self.config['vanp_training']['epochs']} epochs")
        print(f"   Output: {self.config['output']['base_dir']}")
    
    def load_or_create_config(self):
        """Load hoặc tạo config file"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config
        default_config = {
            'data_collection': {
                'models_dir': './results',
                'builds_dir': './Builds',
                'episodes_per_model': 30,  # Reasonable number
                'environments': ['NormalTrain', 'FogTrain'],
                'episode_length': 500,     # 2 minutes @ 4Hz
                'output_dir': './vanp_dataset'
            },
            'vanp_training': {
                'batch_size': 16,          # Memory friendly
                'epochs': 50,
                'lr': 1e-3,
                'embed_dim': 512,
                'lambda_weight': 0.5,      # Balance visual-goal vs visual-action
                'output_dir': './vanp_checkpoints'
            },
            'robust_foraging': {
                'use_vanp_features': True,
                'freeze_vanp_backbone': False,  # Allow fine-tuning
                'competition_dir': './robust_foraging'
            },
            'output': {
                'base_dir': './vanp_workflow_results',
                'save_intermediate': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"📝 Created default config: {self.config_file}")
        return default_config
    
    def step_1_collect_data(self):
        """Step 1: Collect data từ trained models"""
        print("\n" + "="*80)
        print("📊 STEP 1: DATA COLLECTION FROM TRAINED MODELS")
        print("="*80)
        
        data_config = self.config['data_collection']
        
        cmd = [
            'python', 'vanp_data_collector.py',
            '--output-dir', data_config['output_dir'],
            '--models-dir', data_config['models_dir'],
            '--builds-dir', data_config['builds_dir'],
            '--episodes-per-model', str(data_config['episodes_per_model']),
            '--environments'
        ] + data_config['environments']
        
        success = run_command(cmd, "Collecting VANP training data")
        
        if success:
            # Verify data collection
            dataset_file = Path(data_config['output_dir']) / "vanp_samples.jsonl"
            episodes_dir = Path(data_config['output_dir']) / "episodes"
            
            if dataset_file.exists() and episodes_dir.exists():
                # Count samples
                with open(dataset_file, 'r') as f:
                    sample_count = sum(1 for _ in f)
                print(f"📊 Collected {sample_count} VANP samples")
                return True
            else:
                print(f"❌ Data collection incomplete")
                return False
        
        return success
    
    def step_2_train_vanp(self):
        """Step 2: Train VANP model"""
        print("\n" + "="*80)
        print("🧠 STEP 2: VANP MODEL TRAINING")
        print("="*80)
        
        data_config = self.config['data_collection']
        train_config = self.config['vanp_training']
        
        # Check if data exists
        dataset_file = Path(data_config['output_dir']) / "vanp_samples.jsonl"
        episodes_dir = Path(data_config['output_dir']) / "episodes"
        
        if not dataset_file.exists():
            print(f"❌ Dataset file not found: {dataset_file}")
            return False
        
        if not episodes_dir.exists():
            print(f"❌ Episodes directory not found: {episodes_dir}")
            return False
        
        cmd = [
            'python', 'vanp_trainer.py',
            '--dataset-file', str(dataset_file),
            '--episodes-dir', str(episodes_dir),
            '--output-dir', train_config['output_dir'],
            '--batch-size', str(train_config['batch_size']),
            '--epochs', str(train_config['epochs']),
            '--lr', str(train_config['lr']),
            '--embed-dim', str(train_config['embed_dim']),
            '--lambda-weight', str(train_config['lambda_weight'])
        ]
        
        success = run_command(cmd, "Training VANP model")
        
        if success:
            # Verify training output
            checkpoint_file = Path(train_config['output_dir']) / "best_vanp_checkpoint.pt"
            if checkpoint_file.exists():
                print(f"✅ VANP model trained successfully")
                return True
            else:
                print(f"❌ VANP training incomplete")
                return False
        
        return success
    
    def step_3_setup_foraging_integration(self):
        """Step 3: Setup VANP integration for Robust Foraging"""
        print("\n" + "="*80)
        print("🎮 STEP 3: ROBUST FORAGING INTEGRATION")
        print("="*80)
        
        foraging_config = self.config['robust_foraging']
        train_config = self.config['vanp_training']
        
        # Check if VANP model exists
        vanp_checkpoint = Path(train_config['output_dir']) / "best_vanp_checkpoint.pt"
        if not vanp_checkpoint.exists():
            print(f"❌ VANP checkpoint not found: {vanp_checkpoint}")
            return False
        
        # Create VANP encoder for Robust Foraging
        foraging_encoder_code = self.create_foraging_vanp_encoder()
        
        # Save encoder
        encoder_dir = Path("ForagingEncoders")
        encoder_dir.mkdir(exist_ok=True)
        
        encoder_file = encoder_dir / "vanp_encoder.py"
        with open(encoder_file, 'w', encoding="utf-8") as f:
            f.write(foraging_encoder_code)
        
        print(f"📝 Created Robust Foraging VANP encoder: {encoder_file}")
        
        # Create integration config
        integration_config = {
            'vanp_checkpoint': str(vanp_checkpoint),
            'encoder_file': str(encoder_file),
            'freeze_backbone': foraging_config['freeze_vanp_backbone'],
            'embed_dim': self.config['vanp_training']['embed_dim']
        }
        
        config_file = Path(foraging_config['competition_dir']) / "vanp_config.yaml"
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(integration_config, f, default_flow_style=False)
        
        print(f"📝 Created integration config: {config_file}")
        return True
    
    def create_foraging_vanp_encoder(self):
        """Create VANP encoder cho Robust Foraging"""
        return '''#!/usr/bin/env python3
"""
#!/usr/bin/env python3
"""
VANP Encoder for Robust Foraging Challenge
Pre-trained visual encoder với navigation-relevant features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet34
from pathlib import Path


# class VANPEncoder(nn.Module):
#     """VANP Visual Encoder"""
    
#     def __init__(self, output_dim=512):
#         super().__init__()
        
#         # ResNet34 backbone
#         resnet = resnet34(pretrained=False)
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
#         # Extract features
#         features = self.backbone(x)
#         features = self.adaptive_pool(features)
#         features = features.flatten(1)
        
#         # Project
#         features = self.projection(features)
#         return features
import torch
import torch.nn as nn
import torch.nn.init as init
from mlagents.trainers.torch.model_serialization import exporting_to_onnx
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from pathlib import Path

class LightWeatherSimulation:
    def __init__(self):
        self.weather_types = ['light_dim', 'very_light_fog']
    
    def apply_light_dim_effect(self, image):
        """Áp dụng hiệu ứng làm tối nhẹ thay vì night effect mạnh"""
        enhancer = ImageEnhance.Brightness(image)
        dimmed_image = enhancer.enhance(0.85)  # Nhẹ hơn: 0.85 thay vì 0.6
        return dimmed_image
    
    def apply_very_light_fog_effect(self, image):
        """Fog effect rất nhẹ, chỉ ở các góc"""
        img_array = np.array(image).astype(np.float32)
        H, W = img_array.shape
        
        center_x, center_y = W//2, H//2
        y_coords, x_coords = np.ogrid[:H, :W]
        
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Fog intensity nhẹ hơn nhiều: 0.15 max thay vì 0.4
        fog_intensity = (distances / max_distance) * 0.1 + 0.02
        
        fog_color = 190
        fogged = img_array * (1 - fog_intensity) + fog_color * fog_intensity
        fogged = np.clip(fogged, 0, 255)
        
        return Image.fromarray(fogged.astype(np.uint8))

class LightColorJitterAugmentation:
    def __init__(self, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1)):
        """Giảm range từ (0.7, 1.4) xuống (0.9, 1.1)"""
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def apply_jitter(self, image):
        jittered = image.copy()
        
        brightness_factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(jittered)
        jittered = enhancer.enhance(brightness_factor)
        
        contrast_factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(jittered)
        jittered = enhancer.enhance(contrast_factor)
        
        return jittered, f"B:{brightness_factor:.2f},C:{contrast_factor:.2f}"

class LightAugmentationPipeline:
    def __init__(self):
        self.weather_sim = LightWeatherSimulation()
        self.color_jitter = LightColorJitterAugmentation()
        
    def apply_light_augmentation(self, image):
        """Augmentation pipeline nhẹ cho navigation task"""
        augmented = image.copy()
        aug_info = []
        
        # Giảm probability của weather effects: 10% thay vì 25%
        weather_chance = random.random()
        if weather_chance < 0.05:  # 5% thay vì 25%
            if random.random() < 0.5:
                augmented = self.weather_sim.apply_light_dim_effect(augmented)
                aug_info.append("Light dim")
            else:
                augmented = self.weather_sim.apply_very_light_fog_effect(augmented)
                aug_info.append("Very light fog")
        
        # Tăng probability của color jittering nhẹ: 50% thay vì 30-40%
        if random.random() < 0.5:
            augmented, jitter_info = self.color_jitter.apply_jitter(augmented)
            aug_info.append("Light jitter")
            
        return augmented, " + ".join(aug_info) if aug_info else "No augmentation"

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def pool_out_shape(input_shape, kernel_size, stride=2, padding=0):
    h, w = input_shape
    h_out = (h - kernel_size + 2 * padding) // stride + 1
    w_out = (w - kernel_size + 2 * padding) // stride + 1
    return h_out, w_out

class ResNetBlock(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            Swish(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            Swish(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.layers(input_tensor)

class VANPEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.embed_dim = output_size
        self.use_augmentation = True
        self.height = height
        self.width = width
        
        # Sử dụng light augmentation pipeline
        self.augmentation_pipeline = LightAugmentationPipeline()
        
        n_channels = [16, 32, 32]
        n_blocks = 2
        layers = []
        last_channel = initial_channels
        current_height, current_width = height, width

        for channel in n_channels:
            layers.append(nn.Conv2d(last_channel, channel, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            current_height, current_width = pool_out_shape((current_height, current_width), kernel_size=3)
            for _ in range(n_blocks):
                layers.append(ResNetBlock(channel))
            last_channel = channel

        layers.append(Swish())
        self.final_flat_size = n_channels[-1] * current_height * current_width

        self.dense = nn.Linear(self.final_flat_size, output_size)
        init.kaiming_normal_(self.dense.weight, mode='fan_out', nonlinearity='relu', a=1.41)
        if self.dense.bias is not None:
            init.zeros_(self.dense.bias)

        self.sequential = nn.Sequential(*layers)
        
        # Load checkpoint if available
        self._load_checkpoint_if_exists()
    
    def _load_checkpoint_if_exists(self):
        """Try to load checkpoint from common paths"""
        pass
        # possible_paths = [
        #     "./resnet_aug/best_robust_agent.pth",
        #     "./resnet_aug/best_robust_agent_naturevisualencoder.pth",
        #     "C:/Users/admin/Neurips2025/MouseVsAI/mouse_vs_ai_windows/resnet_aug/best_robust_agent_naturevisualencoder.pth"
        # ]
        
        # for checkpoint_path in possible_paths:
        #     if Path(checkpoint_path).exists():
        #         try:
        #             checkpoint = torch.load(checkpoint_path, map_location='cpu')
        #             pretrained_state_dict = checkpoint['model_state_dict']
                    
        #             backbone_weights = {}
        #             for key, value in pretrained_state_dict.items():
        #                 if key.startswith('backbone.'):
        #                     new_key = key[9:]
        #                     backbone_weights[new_key] = value
                    
        #             if backbone_weights:
        #                 missing_keys, unexpected_keys = self.load_state_dict(backbone_weights, strict=False)
        #                 print(f"✅ Loaded checkpoint: {checkpoint_path}")
        #                 return
        #         except Exception as e:
        #             print(f"Failed to load {checkpoint_path}: {e}")
        #             continue

    def _apply_light_augmentation_to_batch(self, visual_obs):
        """Áp dụng augmentation nhẹ cho batch"""
        if not self.training or not self.use_augmentation:
            return visual_obs
        
        batch_size = visual_obs.shape[0]
        
        if visual_obs.shape[-1] == 1:
            np_batch = visual_obs.cpu().numpy()
            augmented_batch = []
            
            for i in range(batch_size):
                img_array = np_batch[i, :, :, 0]
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                
                # Giảm probability từ 40% xuống 25%
                if random.random() < 0.25:
                    augmented_pil, _ = self.augmentation_pipeline.apply_light_augmentation(img_pil)
                else:
                    augmented_pil = img_pil
                
                augmented_array = np.array(augmented_pil).astype(np.float32) / 255.0
                augmented_array = np.expand_dims(augmented_array, axis=-1)
                augmented_batch.append(augmented_array)
            
            augmented_batch = np.stack(augmented_batch, axis=0)
            return torch.from_numpy(augmented_batch).to(visual_obs.device)
        
        else:
            if visual_obs.shape[1] == 1:
                np_batch = visual_obs.cpu().numpy()
                augmented_batch = []
                
                for i in range(batch_size):
                    img_array = np_batch[i, 0, :, :]
                    img_pil = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                    
                    # Giảm probability từ 40% xuống 25%
                    if random.random() < 0.25:
                        augmented_pil, _ = self.augmentation_pipeline.apply_light_augmentation(img_pil)
                    else:
                        augmented_pil = img_pil
                    
                    augmented_array = np.array(augmented_pil).astype(np.float32) / 255.0
                    augmented_array = np.expand_dims(augmented_array, axis=0)
                    augmented_batch.append(augmented_array)
                
                augmented_batch = np.stack(augmented_batch, axis=0)
                return torch.from_numpy(augmented_batch).to(visual_obs.device)
        
        return visual_obs

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Bỏ comment để sử dụng augmentation trong training
        # if self.training and self.use_augmentation:
        #     visual_obs = self._apply_light_augmentation_to_batch(visual_obs)
        # print(visual_obs.shape)
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        # original_shape = visual_obs.shape

        # if len(visual_obs.shape) == 5:  # (B, T, C, H, W)
        #     B, T, C, H, W = visual_obs.shape
        #     visual_obs = visual_obs.view(B * T, C, H, W)
        hidden = self.sequential(visual_obs)
        before_out = hidden.reshape(-1, self.final_flat_size)
        features = torch.relu(self.dense(before_out))
        # if len(original_shape) == 5:
        #     features = features.view(B, T, -1)
        
        return features

class RobustForagingVANPAgent(nn.Module):
    """Complete agent với VANP encoder"""
    
    def __init__(self, vanp_checkpoint_path, action_dim=3, freeze_encoder=False):
        super().__init__()
        
        print(f"🔥 Loading VANP encoder from: {vanp_checkpoint_path}")
        
        # Load VANP encoder
        self.vanp_encoder = VANPEncoder(output_dim=512)
        
        # Load pre-trained weights
        if Path(vanp_checkpoint_path).exists():
            checkpoint = torch.load(vanp_checkpoint_path, map_location='cpu')
            
            # Extract visual encoder weights
            vanp_state = checkpoint['model_state_dict']
            encoder_state = {}
            
            for key, value in vanp_state.items():
                if key.startswith('visual_encoder.'):
                    new_key = key[15:]  # Remove 'visual_encoder.' prefix
                    encoder_state[new_key] = value
            
            # Load weights
            missing, unexpected = self.vanp_encoder.load_state_dict(encoder_state, strict=False)
            print(f"✅ VANP encoder loaded: {len(encoder_state)} params")
            
            if freeze_encoder:
                for param in self.vanp_encoder.parameters():
                    param.requires_grad = False
                print("🧊 VANP encoder frozen")
        
        # Policy head cho Robust Foraging
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim)
        )
        
        # Value head (optional)
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, visual_obs):
        """
        Args:
            visual_obs: (B, C, H, W) grayscale images from foraging env
        """
        # Convert grayscale to RGB if needed
        if visual_obs.shape[1] == 1:
            visual_obs = visual_obs.repeat(1, 3, 1, 1)
        
        # Normalize
        visual_obs = visual_obs.float() / 255.0
        
        # ImageNet normalization (VANP training used this)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(visual_obs.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(visual_obs.device)
        visual_obs = (visual_obs - mean) / std
        
        # Resize to 86x155 (VANP training size)
        visual_obs = F.interpolate(visual_obs, size=(86, 155), mode='bilinear', align_corners=False)
        
        # Extract VANP features
        vanp_features = self.vanp_encoder(visual_obs)  # (B, 512)
        
        # Policy and value
        actions = self.policy_head(vanp_features)  # (B, action_dim)
        value = self.value_head(vanp_features)      # (B, 1)
        
        return actions, value, vanp_features


# Factory function
def create_vanp_foraging_agent(vanp_checkpoint_path, freeze_encoder=False):
    """Create VANP agent for Robust Foraging"""
    return RobustForagingVANPAgent(
        vanp_checkpoint_path=vanp_checkpoint_path,
        action_dim=3,  # Adjust based on foraging action space
        freeze_encoder=freeze_encoder
    )
'''
    
    def step_4_test_integration(self):
        """Step 4: Test VANP integration"""
        print("\n" + "="*80)
        print("🧪 STEP 4: TESTING VANP INTEGRATION")
        print("="*80)
        
        foraging_config = self.config['robust_foraging']
        integration_config_path = Path(foraging_config['competition_dir']) / "vanp_config.yaml"
        
        if not integration_config_path.exists():
            print(f"❌ Integration config not found: {integration_config_path}")
            return False
        
        # Create test script
        test_script = self.create_test_script()
        test_file = Path("test_vanp_integration.py")
        
        with open(test_file, 'w', encoding="utf-8") as f:
            f.write(test_script)
        
        # Run test
        cmd = ['python', str(test_file)]
        success = run_command(cmd, "Testing VANP integration", check=False)
        
        if success:
            print("✅ VANP integration test passed!")
        else:
            print("⚠️  VANP integration test had issues, check logs")
        
        return success
    
    def create_test_script(self):
        """Create integration test script"""
        return '''#!/usr/bin/env python3
"""Test VANP integration"""

import torch
import numpy as np
from pathlib import Path
import yaml
import sys

# Add ForagingEncoders to path
sys.path.append('ForagingEncoders')

def test_vanp_integration():
    """Test VANP encoder loading and inference"""
    print("🧪 Testing VANP integration...")
    
    try:
        from vanp_encoder import create_vanp_foraging_agent
        
        # Load config
        config_path = Path("robust_foraging/vanp_config.yaml")
        if not config_path.exists():
            print("❌ Config file not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vanp_checkpoint = config['vanp_checkpoint']
        
        # Create agent
        print(f"📦 Loading VANP agent from: {vanp_checkpoint}")
        agent = create_vanp_foraging_agent(
            vanp_checkpoint_path=vanp_checkpoint,
            freeze_encoder=config['freeze_backbone']
        )
        
        # Test inference
        print("🔍 Testing inference...")
        dummy_input = torch.randint(0, 255, (2, 1, 86, 155), dtype=torch.uint8)  # Batch of 2 grayscale images
        
        with torch.no_grad():
            actions, values, features = agent(dummy_input)
        
        print(f"✅ Input shape: {dummy_input.shape}")
        print(f"✅ Actions shape: {actions.shape}")
        print(f"✅ Values shape: {values.shape}")
        print(f"✅ Features shape: {features.shape}")
        
        # Check output ranges
        print(f"📊 Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"📊 Value range: [{values.min():.3f}, {values.max():.3f}]")
        print(f"📊 Feature mean: {features.mean():.3f}, std: {features.std():.3f}")
        
        print("🎉 VANP integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ VANP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vanp_integration()
    sys.exit(0 if success else 1)
'''
    
    def run_complete_workflow(self):
        """Run complete VANP workflow"""
        print("🚀 Starting Complete VANP Workflow for Robust Foraging")
        print("=" * 80)
        
        start_time = time.time()
        
        steps = [
            ("Data Collection", self.step_1_collect_data),
            ("VANP Training", self.step_2_train_vanp),
            ("Foraging Integration", self.step_3_setup_foraging_integration),
            ("Integration Testing", self.step_4_test_integration)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            try:
                print(f"\n{'='*20} {step_name} {'='*20}")
                success = step_func()
                results[step_name] = success
                
                if success:
                    print(f"✅ {step_name} completed successfully!")
                else:
                    print(f"❌ {step_name} failed!")
                    if step_name in ["Data Collection", "VANP Training"]:
                        # Critical steps - stop workflow
                        break
                    else:
                        # Non-critical steps - continue
                        continue
                        
            except Exception as e:
                print(f"💥 {step_name} crashed: {e}")
                results[step_name] = False
                if step_name in ["Data Collection", "VANP Training"]:
                    break
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("🎯 VANP WORKFLOW SUMMARY")
        print(f"{'='*80}")
        
        for step_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {step_name:<25}: {status}")
        
        print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
        
        critical_steps = ["Data Collection", "VANP Training"]
        critical_success = all(results.get(step, False) for step in critical_steps)
        
        if critical_success:
            print("\n🎉 VANP workflow completed successfully!")
            print("\n📋 Next steps for Robust Foraging Challenge:")
            print("   1. Download competition environment from:")
            print("      https://robustforaging.github.io/")
            print("   2. Use VANP encoder in ForagingEncoders/vanp_encoder.py")
            print("   3. Train on competition data with VANP features")
            print("   4. Submit to competition!")
            
            # Save workflow results
            self.save_workflow_results(results, total_time)
        else:
            failed_step = next(step for step in critical_steps if not results.get(step, False))
            print(f"\n😞 Critical workflow step failed: {failed_step}")
            print("   Please check logs and retry")
    
    def save_workflow_results(self, results, total_time):
        """Save workflow results"""
        output_config = self.config['output']
        results_dir = Path(output_config['base_dir'])
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Summary report
        summary = {
            'workflow_results': results,
            'total_time_minutes': total_time / 60,
            'config_used': self.config,
            'generated_files': {
                'dataset': str(Path(self.config['data_collection']['output_dir']) / "vanp_samples.jsonl"),
                'vanp_checkpoint': str(Path(self.config['vanp_training']['output_dir']) / "best_vanp_checkpoint.pt"),
                'foraging_encoder': "ForagingEncoders/vanp_encoder.py",
                'integration_config': str(Path(self.config['robust_foraging']['competition_dir']) / "vanp_config.yaml")
            }
        }
        
        summary_file = results_dir / "workflow_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"📄 Workflow summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Complete VANP Workflow")
    parser.add_argument("--step", 
                       choices=['collect', 'train', 'integrate', 'test', 'all'],
                       default='all',
                       help="Which step to run")
    parser.add_argument("--config", type=str, default="vanp_workflow_config.yaml",
                       help="Workflow configuration file")
    
    args = parser.parse_args()
    
    workflow = VANPWorkflow(args.config)
    
    if args.step == 'collect':
        workflow.step_1_collect_data()
    elif args.step == 'train':
        workflow.step_2_train_vanp()
    elif args.step == 'integrate':
        workflow.step_3_setup_foraging_integration()
    elif args.step == 'test':
        workflow.step_4_test_integration()
    elif args.step == 'all':
        workflow.run_complete_workflow()


if __name__ == "__main__":
    main()