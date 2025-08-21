#!/usr/bin/env python3
"""
🔥 Grad-CAM Visualization cho ResNet50
Visualize attention maps sử dụng các kỹ thuật:
- Grad-CAM: Gradient-based Class Activation Mapping
- Grad-CAM++: Improved version cho multiple objects
- Layer-wise CAM: Apply ở nhiều conv layers khác nhau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
import argparse

class ResNet50GradCAM:
    """Grad-CAM visualization cho ResNet50"""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
        # Load ResNet50 model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Register hooks cho các layer quan trọng
        self._register_hooks()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # ResNet50 architecture info
        self._print_architecture()
    
    def _load_model(self, model_path):
        """Load ResNet50 model"""
        if model_path and os.path.exists(model_path):
            print(f"🔥 Loading model from: {model_path}")
            model = models.resnet34(pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try different checkpoint formats
            if 'student_backbone_state_dict' in checkpoint:
                state_dict = checkpoint['student_backbone_state_dict']
                # Remove 'backbone.' prefix if present
                cleaned_state = {}
                for key, value in state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key[9:]
                        cleaned_state[new_key] = value
                    else:
                        cleaned_state[key] = value
                model.load_state_dict(cleaned_state, strict=False)
                print("✅ Loaded DINO ResNet50 weights")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("✅ Loaded custom ResNet50 weights")
        else:
            print("🔥 Using ImageNet pretrained ResNet50")
            model = models.resnet50(pretrained=True)
        
        return model
    
    def _print_architecture(self):
        """Print ResNet50 architecture details"""
        print("\n🏗️  ResNet50 Architecture:")
        print("=" * 50)
        
        # ResNet50 có 4 stages chính + stem
        stages = {
            'Stem': 'conv1 + bn1 + relu + maxpool',
            'Stage 1 (conv2_x)': '3 Bottleneck blocks, 64 channels',
            'Stage 2 (conv3_x)': '4 Bottleneck blocks, 128 channels', 
            'Stage 3 (conv4_x)': '6 Bottleneck blocks, 256 channels',
            'Stage 4 (conv5_x)': '3 Bottleneck blocks, 512 channels'
        }
        
        for stage, description in stages.items():
            print(f"  {stage:<20}: {description}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n📊 Total parameters: {total_params/1e6:.1f}M")
    
    def _register_hooks(self):
        """Register hooks để capture feature maps và gradients"""
        self.feature_maps = {}
        self.gradients = {}
        
        # Hook cho các layer quan trọng
        target_layers = {
            'layer1': 'conv2_x (64 channels)',
            'layer2': 'conv3_x (128 channels)', 
            'layer3': 'conv4_x (256 channels)',
            'layer4': 'conv5_x (512 channels) - Final conv layer'
        }
        
        def forward_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        # Register hooks
        for name, description in target_layers.items():
            layer = getattr(self.model, name)
            layer.register_forward_hook(forward_hook(name))
            layer.register_backward_hook(backward_hook(name))
            print(f"  🔗 Hooked {name}: {description}")
    
    def preprocess_image(self, image_path):
        """Preprocess image cho ResNet50"""
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Original image cho visualization
        original = np.array(image)
        
        # Preprocess cho model
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return input_tensor, original
    
    def generate_gradcam(self, input_tensor, target_class=None, layer_name='layer4'):
        """Generate Grad-CAM cho target class và layer"""
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get feature maps và gradients
        feature_maps = self.feature_maps[layer_name]
        gradients = self.gradients[layer_name]
        
        # Global average pooling của gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weighted combination của feature maps
        cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32, device=self.device)
        
        for i, w in enumerate(pooled_gradients):
            cam += w * feature_maps[0, i, :, :]
        
        # Apply ReLU và normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def generate_gradcam_plus_plus(self, input_tensor, target_class=None, layer_name='layer4'):
        """Generate Grad-CAM++ (improved version)"""
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get feature maps và gradients
        feature_maps = self.feature_maps[layer_name]
        gradients = self.gradients[layer_name]
        
        # Grad-CAM++ weights calculation
        b, k, u, v = gradients.size()
        
        # Global average pooling
        alpha_num = gradients.pow(2)
        alpha_denom = alpha_num.mul(2) + \
                     gradients.mul(feature_maps.pow(2).sum(dim=[2, 3], keepdim=True))
        alpha = alpha_num.div(alpha_denom + 1e-7)
        
        weights = (alpha * gradients).sum(dim=[2, 3], keepdim=True)
        
        # Generate CAM
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0, 0].detach().cpu().numpy()
    
    def generate_layer_wise_cam(self, input_tensor, target_class=None):
        """Generate CAM cho nhiều layers khác nhau"""
        layer_cams = {}
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            try:
                cam = self.generate_gradcam(input_tensor, target_class, layer_name)
                layer_cams[layer_name] = cam
            except Exception as e:
                print(f"⚠️  Failed to generate CAM for {layer_name}: {e}")
                continue
        
        return layer_cams
    
    def overlay_cam_on_image(self, original_image, cam, alpha=0.6):
        """Overlay CAM lên original image"""
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Overlay
        cam_on_image = heatmap + np.float32(original_image) / 255
        cam_on_image = cam_on_image / np.max(cam_on_image)
        
        return cam_on_image
    
    def visualize_attention_maps(self, image_path, output_dir="./gradcam_results"):
        """Visualize tất cả attention maps cho một image"""
        print(f"\n🎨 Visualizing attention maps for: {image_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)
        
        # Get target class
        with torch.no_grad():
            output = self.model(input_tensor)
            target_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        print(f"🎯 Target class: {target_class} (confidence: {confidence:.3f})")
        
        # Generate different CAMs
        print("🔄 Generating attention maps...")
        
        # 1. Grad-CAM (final layer)
        gradcam = self.generate_gradcam(input_tensor, target_class, 'layer4')
        
        # 2. Grad-CAM++
        gradcam_plus = self.generate_gradcam_plus_plus(input_tensor, target_class, 'layer4')
        
        # 3. Layer-wise CAM
        layer_cams = self.generate_layer_wise_cam(input_tensor, target_class)
        
        # Visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'ResNet50 Attention Maps - Class {target_class} ({confidence:.3f})', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Grad-CAM
        axes[0, 1].imshow(gradcam, cmap='jet')
        axes[0, 1].set_title('Grad-CAM (Layer 4)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Grad-CAM++
        axes[0, 2].imshow(gradcam_plus, cmap='jet')
        axes[0, 2].set_title('Grad-CAM++ (Layer 4)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Empty plot for symmetry
        axes[0, 3].axis('off')
        
        # Layer-wise CAMs
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, layer_name in enumerate(layer_names):
            if layer_name in layer_cams:
                axes[1, i].imshow(layer_cams[layer_name], cmap='jet')
                axes[1, i].set_title(f'{layer_name.upper()}', fontweight='bold')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, f"attention_maps_{Path(image_path).stem}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization to: {output_path}")
        
        # Overlay visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'ResNet50 CAM Overlays - Class {target_class}', fontsize=16, fontweight='bold')
        
        # Original
        axes[0].imshow(original_image)
        axes[0].set_title('Original', fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM overlay
        gradcam_overlay = self.overlay_cam_on_image(original_image, gradcam)
        axes[1].imshow(gradcam_overlay)
        axes[1].set_title('Grad-CAM Overlay', fontweight='bold')
        axes[1].axis('off')
        
        # Grad-CAM++ overlay
        gradcam_plus_overlay = self.overlay_cam_on_image(original_image, gradcam_plus)
        axes[2].imshow(gradcam_plus_overlay)
        axes[2].set_title('Grad-CAM++ Overlay', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"overlays_{Path(image_path).stem}.png")
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved overlays to: {overlay_path}")
        
        plt.show()
        
        return {
            'gradcam': gradcam,
            'gradcam_plus': gradcam_plus,
            'layer_cams': layer_cams,
            'target_class': target_class,
            'confidence': confidence
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ResNet50 Grad-CAM Visualization")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to ResNet50 model checkpoint")
    parser.add_argument("--output", type=str, default="./gradcam_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("🔥 ResNet50 Grad-CAM Visualization")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    # Initialize Grad-CAM
    gradcam = ResNet50GradCAM(model_path=args.model, device=args.device)
    
    # Visualize attention maps
    results = gradcam.visualize_attention_maps(args.image, args.output)
    
    print(f"\n🎉 Visualization completed!")
    print(f"📊 Target class: {results['target_class']}")
    print(f"📊 Confidence: {results['confidence']:.3f}")
    print(f"📁 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
