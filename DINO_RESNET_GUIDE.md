# 🔥 DINO ResNet Workflow - CNN-based Self-Supervised Learning

## 🎯 Sự khác biệt: DINO ResNet vs DINOv2 ViT

| Feature | DINO ResNet (CNN) | DINOv2 ViT (Transformer) |
|---------|-------------------|---------------------------|
| Backbone | ResNet18/50 | Vision Transformer |
| Input Processing | Convolution | Patch embedding |
| Config Parameters | `backbone.arch`, `output_dim` | `patch_size`, `num_heads`, `embed_dim` |
| File | `dino_resnet_config.yaml` | `dino_workflow_config.yaml` |

## 📄 DINO ResNet Config (`dino_resnet_config.yaml`)

```yaml
backbone:
  arch: resnet18              # ResNet architecture (resnet18/resnet50)
  output_dim: 512             # ResNet output dimension  
  pretrained: true            # Bắt đầu từ ImageNet pretrained

head:
  bottleneck_dim: 256         # Projection head bottleneck
  out_dim: 8192               # Final output dimension

loss:
  out_dim: 8192               # Loss dimension
  student_temp: 0.1           # Student temperature
  teacher_temp: 0.07          # Teacher temperature

optimizer:
  lr: 0.001                   # General learning rate
  backbone_lr: 0.0001         # Backbone learning rate (lower)
  weight_decay: 0.0001        # Weight decay

training:
  epochs: 50                  # Training epochs
  batch_size: 64              # Batch size
  teacher_momentum: 0.996     # EMA momentum
```

## 🚀 Cách chạy DINO ResNet

### Cách 1: Workflow hoàn chỉnh
```bash
# Chạy toàn bộ pipeline
python dino_resnet_workflow.py --step all

# Chạy từng bước
python dino_resnet_workflow.py --step pretrain   # Pre-training
python dino_resnet_workflow.py --step finetune   # RL fine-tuning
python dino_resnet_workflow.py --step compare    # So sánh kết quả
```

### Cách 2: Pre-training riêng biệt
```bash
# Với default config
python dino_resnet.py --data dataset_mouse_dino/train --output ./dino_resnet_checkpoints

# Với custom config  
python dino_resnet.py --data dataset_mouse_dino/train --config dino_resnet_config.yaml --output ./checkpoints
```

### Cách 3: Training script riêng
```bash
# Sử dụng training config
python train_advanced.py --config config_examples/dino_resnet_training.yaml
```

## ⚙️ Config Parameters chi tiết

### Backbone Settings:
- `arch: resnet18` - Nhẹ, nhanh (11M params)
- `arch: resnet50` - Mạnh hơn (25M params)
- `output_dim: 512` - Feature dimension từ ResNet
- `pretrained: true` - Dùng ImageNet weights

### Training Settings:
- `batch_size: 64` - Tuỳ chỉnh theo GPU memory
- `epochs: 50` - Đủ cho convergence
- `backbone_lr: 0.0001` - Lower LR cho pretrained backbone
- `lr: 0.001` - Higher LR cho projection head

### Loss Settings:
- `out_dim: 8192` - Output space dimension
- `teacher_temp: 0.07` - Sharpening teacher predictions
- `student_temp: 0.1` - Student temperature

## 🔧 Tùy chỉnh theo Hardware

### GPU 8GB:
```yaml
training:
  batch_size: 32
  epochs: 30
backbone:
  arch: resnet18
```

### GPU 16GB+:
```yaml
training:
  batch_size: 64
  epochs: 50
backbone:
  arch: resnet50
```

## 📊 So sánh Architectures

| Architecture | Params | Memory | Speed | Performance |
|--------------|--------|--------|-------|-------------|
| ResNet18     | 11M    | Low    | Fast  | Good        |
| ResNet50     | 25M    | Medium | Med   | Better      |

## 🎯 Files cần thiết

```
mouse_vs_ai_windows/
├── dino_resnet_config.yaml           # ⭐ MAIN CONFIG (ResNet)
├── dino_resnet_workflow.py           # Workflow script
├── dino_resnet.py                    # Pre-training script
├── Encoders/dino_resnet_fixed.py     # ResNet encoder
└── config_examples/
    ├── dino_resnet_training.yaml     # RL training config
    └── dino_resnet_finetuning.yaml   # Fine-tuning config
```

## 💡 Lưu ý quan trọng

### ✅ DINO ResNet (CNN):
- Sử dụng `dino_resnet_config.yaml`
- Parameters: `backbone.arch`, `output_dim`
- Fast inference, ONNX compatible
- Good for computer vision tasks

### ❌ Không phải DINOv2 ViT:
- KHÔNG dùng `dino_workflow_config.yaml`
- KHÔNG có `patch_size`, `num_heads`, `embed_dim`
- Đó là cho Vision Transformer

## 🔥 Quick Start

```bash
# 1. Kiểm tra config
cat dino_resnet_config.yaml

# 2. Chạy pre-training
python dino_resnet_workflow.py --step pretrain

# 3. Fine-tune RL
python dino_resnet_workflow.py --step finetune

# 4. So sánh kết quả
python dino_resnet_workflow.py --step compare
```

**🎯 Đúng rồi! DINO ResNet = CNN backbone + DINO self-supervised learning!**

