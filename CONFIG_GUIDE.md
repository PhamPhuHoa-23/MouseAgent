# 📄 Hướng dẫn Config cho Pre-training

## 🎯 Cấu trúc Config Files

### 1. **Main Workflow Config**: `dino_workflow_config.yaml`
Điều khiển toàn bộ pipeline (data collection → pre-training → RL fine-tuning):

```yaml
data_collection:
  env: NormalTrain                    # Unity environment  
  episodes: 200                       # Số episodes thu thập
  steps_per_episode: 500              # Steps per episode
  output_dir: ./dataset_mouse_dino    # Output directory
  additional_envs:                    # Thu thập thêm từ environments khác
    - FogTrain
    - RandomTrain
  train_val_split: true               # Tách train/val

pretraining:
  epochs: 50                          # Số epochs pre-train
  batch_size: 32                      # Batch size (tuỳ GPU)
  output_dir: ./dino_checkpoints      # Nơi lưu checkpoints
  use_wandb: false                    # Sử dụng W&B logging
  model:
    img_size: 224                     # Input image size
    patch_size: 16                    # ViT patch size
    embed_dim: 384                    # Embedding dimension
    depth: 6                          # Số transformer layers
    num_heads: 6                      # Attention heads

finetuning:
  architectures:
    - dino_encoder                    # Encoder architecture
  runs_per_network: 3                 # Số runs per architecture
  max_steps: 50000                    # Max RL training steps
  output_dir: ./dino_rl_results       # RL results directory
  unfreeze_layers: 2                  # Layers để fine-tune
```

### 2. **DINO Pre-training Config**: `dino_config.yaml` 
Chi tiết config cho self-supervised pre-training (tự động tạo):

```yaml
model:
  img_size: 224                       # Image size
  patch_size: 16                      # Patch size cho ViT
  embed_dim: 384                      # Embedding dimension
  depth: 6                            # Transformer depth
  num_heads: 6                        # Multi-head attention

head:
  out_dim: 65536                      # Output dimension
  bottleneck_dim: 256                 # Bottleneck dimension

loss:
  out_dim: 65536                      # Loss output dim
  teacher_temp: 0.07                  # Teacher temperature
  student_temp: 0.1                   # Student temperature

optimizer:
  lr: 0.0005                          # Learning rate
  weight_decay: 0.04                  # Weight decay

training:
  epochs: 50                          # Training epochs
  batch_size: 32                      # Batch size
  teacher_momentum: 0.996             # EMA momentum
```

### 3. **ML-Agents Config**: `Config/nature.yaml`, etc.
Cho RL fine-tuning:

```yaml
behaviors:
  My Behavior:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      learning_rate: 0.0003
      # ... other PPO hyperparameters
    network_settings:
      normalize: false
      hidden_units: 256
      vis_encode_type: nature_cnn     # Sẽ được thay bằng custom encoder
```

## 🔧 Cách tùy chỉnh Config

### 1. **Tùy chỉnh Pre-training Parameters**

Sửa `dino_workflow_config.yaml`:

```yaml
pretraining:
  epochs: 100              # Tăng epochs cho better quality
  batch_size: 64           # Tăng batch size nếu có GPU mạnh
  model:
    embed_dim: 768         # Tăng model size
    depth: 12              # Model sâu hơn
    num_heads: 12          # Nhiều attention heads hơn
```

### 2. **Điều chỉnh theo GPU Memory**

| GPU Memory | Batch Size | Model Size | Embed Dim | Depth |
|------------|------------|------------|-----------|-------|
| 8GB        | 16-32      | Small      | 192-384   | 3-6   |
| 16GB       | 32-64      | Medium     | 384-512   | 6-9   |
| 24GB+      | 64-128     | Large      | 512-768   | 9-12  |

### 3. **Tùy chỉnh Data Collection**

```yaml
data_collection:
  episodes: 500            # Nhiều data hơn
  steps_per_episode: 1000  # Episodes dài hơn
  additional_envs:         # Thu thập từ nhiều environments
    - FogTrain
    - RandomTrain
    - CustomEnv
```

### 4. **Fine-tuning Settings**

```yaml
finetuning:
  architectures:
    - dino_encoder         # Architecture chính
    - dino_resnet_hybrid   # Thêm architectures khác để so sánh
  runs_per_network: 5      # Nhiều runs để có kết quả stable
  unfreeze_layers: 4       # Fine-tune nhiều layers hơn
```

## 📍 Vị trí các Config Files

```
mouse_vs_ai_windows/
├── dino_workflow_config.yaml           # ⭐ MAIN CONFIG
├── dino_config.yaml                    # Auto-generated từ main config
├── resnet50_workflow_config.yaml       # Config cho ResNet50
├── Config/
│   ├── nature.yaml                     # Base ML-Agents config
│   ├── dinov3_train.yaml              # DINOv3 specific
│   ├── resnet50_config.yaml           # ResNet50 specific
│   └── ...
└── config_examples/
    ├── dino_finetuning.yaml           # Auto-generated finetuning config
    └── resnet50_finetuning.yaml       # Auto-generated ResNet config
```

## 🚀 Cách sử dụng Config

### 1. **Sử dụng default config**:
```bash
# Tự động tạo dino_workflow_config.yaml với default values
python dino_workflow.py --step all
```

### 2. **Sử dụng custom config**:
```bash
# Tạo custom config file
cp dino_workflow_config.yaml my_custom_config.yaml
# Sửa parameters trong my_custom_config.yaml

# Chạy với custom config
python dino_workflow.py --config my_custom_config.yaml --step all
```

### 3. **Override config parameters**:
```bash
# Sử dụng script riêng với custom parameters
python run_pretrain_only.py --epochs 100 --batch-size 64
```

## 🎯 Config Templates

### **Small/Fast Config** (Testing):
```yaml
pretraining:
  epochs: 20
  batch_size: 16
  model:
    embed_dim: 192
    depth: 3
    num_heads: 3
data_collection:
  episodes: 50
```

### **Medium Config** (Balanced):
```yaml
pretraining:
  epochs: 50
  batch_size: 32
  model:
    embed_dim: 384
    depth: 6
    num_heads: 6
data_collection:
  episodes: 200
```

### **Large Config** (Best Quality):
```yaml
pretraining:
  epochs: 100
  batch_size: 64
  model:
    embed_dim: 768
    depth: 12
    num_heads: 12
data_collection:
  episodes: 500
```

## 💡 Tips cho Config

### 1. **Start Small, Scale Up**:
- Bắt đầu với small config để test pipeline
- Tăng dần parameters khi đã ổn định

### 2. **Monitor GPU Usage**:
```bash
# Theo dõi GPU memory
nvidia-smi -l 5

# Giảm batch_size nếu out of memory
```

### 3. **Backup configs**:
```bash
# Backup config trước khi modify
cp dino_workflow_config.yaml dino_workflow_config.yaml.backup
```

### 4. **Version Control**:
- Commit config changes vào git
- Tag các config versions quan trọng

## 🔍 Debug Config Issues

### Config file không tồn tại:
```bash
# Tạo default config
python -c "
from dino_workflow import DINOWorkflow
workflow = DINOWorkflow()
print('✅ Default config created!')
"
```

### Config syntax errors:
```bash
# Validate YAML syntax
python -c "
import yaml
with open('dino_workflow_config.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Config syntax valid!')
"
```

### Wrong parameters:
- Kiểm tra logs khi chạy training
- So sánh với working configs
- Test với minimal config trước

**🎯 Main Config File: `dino_workflow_config.yaml` - Đây là file quan trọng nhất cần tùy chỉnh!**

