# 🎮 DINO ResNet50 RL Training Guide

## 🔥 Pre-trained → RL Fine-tuning

### ✅ Files đã setup:
- `📄 Config/dino_resnet50_rl.yaml` - ML-Agents config
- `📄 Encoders/dino_resnet50_encoder.py` - Encoder with pre-trained weights
- `📄 train_dino_resnet_rl.py` - Training script

## 🚀 Commands để chạy:

### Option 1: Simple (Recommended)
```bash
python train_dino_resnet_rl.py
```

### Option 2: Custom parameters
```bash
python train_dino_resnet_rl.py --runs 5 --max-steps 100000
```

### Option 3: Direct ML-Agents
```bash
set CUSTOM_ENCODER=dino_resnet50_encoder
mlagents-learn Config/dino_resnet50_rl.yaml --env=Builds/NormalTrain/2D go to target v1.exe --run-id=DinoResNet50_Test --max-steps=50000
```

## 🔧 Config Details:

```yaml
# Config/dino_resnet50_rl.yaml
hyperparameters:
  learning_rate: 0.0001     # Lower for pre-trained model
  batch_size: 256           # Larger batch (ResNet efficient)
  buffer_size: 4096         # More experience replay

network_settings:
  vis_encode_type: simple   # Overridden by custom encoder

# Auto-load pre-trained weights from:
# ./dino_resnet_checkpoints/best_checkpoint.pth
```

## 📊 What happens:

1. **Load DINO ResNet50**: Pre-trained weights from self-supervised learning
2. **Fine-tune on RL**: Adapt features for mouse navigation task
3. **Lower Learning Rate**: Preserve pre-trained features
4. **Differential LR**: Backbone vs head learning rates

## 🎯 Expected Results:

| Metric | Random Init | DINO Pre-trained | Improvement |
|--------|-------------|------------------|-------------|
| Convergence | 30-40k steps | 15-25k steps | ~40% faster |
| Success Rate | 75-85% | 85-95% | +10-15% |
| Training Time | 3-4 hours | 2-3 hours | ~25% faster |

## 🔍 Monitor Training:

```bash
# Real-time GPU usage
nvidia-smi -l 5

# Training logs
tail -f results/DinoResNet50_*/run_logs/Player-0.log

# TensorBoard
tensorboard --logdir results/
```

## 🎉 After Training:

```bash
# Check results
dir results\

# Best model weights
dir results\DinoResNet50_*\My Behavior\

# Compare with baseline
python evaluate_models.py --pretrained results/DinoResNet50_* --baseline results/Baseline_*
```

**🚀 Ready to fine-tune DINO ResNet50 on RL task!**

