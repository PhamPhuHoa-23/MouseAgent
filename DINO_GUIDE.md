# 🧠 DINOv2 Self-Supervised Learning Guide

## 🎯 Overview

Tạo **self-supervised visual encoder** với **DINOv2** technique rồi fine-tune cho Mouse vs AI task!

**Pipeline:**
```
Unity Env → Visual Data → DINOv2 Pre-training → RL Fine-tuning → Better Performance! 🚀
```

## 🔄 Complete Workflow

### **⚡ Quick Start (One Command):**
```bash
# Run complete workflow
python dino_workflow.py --step all

# Or step by step
python dino_workflow.py --step collect    # 1. Collect data
python dino_workflow.py --step pretrain   # 2. Pre-train DINOv2  
python dino_workflow.py --step setup      # 3. Setup encoder
python dino_workflow.py --step finetune   # 4. RL fine-tuning
```

### **📊 Step-by-Step Guide:**

## **1. 📸 Data Collection**

Thu thập visual observations từ Unity environments:

```bash
# Collect from single environment
python data_collector.py --env NormalTrain --episodes 100 --steps 500 --output ./dataset_mouse --split

# Collect from multiple environments (more diverse data)
python data_collector.py --env FogTrain --episodes 50 --output ./dataset_fog --split
python data_collector.py --env RandomTrain --episodes 50 --output ./dataset_random --split
```

**💾 Output structure:**
```
dataset_mouse/
├── images/           # Raw images (PNG format)
│   ├── img_000001_ep0_step10_agent0.png
│   └── ...
├── train/           # Training split (80%)
├── val/             # Validation split (20%)  
└── metadata/        # Collection info
    └── collection_info.yaml
```

**🎯 Tips:**
- **Nhiều episodes** = more diverse data
- **Mix environments** = better generalization
- **Random policy** = unbiased data collection

---

## **2. 🧠 DINOv2 Pre-training**

Self-supervised pre-training với DINOv2:

```bash
# Basic pre-training
python dino_pretrain.py --data ./dataset_mouse/train --output ./dino_checkpoints

# With custom config
python dino_pretrain.py --data ./dataset_mouse/train --config dino_config.yaml --output ./dino_checkpoints

# With Weights & Biases logging
python dino_pretrain.py --data ./dataset_mouse/train --wandb --output ./dino_checkpoints
```

**⚙️ DINOv2 Configuration:**
```yaml
# dino_config.yaml
model:
  img_size: 224
  patch_size: 16
  embed_dim: 384      # Model size
  depth: 6           # Transformer layers
  num_heads: 6       # Attention heads

training:
  epochs: 50         # Pre-training epochs
  batch_size: 32     # Adjust for GPU memory
  teacher_momentum: 0.996

optimizer:
  lr: 5e-4          # Learning rate
  weight_decay: 0.04
```

**📊 Expected Training Time:**
- **Small dataset** (10k images): ~2 hours
- **Large dataset** (100k images): ~20 hours  
- **RTX 4060**: ~3-4 images/sec

---

## **3. 🔧 RL Integration**

Sử dụng pre-trained DINOv2 làm encoder:

### **A. Encoder Setup:**
File `Encoders/dino_encoder.py` đã được tạo sẵn với:
- ✅ DINOv2 Vision Transformer backbone
- ✅ Pre-trained weight loading
- ✅ ML-Agents compatibility
- ✅ Fine-tuning options

### **B. Training với DINOv2:**
```bash
# Quick training
python train_easy.py --model dino_encoder --runs 3 --steps 50000 --resume

# Advanced training
python train_advanced.py --config config_examples/dino_finetuning.yaml
```

### **C. Fine-tuning Strategies:**

**🔒 Frozen Features (Fastest):**
```python
# In dino_encoder.py
self.freeze_backbone = True  # Keep backbone frozen
```

**🔓 Partial Fine-tuning (Recommended):**
```python
# Unfreeze last 2 layers
encoder.unfreeze_backbone(unfreeze_last_n_layers=2)
```

**🔥 Full Fine-tuning (Best Performance):**
```python
# Unfreeze all layers
encoder.unfreeze_backbone(unfreeze_last_n_layers=6)  # All layers
```

---

## **4. 📊 Evaluation & Comparison**

So sánh DINOv2 với baselines:

```bash
# Train baseline models
python train_advanced.py --config config_examples/ablation_study.yaml

# Train DINOv2 model  
python train_advanced.py --config config_examples/dino_finetuning.yaml

# Evaluate on test environment
python evaluate.py --model "path/to/dino_model.onnx" --env RandomTest --episodes 100 --log-name "dino_test"
python evaluate.py --model "path/to/baseline_model.onnx" --env RandomTest --episodes 100 --log-name "baseline_test"
```

---

## **💡 Advanced Usage**

### **🎛️ Hyperparameter Tuning:**

```yaml
# config_examples/dino_hyperparameter_sweep.yaml
hyperparameters:
  sweep_params:
    # Fine-tuning specific params
    behaviors.My Behavior.hyperparameters.learning_rate: 
      - 0.00005  # Very low for pre-trained
      - 0.0001   # Standard fine-tuning
      - 0.0003   # Higher for more adaptation
    
    # Unfreeze different amounts
    unfreeze_layers: [1, 2, 3]  # Custom parameter
    
    behaviors.My Behavior.hyperparameters.batch_size: [64, 128, 256]
```

### **🔬 Ablation Studies:**

Test different configurations:

```bash
# 1. Random initialization (no pre-training)
python train_easy.py --model nature_cnn --runs 5

# 2. Frozen DINOv2 features
python train_easy.py --model dino_encoder --runs 5  

# 3. Fine-tuned DINOv2 (last 2 layers)
# Edit dino_encoder.py: encoder.unfreeze_backbone(2)
python train_easy.py --model dino_encoder --runs 5

# 4. Full fine-tuning
# Edit dino_encoder.py: encoder.unfreeze_backbone(6)  
python train_easy.py --model dino_encoder --runs 5
```

---

## **📈 Expected Results**

### **Performance Improvements:**

| Method | Success Rate | Training Time | Notes |
|--------|-------------|---------------|-------|
| Nature CNN (baseline) | ~75% | 2 hours | Random initialization |
| Frozen DINOv2 | ~80% | 1 hour | Fast convergence |
| Fine-tuned DINOv2 | ~85%+ | 3 hours | Best performance |

### **Why DINOv2 Works Better:**

1. **🎯 Better Visual Representations**: Learns general visual features
2. **⚡ Faster Convergence**: Pre-trained features reduce training time  
3. **🎪 Better Generalization**: Works better on FogTrain/RandomTest
4. **📊 Data Efficiency**: Needs less RL training data

---

## **🐛 Troubleshooting**

### **Data Collection Issues:**
```bash
# Check environment exists
ls Builds/NormalTrain/

# Test with fewer episodes first
python data_collector.py --env NormalTrain --episodes 10 --steps 100
```

### **Pre-training Issues:**
```bash
# GPU memory issues - reduce batch size
# Edit dino_config.yaml: batch_size: 16

# Dataset too small warning - collect more data
python data_collector.py --episodes 200 --steps 1000
```

### **RL Training Issues:**
```bash
# Check pre-trained weights loaded
# Look for: "✅ Pre-trained weights loaded successfully!"

# If weights not found, check path:
ls dino_checkpoints/best_checkpoint.pth
```

---

## **🎯 Quick Commands Summary**

```bash
# 🚀 Complete workflow (one command)
python dino_workflow.py

# 📊 Quick comparison
python train_easy.py --model nature_cnn --runs 3    # Baseline
python train_easy.py --model dino_encoder --runs 3   # DINOv2

# 🔍 Analysis
python config_analyzer.py config_examples/dino_finetuning.yaml
```

---

## **📚 Technical Details**

### **DINOv2 Architecture:**
- **Vision Transformer** with 6 layers, 6 heads, 384 embedding dim
- **Patch size**: 16x16 pixels
- **Input resolution**: 224x224 (resized from 155x86)
- **Output**: 384-dimensional features

### **Self-Supervised Loss:**
- **Teacher-Student framework** với EMA updates
- **Cross-entropy loss** between teacher/student predictions
- **Centering mechanism** để tránh collapse

### **Integration Benefits:**
- ✅ **Drop-in replacement** cho existing encoders
- ✅ **GPU accelerated** với CUDA support
- ✅ **Memory efficient** với frozen backbone option
- ✅ **Scalable** to larger datasets/models

---

**🎉 Happy self-supervised learning! Chúc bạn train ra model siêu mạnh!** 🚀
