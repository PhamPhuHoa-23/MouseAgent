# 🔥 ResNet50 Grad-CAM Visualization Guide

## 📋 Tổng quan

Dự án này cung cấp các script để visualize attention maps của ResNet50 sử dụng các kỹ thuật:

- **Grad-CAM**: Gradient-based Class Activation Mapping
- **Grad-CAM++**: Improved version cho multiple objects  
- **Layer-wise CAM**: Apply ở nhiều conv layers khác nhau

## 🏗️ ResNet50 Architecture

ResNet50 có kiến trúc CNN với 4 stages chính:

```
Input (224×224×3)
    ↓
Stem: conv1 + bn1 + relu + maxpool → 56×56×64
    ↓
Layer1 (conv2_x): 3 Bottleneck blocks → 56×56×256
    ↓
Layer2 (conv3_x): 4 Bottleneck blocks → 28×28×512
    ↓
Layer3 (conv4_x): 6 Bottleneck blocks → 14×14×1024
    ↓
Layer4 (conv5_x): 3 Bottleneck blocks → 7×7×2048
    ↓
Global Average Pooling → 1×1×2048
    ↓
Fully Connected → 1000 classes
```

**Tổng số parameters**: ~25.6M

## 🚀 Cách sử dụng

### 1. Phân tích kiến trúc ResNet50

```bash
# Phân tích kiến trúc cơ bản
python resnet50_architecture_analysis.py

# Phân tích + tạo visualization
python resnet50_architecture_analysis.py --visualize

# Phân tích với custom model
python resnet50_architecture_analysis.py --model ./dino_resnet_checkpoints/best_resnet_checkpoint.pth
```

### 2. Grad-CAM visualization cho một ảnh

```bash
# Sử dụng ImageNet pretrained ResNet50
python gradcam_resnet50_visualization.py --image ./dataset_mouse_dino/images/img_000000_ep0_step0_agent0.png

# Sử dụng custom model
python gradcam_resnet50_visualization.py \
    --image ./dataset_mouse_dino/images/img_000000_ep0_step0_agent0.png \
    --model ./dino_resnet_checkpoints/best_resnet_checkpoint.pth

# Chỉ định output directory
python gradcam_resnet50_visualization.py \
    --image ./dataset_mouse_dino/images/img_000000_ep0_step0_agent0.png \
    --output ./my_gradcam_results
```

### 3. Demo Grad-CAM trên nhiều ảnh mouse

```bash
# Chạy demo trên 5 ảnh ngẫu nhiên
python demo_gradcam_mouse_images.py

# Chạy demo với custom model
python demo_gradcam_mouse_images.py --model ./dino_resnet_checkpoints/best_resnet_checkpoint.pth

# Chỉ định số lượng ảnh
python demo_gradcam_mouse_images.py --num-images 10

# Chỉ định dataset directory
python demo_gradcam_mouse_images.py --dataset-dir ./my_mouse_dataset
```

## 📊 Output Files

### Architecture Analysis
- `resnet50_architecture.png`: Visualization kiến trúc ResNet50

### Grad-CAM Results
```
gradcam_results/
├── image_1/
│   ├── attention_maps_img_000000_ep0_step0_agent0.png
│   └── overlays_img_000000_ep0_step0_agent0.png
├── image_2/
│   ├── attention_maps_img_000001_ep0_step1_agent0.png
│   └── overlays_img_000001_ep0_step1_agent0.png
└── ...
```

## 🎨 Visualization Types

### 1. Attention Maps Grid
- **Row 1**: Original Image, Grad-CAM, Grad-CAM++
- **Row 2**: Layer1, Layer2, Layer3, Layer4 CAMs

### 2. Overlay Visualization
- **Original**: Ảnh gốc
- **Grad-CAM Overlay**: Heatmap overlay với ảnh gốc
- **Grad-CAM++ Overlay**: Improved heatmap overlay

## 🔧 Kỹ thuật Grad-CAM

### Grad-CAM (Gradient-weighted Class Activation Mapping)
```python
# 1. Forward pass để lấy feature maps
features = model.backbone(input_image)

# 2. Backward pass để lấy gradients
output[target_class].backward()

# 3. Global average pooling của gradients
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# 4. Weighted combination
cam = sum(w * f for w, f in zip(pooled_gradients, features))
```

### Grad-CAM++
```python
# Cải tiến weights calculation
alpha_num = gradients.pow(2)
alpha_denom = alpha_num.mul(2) + gradients.mul(features.pow(2).sum(dim=[2, 3]))
alpha = alpha_num.div(alpha_denom + 1e-7)
weights = (alpha * gradients).sum(dim=[2, 3])
```

### Layer-wise CAM
- **Layer1 (conv2_x)**: Early features, low-level patterns
- **Layer2 (conv3_x)**: Mid-level features, textures
- **Layer3 (conv4_x)**: High-level features, shapes
- **Layer4 (conv5_x)**: Final features, semantic information

## 📈 Interpretation

### Attention Map Colors
- **Red/Orange**: High attention regions
- **Blue/Green**: Low attention regions
- **White**: Medium attention regions

### What to Look For
1. **Target Object**: Network có focus vào đúng object không?
2. **Background**: Network có bị distract bởi background không?
3. **Feature Hierarchy**: Early layers học gì vs late layers học gì?

## 🎯 Use Cases

### 1. Model Debugging
- Kiểm tra network có học đúng features không
- Identify bias trong training data
- Debug misclassifications

### 2. Feature Analysis
- Understand what each layer learns
- Compare different architectures
- Analyze transfer learning effectiveness

### 3. Research & Development
- Evaluate self-supervised learning methods
- Compare DINO vs ImageNet pretraining
- Analyze attention patterns in RL tasks

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Sử dụng CPU thay vì GPU
python gradcam_resnet50_visualization.py --device cpu
```

#### 2. Model Loading Failed
```bash
# Kiểm tra checkpoint format
python resnet50_architecture_analysis.py --model ./path/to/checkpoint.pth
```

#### 3. Image Not Found
```bash
# Kiểm tra dataset path
ls ./dataset_mouse_dino/images/
```

### Performance Tips

1. **Batch Processing**: Process nhiều ảnh cùng lúc
2. **Memory Management**: Clear cache sau mỗi image
3. **GPU Optimization**: Sử dụng mixed precision nếu cần

## 📚 References

- **Grad-CAM**: [Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391)
- **Grad-CAM++**: [Chattopadhyay et al. (2018)](https://arxiv.org/abs/1710.11063)
- **ResNet**: [He et al. (2016)](https://arxiv.org/abs/1512.03385)
- **DINO**: [Caron et al. (2021)](https://arxiv.org/abs/2104.14294)

## 🤝 Contributing

Để contribute vào dự án:

1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🆘 Support

Nếu gặp vấn đề:

1. Kiểm tra troubleshooting section
2. Tạo issue trên GitHub
3. Liên hệ maintainer

---

**🎉 Happy visualizing! 🎉**
