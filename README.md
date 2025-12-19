# DualGCN-Separation: 基于双路径图神经网络的图像源分离框架

本项目实现了一种先进的深度学习框架，旨在解决复杂的**图像源分离 (Image Source Separation)** 问题。模型采用了 **Dual-Path GCN (双路径图卷积网络)** 架构，能够同时捕获图像的**全局语义拓扑**和**局部细节特征**，并通过多头注意力机制进行深度融合。

配套的训练框架包含了一套组合损失函数（感知损失、结构损失、正交损失等），确保分离出的图像在视觉质量和信号独立性上均达到最优。

---

## 🌟 核心特性 (Key Features)

### 🧠 先进的网络架构
* **双流图网络 (Dual-Stream GNN)**:
    * **Global Path**: 使用 `GINConv` (图同构网络) 提取全局长距离依赖特征。
    * **Local Path**: 使用 `ChebConv` (切比雪夫谱卷积) 提取局部邻域的细粒度特征。
* **注意力融合 (Attention Fusion)**: 集成 `MultiHeadAttentionModule`，实现全局与局部特征的动态交互与对齐。
* **端到端设计**: 包含 CNN 编码器与解码器，通过学习掩码 (Mask) 实现像素级的源分离。

### 📉 强大的复合损失函数
为了解决传统 MSE 损失导致的图像模糊问题，本项目引入了 `ImageSeparationLoss`：
* **Perceptual Loss (VGG16)**: 提升图像的语义真实感和纹理细节。
* **Multi-Scale Sobel Loss**: 强制模型关注边缘结构，保持图像锐度。
* **Source Correlation Loss**: 最小化分离源之间的统计相关性，减少串扰。
* **Mask Orthogonality Loss**: 鼓励不同源的掩码在空间上互斥（不重叠）。
* **PIT (Permutation Invariant Training)**: 自动解决源分离中的顺序排列歧义问题。

---

## 🛠️ 环境依赖 (Requirements)

本项目基于 PyTorch 和 PyTorch Geometric (PyG) 构建。

### 基础依赖
```bash
pip install torch torchvision numpy scipy kornia

```

### 安装 PyTorch Geometric

由于 PyG 依赖于特定的 CUDA 版本，建议使用官方推荐命令安装（以下以 CUDA 11.8 为例）：

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f [https://data.pyg.org/whl/torch-2.0.0+cu118.html](https://data.pyg.org/whl/torch-2.0.0+cu118.html)

```

---

## 🚀 快速开始 (Quick Start)

### 1. 模型初始化

```python
import torch
from model import DualGCN

# 配置参数
num_features = 1024  # GNN 特征维度
num_spks = 2         # 需要分离的源数量 (例如：背景与前景)

# 初始化模型并移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualGCN(
    num_features_global=num_features, 
    num_features_local=num_features, 
    num_spks=num_spks
).to(device)

print(f"Model initialized on {device}")

```

### 2. 训练与损失计算

```python
from loss import ImageSeparationLoss

# 初始化复合损失函数
criterion = ImageSeparationLoss(
    perceptual_weight=0.5,   # 感知损失权重
    structure_weight=1.0,    # 边缘结构损失权重
    correlation_weight=1.0,  # 源独立性权重
    orthogonality_weight=0.5 # 掩码正交权重
).to(device)

# --- 模拟训练步骤 ---
# 假设输入: 
# x: [Batch, 3, H, W] 混合图像
# global_data, local_data: PyG 的 Data 对象 (包含 x, edge_index, batch)
# ref_imgs: list of [Batch, 3, H, W] 原始参考图像

# 1. 前向传播
estimated_sources = model(x, global_data, local_data)

# 2. 准备 Ground Truth 字典
targets = {
    "mix": x,
    "ref": ref_imgs  # [source1, source2]
}

# 3. 计算损失 (自动处理 PIT 排列问题)
loss = criterion(estimated_sources, targets)

# 4. 反向传播
loss.backward()
print(f"Training Loss: {loss.item()}")

```

---

## 📊 性能指标 (Metrics)

项目内置了工业级图像质量评价指标：

* **PSNR (Peak Signal-to-Noise Ratio)**: 衡量图像重建的像素误差。
* **SSIM (Structural Similarity Index)**: 衡量图像结构的相似程度。

```python
from utils import psnr, ssim

score_psnr = psnr(estimated_img, target_img)
score_ssim = ssim(estimated_img, target_img)

```

---



```

