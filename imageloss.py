import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
import torchvision.models as models
import kornia.filters as KF
import torchvision.transforms.functional as TF



def multi_scale_sobel_loss(est_img, ref_img, levels=[1, 2]):
    total_loss = 0.0
    weight = 1.0
    total_weight = 0.0

    for level in levels:
        est_scaled = F.avg_pool2d(est_img, kernel_size=2**level)
        ref_scaled = F.avg_pool2d(ref_img, kernel_size=2**level)
        est_edge = KF.sobel(est_scaled)
        ref_edge = KF.sobel(ref_scaled)

        loss = F.l1_loss(est_edge, ref_edge, reduction='mean')
        total_loss += weight * loss
        total_weight += weight
        weight /= 2  # 每个尺度的影响逐层减弱

    return total_loss / total_weight


def sobel_edge_map(img):  # img: [B, C, H, W]
    if img.shape[1] == 3:
        gray = TF.rgb_to_grayscale(img)
    else:
        gray = img  # already grayscale

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge  # [B, 1, H, W]


def psnr(x, s, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    input:
          x: separated image
          s: reference image
          max_val: maximum value of the images
    Return:
          psnr value
    """
    mse = torch.mean((x - s) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))


def ssim(x, s, window_size=11, size_average=True, val_range=None):
    """
    Calculate Structural Similarity Index (SSIM)
    input:
          x: separated image
          s: reference image
          window_size: size of the gaussian window
          size_average: if True, average SSIM across batch
          val_range: value range of input images (usually 1.0 or 255)
    Return:
          ssim value
    """
    # Implementation of SSIM
    if val_range is None:
        val_range = 1.0

    # SSIM constants
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    # Calculate mean
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(s, window_size, stride=1, padding=window_size // 2)

    # Calculate variance
    sigma_x = F.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(s ** 2, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * s, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    # Calculate SSIM
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
# def source_correlation_loss(sources):
#     """
#     计算源信号之间的相关性损失
#     这个损失鼓励分离出的源信号之间相互独立（低相关性）
#
#     参数:
#         sources: 分离出的源信号列表 [source1, source2, ...]
#
#     返回:
#         correlation_loss: 源信号之间的相关性损失
#     """
#     num_sources = len(sources)
#     if num_sources < 2:
#         return torch.tensor(0.0, device=sources[0].device)
#
#     total_corr = 0.0
#     count = 0
#
#     # 计算每对源信号之间的相关性
#     for i in range(num_sources):
#         for j in range(i+1, num_sources):
#             # 将图像展平为向量
#             source_i_flat = sources[i].view(sources[i].size(0), -1)
#             source_j_flat = sources[j].view(sources[j].size(0), -1)
#
#             # 归一化向量
#             source_i_norm = source_i_flat - source_i_flat.mean(dim=1, keepdim=True)
#             source_j_norm = source_j_flat - source_j_flat.mean(dim=1, keepdim=True)
#
#             source_i_std = torch.sqrt(torch.var(source_i_norm, dim=1, keepdim=True) + 1e-8)
#             source_j_std = torch.sqrt(torch.var(source_j_norm, dim=1, keepdim=True) + 1e-8)
#
#             # 计算相关系数
#             corr = torch.abs(torch.sum(source_i_norm * source_j_norm, dim=1) /
#                             (source_i_std * source_j_std).squeeze() / source_i_flat.size(1))
#
#             total_corr += corr.mean()
#             count += 1
#
#     # 返回平均相关系数作为损失
#     return total_corr / count if count > 0 else torch.tensor(0.0, device=sources[0].device)
def source_correlation_loss(sources):
    """
    sources: tensor of shape [B, num_sources, C, H, W]
    """
    if sources.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B, num_sources, C, H, W], but got shape: {sources.shape}")

    B, N, C, H, W = sources.shape
    total_corr = 0.0
    count = 0

    for i in range(N):
        for j in range(i + 1, N):
            source_i = sources[:, i]  # [B, C, H, W]
            source_j = sources[:, j]  # [B, C, H, W]

            # 展平为向量：[B, C*H*W]
            source_i_flat = source_i.view(B, -1)
            source_j_flat = source_j.view(B, -1)

            # 零均值
            source_i_norm = source_i_flat - source_i_flat.mean(dim=1, keepdim=True)
            source_j_norm = source_j_flat - source_j_flat.mean(dim=1, keepdim=True)

            # 标准差
            source_i_std = torch.sqrt(torch.var(source_i_norm, dim=1, keepdim=True) + 1e-8)
            source_j_std = torch.sqrt(torch.var(source_j_norm, dim=1, keepdim=True) + 1e-8)

            # 余弦相关性（绝对值）
            corr = torch.abs(
                torch.sum(source_i_norm * source_j_norm, dim=1) /
                (source_i_std.squeeze() * source_j_std.squeeze() * source_i_flat.size(1))
            )  # [B]

            total_corr += corr.mean()
            count += 1

    return total_corr / count if count > 0 else torch.tensor(0.0, device=sources.device)


def color_loss(x, s):
    """计算RGB颜色差异损失，更注重色度通道"""
    # 将RGB分解为单独的通道
    x_r, x_g, x_b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    s_r, s_g, s_b = s[:, 0:1], s[:, 1:2], s[:, 2:3]
    
    # 计算每个通道的损失，给予绿色和蓝色通道更高的权重
    # 人眼对绿色和蓝色的变化更敏感
    r_loss = F.mse_loss(x_r, s_r)
    g_loss = F.mse_loss(x_g, s_g)
    b_loss = F.mse_loss(x_b, s_b)
    
    # 加权组合
    return r_loss * 0.3 + g_loss * 0.4 + b_loss * 0.3
    
def permute_quality(s_lists, ref_lists, quality_fn=psnr):
    """
    Calculate quality metric for all permutations and find max
    input:
          s_lists: list of separated images
          ref_lists: list of reference images
          quality_fn: quality metric function (psnr or ssim)
    Return:
          max quality value
    """
    length = len(s_lists)
    results = []
    for p in permutations(range(length)):
        s_list = [ref_lists[n] for n in p]
        result = sum([quality_fn(s, r) for s, r in zip(s_lists, s_list)]) / length
        results.append(result)
    return max(results)


class PerceptualLoss(nn.Module):
    """
    Correct implementation of perceptual loss with VGG16 features.
    Features are extracted in a single forward pass without redundant computation.
    """
    def __init__(self, feature_layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 features
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Register mean/std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Validate layer indices (example indices for ReLU layers)
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Check if layers are valid
        max_layer = max(feature_layers)
        if max_layer >= len(self.vgg):
            raise ValueError(f"Layer index {max_layer} exceeds VGG depth {len(self.vgg)}")

    def forward(self, input, target):
        # Normalize using dynamic device
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)
        
        # Initialize loss
        loss = 0.0
        
        # Extract features in a single forward pass
        x, y = input, target
        features_x, features_y = [], []
        current_layer = 0
        
        for layer_idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            
            # Check if current layer is a target feature layer
            if layer_idx in self.feature_layers:
                features_x.append(x)
                features_y.append(y)
        
        # Compute weighted MSE loss for each feature layer
        for fx, fy, weight in zip(features_x, features_y, self.weights):
            loss += torch.log1p(weight * F.mse_loss(fx, fy))
        
        return loss


class ImageSeparationLoss(nn.Module):
    """
    Loss function for image separation
    Combines KL-Divergence, L1, L2, and Perceptual loss with permutation-invariant training
    """

    def __init__(self, kl_weight=1, l1_weight=1, l2_weight=1, perceptual_weight=0.5, correlation_weight=1, structure_weight=1.0, orthogonality_weight=0.5): #新增struc...
        super(ImageSeparationLoss, self).__init__()
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = PerceptualLoss()
        self.correlation_weight = correlation_weight
        self.structure_weight = structure_weight  # 用于边缘保持（Sobel Loss）   ✅ 更清晰
        self.orthogonality_weight = orthogonality_weight  # ✅ 新增
        self.fix_order = True

    def forward(self, ests, egs, masks=None):
        """
        Calculate loss for image separation
        input:
              ests: list of estimated images
              egs: dictionary containing mixed image and reference images
        Return:
              loss: combined loss
        """
        refs = egs["ref"]
        num_sources = len(refs)
        
        # 计算源信号相关性损失
        corr_loss = source_correlation_loss(ests)


        if self.fix_order:
            # 使用固定的顺序 [0,1] - 即ests[0]对应refs[0]，ests[1]对应refs[1]
            fixed_permute = list(range(num_sources))
            reconstruction_loss = self._calculate_loss(ests, refs, fixed_permute)
            
            # 添加源信号相关性正则项
            combined_loss = reconstruction_loss + self.correlation_weight * corr_loss
            
            return combined_loss
        else:
            # 原始排列不变训练逻辑
            def separation_loss(permute):
                return self._calculate_loss(ests, refs, permute)
            
            N = egs["mix"].size(0)
            loss_mat = torch.stack([separation_loss(p) for p in permutations(range(num_sources))])
            min_loss, _ = torch.min(loss_mat, dim=0)  # Get min loss (max quality)
            
            # 添加源信号相关性正则项
            combined_loss = torch.sum(min_loss) / N + self.correlation_weight * corr_loss
            
            return combined_loss

    # def _calculate_loss(self, ests, refs, permute):
    #     # KL-Divergence loss
    #     kl_values = []
    #     # L1 loss
    #     l1_values = []
    #     # L2 loss (MSE)
    #     l2_values = []
    #     # Perceptual loss
    #     perceptual_values = []
    #
    #     for s, t in enumerate(permute):
    #         # If batched, keep batch dimension, otherwise add one
    #         est_img = ests[s] if ests[s].dim() == 4 else ests[s].unsqueeze(0)
    #         ref_img = refs[t] if refs[t].dim() == 4 else refs[t].unsqueeze(0)
    #
    #         # Calculate L1 loss
    #         l1_loss = F.l1_loss(est_img, ref_img)
    #         l1_values.append(l1_loss)
    #
    #         # Calculate L2 loss (MSE)
    #         l2_loss = F.mse_loss(est_img, ref_img)
    #         l2_values.append(l2_loss)
    #
    #         # Calculate perceptual loss
    #         perceptual_values.append(self.perceptual_loss(est_img, ref_img))
    #
    #     # Average the loss values
    #     kl_loss = sum(kl_values) / len(permute)
    #     l1_loss = sum(l1_values) / len(permute)
    #     l2_loss = sum(l2_values) / len(permute)
    #     perceptual_loss = sum(perceptual_values) / len(permute)
    #
    #     # Combined loss with all terms
    #     combined_loss = (self.perceptual_weight * perceptual_loss + self.l1_weight * l1_loss +self.l2_weight * l2_loss
    #     )
    #
    #     return combined_loss


    def _calculate_loss(self, ests, refs, permute, masks=None):
        l1_values = []
        l2_values = []
        perceptual_values = []
        sobel_values = []  # ✅ 新增：用于存储每对图像的结构损失

        max_index_est = len(ests) - 1
        max_index_ref = len(refs) - 1

        for s, t in enumerate(permute):
            if s > max_index_est or t > max_index_ref:
                continue

            est_img = ests[s]
            ref_img = refs[t]

            # [B, C, H, W]
            if est_img.dim() == 3:
                est_img = est_img.unsqueeze(0)
            if ref_img.dim() == 3:
                ref_img = ref_img.unsqueeze(0)

            batch_size = min(est_img.size(0), ref_img.size(0))
            est_img = est_img[:batch_size]
            ref_img = ref_img[:batch_size]

            l1_values.append(F.l1_loss(est_img, ref_img, reduction='mean'))
            l2_values.append(F.mse_loss(est_img, ref_img, reduction='mean'))
            perceptual_values.append(self.perceptual_loss(est_img, ref_img))  # 注意确保它带梯度

            # # ✅ 新增：结构感知（Sobel）Loss
            # est_edge = KF.sobel(est_img)
            # ref_edge = KF.sobel(ref_img)
            # sobel_values.append(F.l1_loss(est_edge, ref_edge, reduction='mean'))
            # 改成多尺度结构感知 loss：
            sobel_values.append(multi_scale_sobel_loss(est_img, ref_img))

        if len(l1_values) == 0:
            return ests[0].sum() * 0  # 保持计算图结构

        l1_loss = torch.stack(l1_values).mean()
        l2_loss = torch.stack(l2_values).mean()
        perceptual_loss = torch.stack(perceptual_values).mean()
        sobel_loss = torch.stack(sobel_values).mean()  # ✅ 新增：结构损失均值

        combined_loss = (
                self.perceptual_weight * perceptual_loss +
                self.l1_weight * l1_loss +
                self.l2_weight * l2_loss +
                self.structure_weight * sobel_loss  ###新增
        )
        # ✅ 新增：掩码正交损失
        if masks is not None:
            orth_loss = self.mask_orthogonality_loss(masks)
            combined_loss += self.orthogonality_weight * orth_loss

        return combined_loss

        # ===== 在这里加入 ↓↓↓ =================================

    def mask_orthogonality_loss(self, masks):
        """
        Encourage different masks to be non‑overlapping (orthogonal in space)
        masks: [B, num_spks, C, H, W]
        """
        B, N, C, H, W = masks.shape
        masks_flat = masks.view(B, N, -1)  # [B, N, C*H*W]
        loss = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dot = (masks_flat[:, i] * masks_flat[:, j]).sum(dim=1)  # [B]
                loss += dot.mean()
        return loss / (N * (N - 1) / 2)  # normalize

    # ===== 结束 ========================================

    def set_fixed_order(self, use_fixed_order=False):
        """设置是否使用固定顺序"""
        self.fix_order = use_fixed_order


# For testing
if __name__ == "__main__":
    # Create test tensors
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)

    # Calculate metrics
    psnr_val = psnr(img1, img2)
    ssim_val = ssim(img1, img2)

    print(f"PSNR: {psnr_val.item():.2f} dB")
    print(f"SSIM: {ssim_val.item():.4f}")

    # Test perceptual loss
    percept_loss = PerceptualLoss()
    perceptual_val = percept_loss(img1, img2)
    print(f"Perceptual Loss: {perceptual_val.item():.4f}")

    # Test combined loss function
    loss_fn = ImageSeparationLoss()
    ests = [img1, img2]
    egs = {"mix": torch.rand(1, 3, 256, 256), "ref": [img1, img2]}
    loss = loss_fn(ests, egs)
    print(f"Combined Loss: {loss.item():.4f}")

    # Test with different weights
    loss_fn2 = ImageSeparationLoss(kl_weight=0.4, l1_weight=0.3, l2_weight=0.2, perceptual_weight=0.1)
    loss2 = loss_fn2(ests, egs)
    print(f"Combined Loss (custom weights): {loss2.item():.4f}")