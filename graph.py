import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.modules.normalization import LayerNorm
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch.nn.functional as F
from torchvision import models
import cv2
import sys
# sys.path.append("./img_gist_feature/")
# from utils_gist import *

# 定义特征提取函数
class FeatureExtractor:
    def __init__(self):
        # Load a pre-trained ResNet model
        self.resnet_model = models.resnet34(pretrained=True)
        self.resnet_model.eval()  # Set to evaluation mode
        self.color_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        
        
        # Define image transformations

    def resnet_img(self, input_x):
        """
        Extract features from the input image using the pre-trained ResNet model.
        """
        
        # input_x = input_x.unsqueeze(0)
        x = self.resnet_model.conv1(input_x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        x = self.resnet_model.layer2(x)
        # x = self.resnet_model.layer3(x)
        x = torch.flatten(x, 2)
        return x

    def extract_features(self, img_tensor):
        """
        Load an image, transform it, and extract features.
        """
        # Load and transform the image

        # Add a batch dimension and set to inference mode
        x = torch.unsqueeze(img_tensor, dim=0)
        x.requires_grad = False

        # Extract features
        features = self.resnet_img(x).squeeze()
        # return features.detach().numpy()
        return features
'''
def feature_extract(img_tensor):

    # s_img_url = img_path
    gist_helper = GistUtils()
    # np_img = cv2.imread(s_img_url, -1)
    # print("default: rgb")
    if img_tensor.max() <= 1.0:
        img_tensor = img_tensor * 255.0
    np_img = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    np_gist = gist_helper.get_gist_vec(np_img)
    feature = np.array(np_gist)
    #print("shape ", feature.shape)
    return feature
'''
class VGGFeatureExtractor:
    def __init__(self, vgg_model='vgg16', pretrained=True, feature_layers=[3, 8]):
        """
        Initialize VGG feature extractor.

        Args:
            vgg_model (str): The VGG model to use ('vgg16' or 'vgg19')
            pretrained (bool): Whether to use pre-trained weights
            feature_layers (list): Layer indices to extract features from
        """
        # Load a pre-trained VGG model
        if vgg_model == 'vgg16':
            self.vgg_model = models.vgg16(pretrained=pretrained).features
        elif vgg_model == 'vgg19':
            self.vgg_model = models.vgg19(pretrained=pretrained).features
        else:
            raise ValueError(f"Unsupported VGG model: {vgg_model}")

        self.vgg_model.eval()  # Set to evaluation mode

        # Define the desired feature layers to extract
        self.feature_layers = feature_layers

        # Image normalization for ImageNet pre-trained models
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def vgg_features(self, input_x):
        """
        Extract hierarchical features from the input using VGG.

        Args:
            input_x: Input tensor of shape [B, C, H, W]

        Returns:
            Features from the specified layer of VGG
        """
        # Apply normalization
        input_x = self.transforms(input_x)

        # Extract features
        features = []
        x = input_x

        # Extract features from specified layers
        for i, layer in enumerate(self.vgg_model):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)

        # Use the last specified layer's features
        return features[-1]

    def extract_features(self, img_tensor):
        """
        Load an image tensor, transform it, and extract features.

        Args:
            img_tensor: Input image tensor [C, H, W]

        Returns:
            Extracted features
        """
        # Add a batch dimension and set to inference mode
        x = torch.unsqueeze(img_tensor, dim=0)
        x.requires_grad = False

        # Extract features
        features = self.vgg_features(x)

        # Reshape features to be more useful for graph creation
        # Convert from [1, C, H, W] to [C, H*W]
        features = torch.flatten(features, 2)

        return features.squeeze(0)
def create_graph(image_tensor, window_size, step_size, pad_value=0, global_limit=0.3, local_limit=0.8):
    #image = Image.open(fpath)
    #width, height = image.size
    #width_part, height_part = int(width/n), int(height/n)
    # print(width,height)
    #transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # norm = LayerNorm([1, 3, 128, 128])
    # transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # image_tensor = norm(image_tensors).to('cuda:0')
    # transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    if image_tensor.is_cuda:  # 检查张量是否在CUDA设备上
        image_tensor = image_tensor.cpu()  # 如果是，复制到CPU
    image_tensor = image_tensor.squeeze(0)
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    h, w = window_size
    s_h, s_w = step_size
    pad_h = (s_h - (H - h) % s_h) % s_h
    pad_w = (s_w - (W - w) % s_w) % s_w
    pad_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=pad_value)
    H_pad, W_pad = pad_tensor.shape[1], pad_tensor.shape[2]
    n_h = (H_pad - h) // s_h + 1
    n_w = (W_pad - w) // s_w + 1
    # a = channels[0]
    # b = channels[1]
    # c = channels[2]
    # image_list = image_tensor.squeeze().tolist()
    # 初始化下采样模型
    extractor = FeatureExtractor()
    # extractor = VGGFeatureExtractor(vgg_model='vgg16', pretrained=True, feature_layers=[16])
    G_global =nx.Graph()
    G_local = nx.Graph()
    patches =[]
    # features = []
    # 将图像分为16个块并下采样
    # for h in range(0, len(image_list)):
    for i in range(n_h):
        for j in range(n_w):
            y_start = i * s_h
            y_end = y_start + h
            x_start = j * s_w
            x_end = x_start + w
            patch = pad_tensor[:, y_start:y_end, x_start:x_end]  # 保留所有通道
            # 下采样
            # feature = extractor.extract_features(img_tensor=patch)
            # print(feature.shape)
            # 提取特征（这里简单地使用下采样后的图像块作为特征）
            patches.append(patch)
            # features.append(feature)

    for i in range(len(patches)):
        feature = extractor.extract_features(img_tensor=patches[i]).detach().numpy()
        node_name = f'Node_{i}'
        # print(node_name)
        G_global.add_node(node_name, feature=feature)
        G_local.add_node(node_name, feature=feature)


    for node_1 in G_global.nodes():
        for node_2 in G_global.nodes():
            if node_1 != node_2:
                # feature_1 = np.array(G_global.nodes[node_1]['feature'].cpu().detach().numpy())
                # feature_2 = np.array(G_global.nodes[node_2]['feature'].cpu().detach().numpy())
                feature_1 = torch.from_numpy(G_global.nodes[node_1]['feature'])
                feature_2 = torch.from_numpy(G_global.nodes[node_2]['feature'])
                cos_sim = F.cosine_similarity(feature_1, feature_2)[0].item()
                # cos_sim = F.mse_loss(feature_1, feature_2).item()
                # cos_sim = 1 - spatial.distance.cosine(feature_1.flatten(), feature_2.flatten())
                # corr, _ = pearsonr(feature_1.flatten(), feature_2.flatten())
                # correlation_coefficients.append(corr)
                # corr = np.corrcoef(feature_1, feature_2, rowvar=False)[0, 1]
                # cor = find_second_max(corr)
                if cos_sim >= global_limit:
                    G_global.add_edge(node_1, node_2)

    for node_1 in G_local.nodes():
        for node_2 in G_local.nodes():
            if node_1 != node_2:
                feature_1 = torch.from_numpy(G_local.nodes[node_1]['feature'])
                feature_2 = torch.from_numpy(G_local.nodes[node_2]['feature'])
                cos_sim = F.cosine_similarity(feature_1, feature_2)[0].item()
                # mse = F.mse_loss(feature_1, feature_2).item()
                # cos_sim =1 - spatial.distance.cosine(feature_1.flatten(), feature_2.flatten())
                # corr, _ = pearsonr(feature_1.flatten(), feature_2.flatten())
                # corr = np.corrcoef(feature_1, feature_2, rowvar=False)[0, 1]
                if cos_sim >= local_limit:
                    G_local.add_edge(node_1, node_2)

    #print('G_global', G_global)
    # print('G_local', G_local)
    return G_global, G_local
import torch
import torch.nn.functional as F
import networkx as nx

# def build_hyperedge_dict_from_graph(G):
#     """
#     将 networkx 图 G 转换为超边字典 hyperedge_dict
#     每个节点+其邻居作为一个超边
#     """
#     hyperedge_dict = {}
#     for edge_id, node in enumerate(G.nodes()):
#         neighbors = list(G.neighbors(node))
#         # 包含自己和邻居作为一个超边
#         hyper_nodes = [node] + neighbors
#         # 假设节点命名为 'Node_i'，我们提取数字索引 i
#         node_indices = [int(n.split('_')[1]) for n in hyper_nodes]
#         hyperedge_dict[edge_id] = node_indices
#     return hyperedge_dict

#dualgcn_q
# def create_graph(image_tensor, window_size, step_size, pad_value=0, global_limit=0.3, local_limit=0.8):
#     #image = Image.open(fpath)
#     #width, height = image.size
#     #width_part, height_part = int(width/n), int(height/n)
#     # print(width,height)
#     #transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#     # norm = LayerNorm([1, 3, 128, 128])
#     # transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#     # image_tensor = norm(image_tensors).to('cuda:0')
#     # transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
#     if image_tensor.is_cuda:  # 检查张量是否在CUDA设备上
#         image_tensor = image_tensor.cpu()  # 如果是，复制到CPU
#     image_tensor = image_tensor.squeeze(0)
#     H, W = image_tensor.shape[1], image_tensor.shape[2]
#     h, w = window_size
#     s_h, s_w = step_size
#     pad_h = (s_h - (H - h) % s_h) % s_h
#     pad_w = (s_w - (W - w) % s_w) % s_w
#     pad_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=pad_value)
#     H_pad, W_pad = pad_tensor.shape[1], pad_tensor.shape[2]
#     n_h = (H_pad - h) // s_h + 1
#     n_w = (W_pad - w) // s_w + 1
#     # a = channels[0]
#     # b = channels[1]
#     # c = channels[2]
#     # image_list = image_tensor.squeeze().tolist()
#     # 初始化下采样模型
#     extractor = FeatureExtractor()
#     # extractor = VGGFeatureExtractor(vgg_model='vgg16', pretrained=True, feature_layers=[16])
#     G_global =nx.Graph()
#     G_local = nx.Graph()
#     patches =[]
#     # features = []
#     # 将图像分为16个块并下采样
#     # for h in range(0, len(image_list)):
#     for i in range(n_h):
#         for j in range(n_w):
#             y_start = i * s_h
#             y_end = y_start + h
#             x_start = j * s_w
#             x_end = x_start + w
#             patch = pad_tensor[:, y_start:y_end, x_start:x_end]  # 保留所有通道
#             # 下采样
#             # feature = extractor.extract_features(img_tensor=patch)
#             # print(feature.shape)
#             # 提取特征（这里简单地使用下采样后的图像块作为特征）
#             patches.append(patch)
#             # features.append(feature)
#
#     for i in range(len(patches)):
#         feature = extractor.extract_features(img_tensor=patches[i]).detach().numpy()
#
#         node_name = f'Node_{i}'
#         # print(node_name)
#         G_global.add_node(node_name, feature=feature)
#         G_local.add_node(node_name, feature=feature)
#
#
#     for node_1 in G_global.nodes():
#         for node_2 in G_global.nodes():
#             if node_1 != node_2:
#                 # feature_1 = np.array(G_global.nodes[node_1]['feature'].cpu().detach().numpy())
#                 # feature_2 = np.array(G_global.nodes[node_2]['feature'].cpu().detach().numpy())
#                 feature_1 = torch.from_numpy(G_global.nodes[node_1]['feature'])
#                 feature_2 = torch.from_numpy(G_global.nodes[node_2]['feature'])
#                 cos_sim = F.cosine_similarity(feature_1, feature_2, dim=0).item()
#
#                 # cos_sim = F.mse_loss(feature_1, feature_2).item()
#                 # cos_sim = 1 - spatial.distance.cosine(feature_1.flatten(), feature_2.flatten())
#                 # corr, _ = pearsonr(feature_1.flatten(), feature_2.flatten())
#                 # correlation_coefficients.append(corr)
#                 # corr = np.corrcoef(feature_1, feature_2, rowvar=False)[0, 1]
#                 # cor = find_second_max(corr)
#                 if cos_sim >= global_limit:
#                     G_global.add_edge(node_1, node_2)
#
#     for node_1 in G_local.nodes():
#         for node_2 in G_local.nodes():
#             if node_1 != node_2:
#                 feature_1 = torch.from_numpy(G_local.nodes[node_1]['feature'])
#                 feature_2 = torch.from_numpy(G_local.nodes[node_2]['feature'])
#                 cos_sim = F.cosine_similarity(feature_1, feature_2, dim=0).item()
#                 # mse = F.mse_loss(feature_1, feature_2).item()
#                 # cos_sim =1 - spatial.distance.cosine(feature_1.flatten(), feature_2.flatten())
#                 # corr, _ = pearsonr(feature_1.flatten(), feature_2.flatten())
#                 # corr = np.corrcoef(feature_1, feature_2, rowvar=False)[0, 1]
#                 if cos_sim >= local_limit:
#                     G_local.add_edge(node_1, node_2)
#
#     #print('G_global', G_global)
#     # print('G_local', G_local)
#     # return G_global, G_local
#     return G_global, G_local

# import torch
# import torch.nn.functional as F
# import networkx as nx
# from graph import build_hyperedge_dict_from_graph
#
#
# def create_graph(image_tensor, window_size, step_size, pad_value=0, global_limit=0.3, local_limit=0.8):
#     if image_tensor.is_cuda:
#         image_tensor = image_tensor.cpu()
#     image_tensor = image_tensor.squeeze(0)  # [C, H, W]
#     C, H, W = image_tensor.shape
#     h, w = window_size
#     s_h, s_w = step_size
#
#     pad_h = (s_h - (H - h) % s_h) % s_h
#     pad_w = (s_w - (W - w) % s_w) % s_w
#     pad_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=pad_value)
#     H_pad, W_pad = pad_tensor.shape[1], pad_tensor.shape[2]
#     n_h = (H_pad - h) // s_h + 1
#     n_w = (W_pad - w) // s_w + 1
#
#     extractor = FeatureExtractor()  # 你的ResNet extractor
#
#     G_global = nx.Graph()
#     G_local = nx.Graph()
#     patches = []
#
#     for i in range(n_h):
#         for j in range(n_w):
#             y_start = i * s_h
#             y_end = y_start + h
#             x_start = j * s_w
#             x_end = x_start + w
#             patch = pad_tensor[:, y_start:y_end, x_start:x_end]  # shape [C, h, w]
#             patches.append(patch)
#
#     for i, patch in enumerate(patches):
#         # 处理patch特征
#         if patch.dim() != 3:
#             raise ValueError(f"Unexpected patch shape: {patch.shape}")
#
#         # 如果是3通道，使用resnet extractor
#         if patch.size(0) == 3:
#             patch_input = patch.unsqueeze(0)  # [1,3,h,w]
#             with torch.no_grad():
#                 feature = extractor.extract_features(img_tensor=patch_input).squeeze().cpu().numpy()
#         else:
#             # 多通道，直接flatten作为特征
#             feature = patch.flatten().cpu().numpy()
#
#         node_name = f'Node_{i}'
#         G_global.add_node(node_name, feature=feature)
#         G_local.add_node(node_name, feature=feature)
#
#     for G, limit in [(G_global, global_limit), (G_local, local_limit)]:
#         nodes = list(G.nodes())
#         for i in range(len(nodes)):
#             for j in range(i + 1, len(nodes)):
#                 feat1 = torch.from_numpy(G.nodes[nodes[i]]['feature']).float()
#                 feat2 = torch.from_numpy(G.nodes[nodes[j]]['feature']).float()
#                 cos_sim = F.cosine_similarity(feat1, feat2, dim=0).item()
#                 if cos_sim >= limit:
#                     G.add_edge(nodes[i], nodes[j])
#
#     global_hyperedge_dict = build_hyperedge_dict_from_graph(G_global)
#     local_hyperedge_dict = build_hyperedge_dict_from_graph(G_local)
#
#     return G_global, G_local, global_hyperedge_dict, local_hyperedge_dict



from torch_geometric.data import Data
# 新加的 dualgcn_q
# def nx_to_pyg_data(G):
#     # 提取特征矩阵 x
#     features = [torch.from_numpy(G.nodes[n]['feature']).float() for n in G.nodes()]
#     x = torch.stack(features)
#
#     # 构建 edge_index
#     edge_index = []
#     for src, dst in G.edges():
#         src_id = int(src.split('_')[1])
#         dst_id = int(dst.split('_')[1])
#         edge_index.append([src_id, dst_id])
#         edge_index.append([dst_id, src_id])  # 如果是无向图，双向添加
#
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#
#     return Data(x=x, edge_index=edge_index)

def visualize(G_global, G_local):
    plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(G_global, seed=42)
    nx.draw(G_global, pos, node_color='skyblue', with_labels=False, node_size=100, font_size=10, font_color='blue')
    edge_width = 1.5  # 设置边的宽度为一个固定值
    nx.draw_networkx_edges(G_global, pos, width=edge_width, edge_color='gray', alpha=0.7)
    plt.show()

    plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(G_local, seed=42)
    nx.draw(G_local, pos, node_color='lightsalmon', with_labels=False, node_size=100, font_size=10,
            font_color='lightcoral')
    edge_width = 1.5  # 设置边的宽度为一个固定值
    nx.draw_networkx_edges(G_local, pos, width=edge_width, edge_color='gray', alpha=0.7)
    plt.show()
    print('G_global', G_global)
    print('G_local', G_local)


if __name__ == '__main__':
    image_path = r"C:\\Users\\lyd\\Desktop\\bc\\123\\NWPU\\train\\A\\A_20.jpg"
    image = Image.open(image_path)
    transform = transforms.Compose([
                                    transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    # print(image_tensor.shape)
    G_global, G_local = create_graph(image_tensor, window_size=[64, 64], step_size=[32, 32])
    # G_global, G_local = create_graph(image_tensor, window_size=[8, 8], step_size=[8, 8])
    visualize(G_global, G_local)
# transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
# image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

# 可视化有向图（仅显示节点）
# nx.draw(G, with_labels=True, node_size=500, node_color='lightblue', font_size=10, arrowstyle='->', arrowsize=20)
# plt.show()