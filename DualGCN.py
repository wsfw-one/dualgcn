import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, ChebConv, global_add_pool, GINConv
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from scipy.signal import convolve
import pdb
class BiAffineModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BiAffineModule, self).__init__()
        self.W1 = nn.Parameter(torch.randn(output_dim, input_dim))
        self.W2 = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, GLO, LOC):
        # 计算 Hsyn' = softmax(Hsyn * W1 * Hsem^T)
        GLO_transposed = GLO.permute(0, 2, 1)  # 转置 Hsyn 的最后两个维度
        LOC_transposed = LOC.permute(0, 2, 1)  # 转置 Hsem 的最后两个维度
        scores1 = torch.matmul(GLO, torch.matmul(self.W1, LOC_transposed))
        GLO_prime = F.softmax(scores1, dim=1)

        # 计算 Hsem' = softmax(Hsem * W2 * Hsyn^T)
        scores2 = torch.matmul(LOC, torch.matmul(self.W2, GLO_transposed))
        LOC_prime = F.softmax(scores2, dim=1)

        return GLO_prime, LOC_prime


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Parameters for linear transformations in each attention head
        self.W_q = nn.Parameter(torch.randn(num_heads, input_dim, self.head_dim))
        # print('self.W_q',self.W_q.shape)#[2, 64, 32] [64,4096,64]
        self.W_k = nn.Parameter(torch.randn(num_heads, input_dim, self.head_dim))
        # print('self.W_k',self.W_k.shape)#[2, 64, 32]
        self.W_v = nn.Parameter(torch.randn(num_heads, input_dim, self.head_dim))
        # print('self.W_v',self.W_v.shape)#[2, 64, 32]
        # Parameter for output projectiondd
        self.W_o = nn.Parameter(torch.randn(num_heads * self.head_dim, input_dim))

    def forward(self, GLO, LOC):
        # pdb.set_trace()
        GLO = GLO
        LOC = LOC
        # Split input into multi-heads
        # GLO_heads = torch.matmul(GLO, self.W_q)
        # print(GLO_heads.shape)
        # pdb.set_trace()
        GLO_heads = torch.matmul(GLO, self.W_q.to("cuda:0")).view(-1, self.num_heads, GLO.size(1), self.head_dim)
        LOC_heads = torch.matmul(LOC, self.W_k.to("cuda:0")).view(-1, self.num_heads, LOC.size(1), self.head_dim)

        # Calculate scaled dot-product attention for each head
        attn_scores = torch.matmul(GLO_heads, LOC_heads.permute(0, 1, 3, 2)) #[1,64,64,64]
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values in each head
        GLO_heads = torch.matmul(attn_weights, torch.matmul(LOC, self.W_v.to("cuda:0")).view(-1, self.num_heads, LOC.size(1), self.head_dim))

        # Concatenate and project the multi-head outputs
        GLO_prime = GLO_heads.view(-1, GLO.size(1), self.num_heads * self.head_dim)
        GLO_prime = torch.matmul(GLO_prime, self.W_o.to("cuda:0"))

        # Similarly, calculate LOC_prime using LOC as query and GLO as key
        LOC_heads = torch.matmul(LOC, self.W_q.to("cuda:0")).view(-1, self.num_heads,
                                                     LOC.size(1), self.head_dim)
        GLO_heads = torch.matmul(GLO, self.W_k.to("cuda:0")).view(-1, self.num_heads,
                                                     GLO.size(1), self.head_dim)

        attn_scores = torch.matmul(LOC_heads, GLO_heads.permute(0, 1, 3, 2))
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        LOC_heads = torch.matmul(attn_weights,
                                 torch.matmul(GLO, self.W_v.to("cuda:0")).view(-1, self.num_heads, GLO.size(1), self.head_dim))
        #print('LOC_heads',LOC_heads.shape)
        LOC_prime = LOC_heads.view(-1, LOC.size(1), self.num_heads * self.head_dim)
        #print('LOC_prime',LOC_prime.shape)
        LOC_prime = torch.matmul(LOC_prime, self.W_o.to("cuda:0"))
        #print('GLO_prime',GLO_prime.shape,LOC_prime.shape,self.W_o.shape)
        return GLO_prime, LOC_prime


class GINGlobalModel(nn.Module):
    def __init__(self, num_features, dropout=0.3):
        super(GINGlobalModel, self).__init__()

        # GIN layers for global information
        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU()))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(256)
        self.norm2 = LayerNorm(512)
        self.norm3 = LayerNorm(1024)
        self.residual_proj1 = nn.Linear(num_features, 256)
        self.residual_proj2 = nn.Linear(256, 512)
        self.residual_proj3 = nn.Linear(512, 1024)
    def forward(self, data):
        data = data.to("cuda:0")

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = edge_index.to("cuda:0")
        res1 = self.residual_proj1(x)
        x = self.conv1(x, edge_index)
        x = x + res1
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x)
        
        res2 = self.residual_proj2(x)
        x = self.conv2(x, edge_index)
        x = x + res2
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        
        res3 = self.residual_proj3(x)
        x = self.conv3(x, edge_index)
        x = x + res3
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm3(x)

        return x

class ChebLocalModel(nn.Module):
    def __init__(self, num_features, K, dropout=0.3):
        super(ChebLocalModel, self).__init__()

        # ChebNet layers for local information
        self.conv1 = ChebConv(num_features, 256, K)
        self.conv2 = ChebConv(256, 512, K)
        self.conv3 = ChebConv(512, 1024, K)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(256)
        self.norm2 = LayerNorm(512)
        self.norm3 = LayerNorm(1024)
        self.residual_proj1 = nn.Linear(num_features, 256)
        self.residual_proj2 = nn.Linear(256, 512)
        self.residual_proj3 = nn.Linear(512, 1024)
        
    def forward(self, data):
        data = data.to('cuda:0')
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to("cuda:0")
        # ChebNet layer 1 for local information
        res1 = self.residual_proj1(x)
        x = self.conv1(x, edge_index)
        x = x + res1
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x)
        
        res2 = self.residual_proj2(x)
        x = self.conv2(x, edge_index)
        x = x + res2
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)

        # ChebNet layer 2 for local information
        res3 = self.residual_proj3(x)
        x = self.conv3(x, edge_index)
        x = x + res3
        x = F.relu(x)
        x = self.dropout(x)
        # print("x1", x.shape)
        x = self.norm3(x)

        return x


class DualGCN(nn.Module):
    def __init__(self, num_features_global, num_features_local, K=2, num_spks = 2):
        super(DualGCN, self).__init__()

        # self.encoder_1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=48, stride=48, padding=0)
        # self.encoder = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(4, 4), stride=(4, 4))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.relu = nn.LeakyReLU()
        # Global GCN+
        self.global_gcn = GINGlobalModel(num_features_global)

        # Local GCN
        self.local_gcn = ChebLocalModel(num_features_local, K)

        # BiAffine Module for fusion
        # self.biaffine = BiAffineModule(64, 64)  # Adjust input dimensions as needed
        self.num_spks = num_spks
        self.activation = nn.LeakyReLU()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensure output is in [0,1] for images
        )

        # Multi-Head Attention Module for fusion
        self.attention = MultiHeadAttentionModule(1024, num_heads=49)

    def forward(self, x, global_data, local_data):
        # pdb.set_trace()
        # x = x.reshape(1,49152)
        x = self.encoder(x)
        batch_size = x.size(0)
        # x = self.relu(x)
        # x = x.view(x.size(0), x.size(1), -1)
        # pdb.set_trace()
        # x = x.unsqueeze(0)
        global_features = self.global_gcn(global_data)
        local_features = self.local_gcn(local_data)
        GLO_prime, LOC_prime = self.attention(global_features, local_features)
        GLO_LOC_features = torch.cat((GLO_prime, LOC_prime), dim=0)
        A, B, C = GLO_LOC_features.size()
        GLO_LOC_features = GLO_LOC_features.reshape(1, A*B, C)
        m = torch.chunk(GLO_LOC_features, chunks=self.num_spks, dim=1)
        m = self.activation(torch.stack(m, dim=0))
        source_features = []
        for i in range(self.num_spks):
            # Reshape mask to match encoder output dimension
            mask = m[i].view(batch_size, -1, x.size(2), x.size(3))
            # Apply mask to encoded features
            source_feature = x * mask
            source_features.append(source_feature)

        # Decode each source
        outputs = []
        for source_feature in source_features:
            output = self.decoder(source_feature)
            outputs.append(output)

        return outputs

