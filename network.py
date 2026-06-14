      
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        value = self.fc2(x2)
        return value

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        probs = F.softmax(x3, dim=-1)
        return probs
    
class ScaledDotProductAttention(nn.Module):
    '''
    核心:
        在forward中实现
            U = softmax(Q * K^T/sqrt(d_k))
            output = softmax(Q * K^T / sqrt(d_k)) * V
    '''
    def __init__(self, d_k=None):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        if d_k is not None:
            self.scale = 1 / np.sqrt(d_k)
        else:
            self.scale = None

    def forward(self, Q, K, V, mask=None):
        U = torch.bmm(Q, K.transpose(1, 2))
        if self.scale is not None:
            U = U * self.scale
        else:
            U = U / np.sqrt(K.size(-1))
        
        if mask is not None:
            U = U.masked_fill(mask, -1e9)
        
        A = self.softmax(U)
        O = torch.bmm(A, V)
        return A, O

class MultiHeadAttention(nn.Module):
    '''
    核心:
    实现单头变多头(Linear)/维度变换(permute)/多头拼接(concat)/仿射变换
    '''
    def __init__(self, d_model, d_v_, d_q, d_k, d_v, n_head, batch_size):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # Q, K原始输入维度(模型总维度)
        self.d_v_ = d_v_
        self.d_k = d_k
        self.d_q = d_q
        self.d_v = d_v
        self.n_head = n_head
        self.batch_size = batch_size
        self.attn = ScaledDotProductAttention()
        assert(self.d_k == self.d_q)    # 一般情况下，d_k和d_q相等
        self.fc_q = nn.Linear(self.d_model, self.n_head * self.d_q)
        self.fc_k = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.fc_v = nn.Linear(self.d_model, self.n_head * self.d_v)
    def forward(self, Q, K, V, mask=None):
        n_q = Q.size(1)
        n_k = K.size(1)
        n_v = V.size(1)
        
        Q = self.fc_q(Q)
        Q = Q.view(self.batch_size, n_q, self.n_head, self.d_q)
        Q = Q.permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_q)

        K = self.fc_k(K)
        K = K.view(self.batch_size, n_k, self.n_head, self.d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)

        V = self.fc_v(V)
        V = V.view(self.batch_size, n_v, self.n_head, self.d_v)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)  # (n_head * batch_size, n_v, d_v)

        A, O = self.attn(Q, K, V, mask)
        # A = A.view(self.n_head, self.batch_size, n_q, n_k).permute(1, 2, 0, 3)
        O = O.view(self.n_head, self.batch_size, n_q, self.d_v) \
              .permute(1, 2, 0, 3).contiguous() \
              .view(self.batch_size, n_q, self.n_head * self.d_v)
        return A, O

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_v_, d_q, d_k, d_v, n_head, batch_size):
        super(SelfAttention, self).__init__()
        self.n_head = n_head
        self.batch_size = batch_size
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(self.d_model, self.d_q)
        self.W_k = nn.Linear(self.d_model, self.d_k)
        self.W_v = nn.Linear(self.d_model, self.d_v)
        self.mha = MultiHeadAttention(d_model, d_v_, d_q, d_k, d_v, n_head, batch_size)
    
    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        A, O = self.mha(Q, K, V, mask)
        return A, O

class FeedForwardNetwork(nn.Module):
    '''
    前馈网络:
        dropout层为了防止过拟合，随机丢弃一些神经元，提升鲁棒性/稳定性
        Linear层先升维后降维，升到高维空间容纳复杂特征/特征解耦;
        降到原始维度，提取高维信息传递给下一层
    '''
    def __init__(self, d_model, d_ff, d_dropout):
        super(FeedForwardNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(d_dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(d_dropout)
        )
        
    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out

class FrameFeatureExtractor(nn.Module):
    '''
    核心:
        实现轨迹信息提取网络，使用CNN和MHA。
        1. 卷积层提取局部特征，捕捉空间关系
        2. MHA提取特征信息，捕捉全局和时序关系
    '''
    def __init__(self, batch_size, seq_len, input_height, input_weight, feature_dim, d_model=128, n_head=8, input_channels=3):
        super(FrameFeatureExtractor, self).__init__()
        self.input_width = input_weight
        self.input_height = input_height
        self.cnn = SimpleCNN(input_channels, self.input_width, self.input_height, feature_dim)
        self.batch_size = batch_size
        self.seq_len = seq_len    # 和输入数据的维度(帧数)有关
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = self.d_model // self.n_head
        self.d_q = self.d_k
        # 在self-attention中, d_k == d_q, 这里为方便, 把d_v也设置成一样的值
        self.d_v = self.d_model // self.n_head
        self.mha = MultiHeadAttention(self.d_model, self.d_model, self.d_q, self.d_k, self.d_v, n_head, self.batch_size)
        # 处理原始数据需要投影
        self.patch_embedding = nn.Linear(feature_dim, d_model)  # 处理CNN输出的数据
        # 处理时序数据需要位置编码
        # nn.Parameter由pytorch的自动求导和优化器机制实现, 无内置运算逻辑
        # 因此必须加入计算图
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))
        self.global_flat = nn.Linear(self.d_model * seq_len, feature_dim)   # 展平层，将提取特征映射到feature_dim维度
    def forward(self, x, visualize=False):
        # 1. CNN提取局部特征
        b, c, h, w = x.shape    # TODO: x添加patch信息
        cnn_out, x3, x2, x1 = self.cnn(x)
        
        # ================visulize start================
        if visualize:
            self.cnn.visualize(x1)
            self.cnn.visualize(x2)
            self.cnn.visualize(x3)
        # ================visualize  end================
        # 2. 投影、位置编码，为MHA准备
        
        cnn_out = cnn_out.permute(0, 2, 3, 1).contiguous().view(b, h*w, c) # (b, c, h, w)->(b, h*w, c)
        embedding_out = self.patch_embedding(cnn_out) + self.pos_embedding
        # 3. MHA提取特征
        mha_out, _ = self.mha(embedding_out)
        # 4. reshape/flat展平输出
        mha_out = mha_out.view(b, -1)  # 展平多头注意力输出
        out = self.global_flat(mha_out) # (batch_size, feature_dim)
        return out
