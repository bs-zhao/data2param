import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
###################
# non-sequential
###################

class NonPositionAttentionAggregator1(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "qkv": nn.Linear(hidden_dim, 3*hidden_dim),
                "proj": nn.Linear(hidden_dim, hidden_dim),
                "norm": nn.LayerNorm(hidden_dim)
            })
            self.layers.append(layer)

    def forward(self, x, mask):
        B, S, _ = x.shape
        
        for layer in self.layers:
            residual = x
            qkv = layer["qkv"](x).reshape(B, S, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2).bool(), -1e9)
            attn = F.softmax(attn, dim=-1)
            
            x = (attn @ v).transpose(1, 2).reshape(B, S, -1)
            x = layer["proj"](x)
            x = layer["norm"](x + residual)
        
        return x.mean(dim=1)
    
    
class NonPositionAttentionAggregator2(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, use_cls_token=True):
        super().__init__()
        
        self.use_cls_token = use_cls_token

        # 可选的全局聚合 token（类似 ViT 的 CLS）
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # (1, 1, r_dim)

        # 多头自注意力
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, r, mask=None):
        """
        r: (batch, n_trial, hidden_dim)  # 每个 context 点的嵌入
        mask: (batch, n_trial) or None，1=有效, 0=无效
        """
        
        B, N, D = r.shape  # batch, n_trial, n_emb

        if mask is not None:
            mask = ~mask.bool()  # key_padding_mask: 1 表示要 **忽略** 的位置

        # 如果使用 CLS token，就把它拼接进 context set
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch, 1, n_emb)
            r = torch.cat([cls_tokens, r], dim=1)  # (batch, n_trial+1, n_emb)
            if mask is not None:
                mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=r.device), mask], dim=1)

        # Self-Attention + 残差连接
        r_agg, _ = self.attn(r, r, r, key_padding_mask=mask)
        r = self.norm1(r + r_agg)  # 残差连接 + 归一化

        # 前馈网络 + 残差连接
        r_ffn = self.ffn(r)
        r = self.norm2(r + r_ffn)

        # 如果用了 CLS token，返回它；否则平均池化
        return r[:, 0] if self.use_cls_token else r.mean(dim=1)


#%%
###################
# sequential
###################

class DeepLSTMEncoder(nn.Module):
    def __init__(self, dim_input, dim_output, num_layers=4, dropout=0.0):
        super(DeepLSTMEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=self.dim_input,
            hidden_size=self.dim_output,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)  # output: (batch, seq_len, dim_output)
        
        return hn[-1]

class LSTMAggregator(nn.Module):
    """
    处理每个 trial 的时间序列特征: (B, N, L, dim_data_time) -> (B, N, dim_emb_time)
    使用 LSTM (或多层 LSTM) 提取时序信息
    """
    def __init__(self, dim_data_time, dim_emb_time, lstm_hidden_size=64, num_layers=2):
        super().__init__()
        self.dim_data_time = dim_data_time
        self.dim_emb_time = dim_emb_time
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=dim_data_time,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_layers,
                            batch_first=True,  # (B, L, input_size)
                            bidirectional=False)
        # 将 LSTM 输出的 hidden state 投影到 dim_emb_time
        self.fc = nn.Linear(lstm_hidden_size, dim_emb_time)
        
    def forward(self, x):
        # x shape: (B, N, L, dim_data_time)
        B, N, L, _ = x.shape
        # 合并 B*N 维度
        x = x.view(B*N, L, -1)  # 变成 (B*N, L, dim_data_time)
        
        # LSTM
        outputs, (h_n, c_n) = self.lstm(x)  
        # outputs shape: (B*N, L, lstm_hidden_size)
        # h_n shape: (num_layers, B*N, lstm_hidden_size)
        
        # 取最后一个时间步的 hidden state (也可做 mean pooling 等其他策略)
        last_hidden = h_n[-1]  # 取最后一层的输出
        # 投影到 dim_emb_time
        emb_time = self.fc(last_hidden)  # (B*N, dim_emb_time)
        
        # reshape 回 (B, N, dim_emb_time)
        emb_time = emb_time.view(B, N, self.dim_emb_time)
        return emb_time