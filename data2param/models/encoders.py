import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.block(x))

class ResidualEncoder(nn.Module):
    def __init__(self, dim_data, midFeature, num_blocks, dropout=0.0):
        super().__init__()

        self.input_proj = nn.Linear(dim_data, midFeature)
        
        self.encoder = nn.Sequential(*[
            ResidualBlock(midFeature, dropout) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return x

class Connect(nn.Module):
    """
    将 (B, N, dim_emb_trial) 与 (B, N, dim_emb_time) 等进行融合
    这里简单地采用 concat + MLP 的方式
    """
    def __init__(self, dim_in1, dim_in2, dim_out, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in1 + dim_in2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_out)
        )
        
    def forward(self, x1, x2):
        # x1, x2 形状: (B, N, D1) 和 (B, N, D2)
        x = torch.cat([x1, x2], dim=-1)  # 在最后一维拼接
        return self.mlp(x)