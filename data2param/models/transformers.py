import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码层，为序列中的每个位置添加唯一的编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区而不是参数，这样在训练时不会更新
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
        返回:
            添加了位置编码的张量 [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制模块"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 定义线性变换层
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        Q = self.q_linear(query)  # [batch_size, seq_len, hidden_dim]
        K = self.k_linear(key)    # [batch_size, seq_len, hidden_dim]
        V = self.v_linear(value)  # [batch_size, seq_len, hidden_dim]
        
        # 将张量重塑为[batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力得分
        # Q: [batch_size, num_heads, query_len, head_dim]
        # K: [batch_size, num_heads, key_len, head_dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码到所有头
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # 注意力权重和dropout
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 聚合value向量
        x = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 合并多头注意力的结果
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, -1, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]
        
        # 最终的线性层
        x = self.fc_out(x)
        
        return x

class FeedForwardNetwork(nn.Module):
    """Position-wise前馈网络"""
    
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_src, mask=None, enc_mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 编码器-解码器注意力子层
        attn_output = self.enc_attn(x, enc_src, enc_src, enc_mask)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x

class PositionalTransformer(nn.Module):
    """包含位置关系的Transformer模型"""
    
    def __init__(self, hidden_dim, num_heads, num_layers, ff_dim=None, dropout=0.1, max_len=5000, use_positional=True):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = hidden_dim * 4
            
        self.use_positional = use_positional
        
        # 位置编码
        if use_positional:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入序列 [batch_size, seq_len, hidden_dim]
            mask: 掩码张量 [batch_size, seq_len]
        返回:
            输出序列 [batch_size, seq_len, hidden_dim]
        """
        # 应用位置编码（如果启用）
        if self.use_positional:
            x = self.pos_encoder(x)
        
        x = self.dropout(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        return x

class Transformer1(nn.Module):
    """考虑trial之间顺序关系的transformer聚合器"""
    
    def __init__(self, hidden_dim, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.transformer = PositionalTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_positional=True  # 启用位置编码，显式考虑位置关系
        )
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            mask: 掩码张量 [batch_size, seq_len]
        返回:
            聚合后的表示 [batch_size, hidden_dim]
        """
        # 应用transformer编码
        transformer_output = self.transformer(x, mask)  # [batch_size, seq_len, hidden_dim]
        
        # 如果提供了掩码，则使用掩码进行平均池化
        if mask is not None:
            # 扩展mask以匹配隐藏维度
            extended_mask = mask.unsqueeze(-1).expand_as(transformer_output)
            # 应用掩码并计算平均值
            sum_embeddings = torch.sum(transformer_output * extended_mask, dim=1)
            sum_mask = torch.sum(extended_mask, dim=1)
            # 避免除以零
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            # 计算平均值
            pooled_output = sum_embeddings / sum_mask
        else:
            # 如果没有掩码，简单地对所有位置求平均
            pooled_output = torch.mean(transformer_output, dim=1)
        
        return pooled_output
