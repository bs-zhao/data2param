import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# 1. 多层感知器 (MLP) 工具类
# ----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        """
        input_dim:  输入特征维度
        hidden_dims: 隐藏层尺寸列表，如 [128, 64]
        output_dim:  输出特征维度
        dropout:     Dropout 概率
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hd
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x 的形状: (batch_size, ..., input_dim)
        # 因为 BatchNorm1d 要求通道维度在中间，可以将 x reshape 到 (batch_size * ..., input_dim)
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])  
        x = self.net(x)
        # 再 reshape 回去
        x = x.view(*orig_shape[:-1], -1)
        return x

# ----------------------------------------------------
# 2. 对 x_data 进行编码 (Encoder)
#    - 用 MLP 将 (batch, trial, dim_data) -> (batch, trial, dim_middle)
# ----------------------------------------------------
class EncoderData(nn.Module):
    def __init__(self, dim_data, hidden_dims, dim_middle, dropout=0.1):
        super().__init__()
        self.mlp = MLP(dim_data, hidden_dims, dim_middle, dropout)

    def forward(self, x):
        # x: (batch, trial, dim_data)
        return self.mlp(x)  # (batch, trial, dim_middle)

# ----------------------------------------------------
# 3. 序列编码器 (SeqTransformerEncoder)
#    - 对 (batch, trial, length_seq, dim_seq) 做时序处理
#    - 输出 (batch, trial, d_seqEnc)
# ----------------------------------------------------
class SeqTransformerEncoder(nn.Module):
    def __init__(self, dim_input, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(dim_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(d_model)

        # 这里简化做法：取所有 time step 的平均作为序列表示
        # 也可添加 [CLS] token 或者使用最后时刻输出等
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (batch, trial, length_seq, dim_seq)
        return: (batch, trial, d_model)
        """
        b, t, l, d = x.shape
        # 合并 batch 和 trial 维度
        x = x.view(b*t, l, d)   # (b*t, length_seq, dim_seq)
        x = self.linear_in(x)   # (b*t, length_seq, d_model)

        x = self.transformer(x) # (b*t, length_seq, d_model)
        x = self.ln_out(x)

        # 池化 (time dimension)
        # x.transpose(1,2): (b*t, d_model, length_seq)
        x_pooled = self.pool(x.transpose(1,2)).squeeze(-1)  # (b*t, d_model)

        # reshape 回 (batch, trial, d_model)
        x_pooled = x_pooled.view(b, t, -1)
        return x_pooled

# ----------------------------------------------------
# 4. MAB (Multihead Attention Block) 
#    - 用于后面的 Cross-Attention 机制
# ----------------------------------------------------
class MAB(nn.Module):
    """
    MAB(X, Y) = LN( X + MultiHeadAtt(X, Y) )
                + LN( X + FFN(...) )
    """
    def __init__(self, dim_Q, dim_K, num_heads, dim_ff=None, dropout=0.1):
        super().__init__()
        if dim_ff is None:
            dim_ff = dim_Q * 4
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_Q, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim_Q)

        self.ffn = nn.Sequential(
            nn.Linear(dim_Q, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_Q)
        )
        self.ln2 = nn.LayerNorm(dim_Q)

    def forward(self, Q, K):
        """
        Q: (batch, n_q, d_Q)
        K: (batch, n_k, d_K)  # 这里假设 d_Q == d_K
        V = K
        """
        out, _ = self.multihead_attn(Q, K, K)
        out = self.ln1(Q + out)      # 残差 + LN

        out2 = self.ffn(out)
        out = self.ln2(out + out2)   # 残差 + LN
        return out

# ----------------------------------------------------
# 5. CrossAttentionLayer
#    - 多步交互的一步：先让 x_data attend 到 x_seq，再让 x_seq attend 到 x_data
# ----------------------------------------------------
class CrossAttentionLayer(nn.Module):
    def __init__(self, dim_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.mab_data_to_seq = MAB(dim_model, dim_model, num_heads, dropout=dropout)
        self.mab_seq_to_data = MAB(dim_model, dim_model, num_heads, dropout=dropout)

    def forward(self, x_data, x_seq):
        """
        x_data: (b, t, d_model)
        x_seq:  (b, t, d_model)
        """
        # 1) x_data as Query, x_seq as Key/Value
        x_data_new = self.mab_data_to_seq(x_data, x_seq)

        # 2) x_seq as Query, x_data as Key/Value
        x_seq_new = self.mab_seq_to_data(x_seq, x_data)

        return x_data_new, x_seq_new

# ----------------------------------------------------
# 6. CrossAttentionEncoder
#    - 堆叠多层 CrossAttentionLayer，形成多步交互
# ----------------------------------------------------
class CrossAttentionEncoder(nn.Module):
    def __init__(self, dim_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x_data, x_seq):
        for layer in self.layers:
            x_data, x_seq = layer(x_data, x_seq)
        return x_data, x_seq

# ----------------------------------------------------
# 7. 多分支聚合 (MultiBranchTrialAggregator)
#    - 对 (batch, trial, dim_in) 在 trial 维度做多分支聚合:
#      1) 平均池化
#      2) 最大池化
#      3) 注意力池化 (使用一个可学习 query 向量)
#    - 将三者拼接后，通过小型 MLP 融合，得到 (batch, dim_in)
# ----------------------------------------------------
class MultiBranchTrialAggregator(nn.Module):
    """
    对 (batch, trial, dim_in) 在 trial 维度做多分支聚合:
      1) 平均池化 (只对 mask=1 的位置)
      2) 最大池化 (只对 mask=1 的位置)
      3) 注意力池化 (只对 mask=1 的位置)
    将三者拼接后，通过 MLP 融合，得到 (batch, dim_in)。
    """
    def __init__(self, dim_in, num_heads=4, dropout=0.1):
        super().__init__()
        # 用于注意力池化的可学习 query
        self.attn_query = nn.Parameter(torch.randn(1, 1, dim_in))

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_in,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 用 MLP 融合三种池化结果
        self.mlp_merge = nn.Sequential(
            nn.Linear(dim_in * 3, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in)
        )

    def forward(self, x, mask):
        """
        x:    (batch, trial, dim_in)
        mask: (batch, trial), 取值为1或0；0表示对应trial无效，需要忽略
        return: (batch, dim_in)
        """
        b, t, d = x.shape
        # 保证 mask 中 1 的个数至少为 1（以免除以 0）
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)

        # ------------- (1) 平均池化 (mask=1 的位置) -------------
        # 先将无效 trial 的特征置为 0，然后在 batch 维度上做 sum
        # 再除以有效的 trial 数量
        x_sum = (x * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (b, 1, d)
        mean_agg = x_sum / valid_count.unsqueeze(-1)               # (b, 1, d)

        # ------------- (2) 最大池化 (mask=1 的位置) -------------
        # 对无效 trial 赋予一个很小的值，以便在 max 池化时被忽略
        x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        max_agg, _ = x_masked.max(dim=1, keepdim=True)  # (b, 1, d)
        # 如果该 batch 全部 trial 都无效，会得到 -inf，这里简单处理成 0
        max_agg = torch.where(torch.isinf(max_agg), torch.zeros_like(max_agg), max_agg)

        # ------------- (3) 注意力池化 (mask=1 的位置) -------------
        # PyTorch MultiheadAttention 可以用 key_padding_mask 来忽略某些位置
        # key_padding_mask: shape (batch, src_len)，True 表示要忽略
        query = self.attn_query.repeat(b, 1, 1)  # (b, 1, d)
        key_padding_mask = (mask == 0)           # (b, t), True 表示忽略
        attn_out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        # (b, 1, d)

        # 拼接三种结果: (b, 1, 3*d)
        cat = torch.cat([mean_agg, max_agg, attn_out], dim=-1)

        # MLP 融合
        merged = self.mlp_merge(cat)  # (b, 1, d)
        return merged.squeeze(1)      # (b, d)

# ----------------------------------------------------
# 8. 解码器 (Decoder)
# ----------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, dim_input, hidden_dims, dim_output, dropout=0.1):
        super().__init__()
        self.mlp = MLP(dim_input, hidden_dims, dim_output, dropout)

    def forward(self, x):
        return self.mlp(x)  # (batch, dim_output)

# ----------------------------------------------------
# 9. 最终整合模型 (ComplexCrossAttentionModel)
# ----------------------------------------------------    
class ComplexCrossAttentionModel(nn.Module):
    def __init__(
        self,
        dim_data,         # x_data 特征维度
        length_seq,
        dim_seq,          # x_seq 每个时间步的特征维度
        dim_output=10,
        dim_dataEnc=64,   # 编码 x_data 后的维度
        d_model=128,      # 序列编码 + Cross-Attention 的统一维度
        num_ca_layers=2,  # Cross-Attention 层数
    ):
        super().__init__()
        
        self.dim_data = dim_data
        self.length_seq = length_seq
        self.dim_seq = dim_seq
        self.dim_output = dim_output        

        # 1) 对 x_data 做 MLP 编码 -> (b, t, dim_dataEnc)
        self.encoder_data = EncoderData(
            dim_data=dim_data,
            hidden_dims=[128, 64],
            dim_middle=dim_dataEnc,
            dropout=0.1
        )
        # 如果 dim_dataEnc != d_model，需要投影到同一维度
        self.proj_data = nn.Linear(dim_dataEnc, d_model)

        # 2) 对 x_seq 做时序编码 -> (b, t, d_model)
        self.encoder_seq = SeqTransformerEncoder(
            dim_input=dim_seq,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1
        )

        # 3) Cross-Attention 编码，多步交互
        self.cross_att_encoder = CrossAttentionEncoder(
            dim_model=d_model,
            num_heads=4,
            num_layers=num_ca_layers,
            dropout=0.1
        )

        # 4) 聚合 trial 维度 (改动：使用支持 mask 的 aggregator)
        self.aggregator = MultiBranchTrialAggregator(
            dim_in=d_model * 2,  # 这里我们打算把 x_data_ca 和 x_seq_ca 最终拼接
            num_heads=4,
            dropout=0.1
        )

        # 5) 解码器
        self.decoder = Decoder(
            dim_input=d_model * 2,
            hidden_dims=[128, 64],
            dim_output=dim_output,
            dropout=0.1
        )

    def forward(self, x, mask):
        """
        x_data: (batch, trial, dim_data)
        x_seq:  (batch, trial, length_seq, dim_seq)
        mask:   (batch, trial)  # 1 表示有效, 0 表示无效
        return: (batch, dim_output)
        """
        b, t, _ = x.shape        
        
        x_data = x[:, :, :self.dim_data]
        x_seq = x[:, :, self.dim_data:].reshape(b, t, self.length_seq, self.dim_seq)        

        # 1) 编码 x_data
        x_data_enc = self.encoder_data(x_data)  # (b, t, dim_dataEnc)
        x_data_enc = self.proj_data(x_data_enc) # (b, t, d_model)

        # 2) 编码 x_seq
        x_seq_enc = self.encoder_seq(x_seq)     # (b, t, d_model)

        # 3) Cross-Attention，多层往返交互
        x_data_ca, x_seq_ca = self.cross_att_encoder(x_data_enc, x_seq_enc)
        # x_data_ca, x_seq_ca 均是 (b, t, d_model)

        # 4) 将二者拼接 -> (b, t, 2*d_model)
        x_cat = torch.cat([x_data_ca, x_seq_ca], dim=-1)

        # 5) 对 trial 维度做多分支聚合 (传入 mask)
        x_agg = self.aggregator(x_cat, mask)  # (b, 2*d_model)

        # 6) 解码
        out = self.decoder(x_agg)             # (b, dim_output)
        return out
