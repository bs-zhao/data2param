import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderSubj(nn.Module):
    """
    对 subject 特征进行编码: (B, dim_data_subj) -> (B, dim_emb_subj)
    """
    def __init__(self, dim_data_subj, dim_emb_subj, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_data_subj, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_emb_subj)
        )
        
    def forward(self, x):
        return self.net(x)

class EncoderTrial(nn.Module):
    """
    对 trial 特征进行编码: (B, N, dim_data_trial) -> (B, N, dim_emb_trial)
    """
    def __init__(self, dim_data_trial, dim_emb_trial, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_data_trial, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_emb_trial)
        )
        
    def forward(self, x):
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        x = self.mlp(x)
        x = x.view(B, N, -1)
        return x

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
        x:    (B, N, dim_in)
        mask: (B, N), 取值为1或0；0表示对应 trial 无效，需要忽略
        return: (B, dim_in)
        """
        b, t, d = x.shape
        # 保证 mask 中 1 的个数至少为 1（以免除以 0）
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)

        # ------------- (1) 平均池化 (mask=1 的位置) -------------
        # 先将无效 trial 的特征置为 0，然后在 trial 维度上求和，再除以有效 trial 数量
        x_sum = (x * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (b, 1, d)
        mean_agg = x_sum / valid_count.unsqueeze(-1)               # (b, 1, d)

        # ------------- (2) 最大池化 (mask=1 的位置) -------------
        # 对无效 trial 赋予一个很小的值，以便在 max 池化时被忽略
        x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        max_agg, _ = x_masked.max(dim=1, keepdim=True)  # (b, 1, d)
        # 如果该 batch 全部 trial 都无效，会得到 -inf，这里简单处理成 0
        max_agg = torch.where(torch.isinf(max_agg), torch.zeros_like(max_agg), max_agg)

        # ------------- (3) 注意力池化 (mask=1 的位置) -------------
        # 使用 key_padding_mask 忽略无效位置
        query = self.attn_query.repeat(b, 1, 1)  # (b, 1, d)
        key_padding_mask = (mask == 0)           # (b, N)，True 表示忽略
        attn_out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        # (b, 1, d)

        # 拼接三种结果: (b, 1, 3*d)
        cat = torch.cat([mean_agg, max_agg, attn_out], dim=-1)
        merged = self.mlp_merge(cat)  # (b, 1, d)
        return merged.squeeze(1)      # (b, d)

class Connect(nn.Module):
    """
    将 subject embedding 与 trial 聚合 embedding 融合:
    (B, dim_emb_subj) 和 (B, dim_emb_agg) -> (B, dim_fused)
    这里采用 concat 后 MLP 进行融合
    """
    def __init__(self, dim_in1, dim_in2, dim_out, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in1 + dim_in2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_out)
        )
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        return self.mlp(x)

class Decoder(nn.Module):
    """
    将融合后的向量映射到最终输出: (B, dim_in) -> (B, dim_output)
    """
    def __init__(self, dim_in, dim_output, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_output)
        )
        
    def forward(self, x):
        return self.net(x)

class RegSubjTrial(nn.Module):
    """
    完整网络：
      1. 对 subject 进行编码
      2. 对 trial 进行编码
      3. 用 MultiBranchTrialAggregator 对所有 trial 进行多分支聚合（mask=0的 trial 会被忽略）
      4. 将 subject embedding 与聚合后的 trial embedding 融合
      5. 经过 Decoder 得到最终输出
    """
    def __init__(self, dim_data_subj, dim_emb_subj,
                 dim_data_trial, dim_emb_trial,
                 aggregator_dim,  # 多分支聚合器的输入与输出维度
                 dim_output,
                 hidden_size_connect=64, hidden_size_decoder=64):
        super().__init__()
        self.encoder_subj = EncoderSubj(dim_data_subj, dim_emb_subj)
        self.encoder_trial = EncoderTrial(dim_data_trial, dim_emb_trial)
        # 这里我们要求 encoder_trial 的输出维度与 aggregator 的 dim_in 一致
        # 如果 dim_emb_trial 与 aggregator_dim 不一致，可在此添加映射层
        self.aggregator_trial = MultiBranchTrialAggregator(dim_in=aggregator_dim)
        self.connect = Connect(dim_emb_subj, aggregator_dim, aggregator_dim, hidden_size=hidden_size_connect)
        self.decoder = Decoder(aggregator_dim, dim_output, hidden_size=hidden_size_decoder)
        
    def forward(self, x_subj, x_trial, mask):
        """
        x_subj:   (B, dim_data_subj)
        x_trial:  (B, N, dim_data_trial)
        mask:     (B, N)，取值为1或0，0表示对应 trial 无效
        """
        emb_subj = self.encoder_subj(x_subj)        # (B, dim_emb_subj)
        emb_trial = self.encoder_trial(x_trial)       # (B, N, dim_emb_trial)
        # 如果 encoder_trial 输出的维度与 aggregator 所要求的不一致，可在此添加映射
        agg_trial = self.aggregator_trial(emb_trial, mask)  # (B, aggregator_dim)
        fused = self.connect(emb_subj, agg_trial)       # (B, aggregator_dim)
        out = self.decoder(fused)                       # (B, dim_output)
        return out

if __name__ == "__main__":
    # 假设超参数如下
    B = 2                   # batch size
    N = 5                   # trial 数
    dim_data_subj = 8
    dim_data_trial = 16
    dim_emb_subj = 12
    dim_emb_trial = 16     # encoder_trial 的输出维度
    aggregator_dim = 16     # 这里要求 aggregator 的输入维度与 encoder_trial 输出一致
    dim_output = 3

    model = RegSubjTrial(
        dim_data_subj=dim_data_subj,
        dim_emb_subj=dim_emb_subj,
        dim_data_trial=dim_data_trial,
        dim_emb_trial=dim_emb_trial,
        aggregator_dim=aggregator_dim,
        dim_output=dim_output
    )

    x_subj = torch.randn(B, dim_data_subj)
    x_trial = torch.randn(B, N, dim_data_trial)
    mask = torch.randint(0, 2, (B, N)).float()  # 随机生成 0 或 1 的 mask

    out = model(x_subj, x_trial, mask)
    print("Output shape:", out.shape)  # 期望输出形状: (B, dim_output)
