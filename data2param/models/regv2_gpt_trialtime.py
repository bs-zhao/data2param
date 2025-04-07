import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderTrial(nn.Module):
    """
    对每个 trial 的特征进行编码: (B, N, dim_data_trial) -> (B, N, dim_emb_trial)
    可以对每个 trial 的数据共用同一个 MLP, 对 N 个 trial 并行处理
    """
    def __init__(self, dim_data_trial, dim_emb_trial, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_data_trial, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_emb_trial)
        )
        
    def forward(self, x):
        # x shape: (B, N, dim_data_trial)
        B, N, _ = x.shape
        # 合并 B*N, 进行 MLP 编码
        x = x.view(B*N, -1)
        x = self.mlp(x)
        # 再 reshape 回 (B, N, dim_emb_trial)
        x = x.view(B, N, -1)
        return x

class AggregatorTime(nn.Module):
    """
    处理每个 trial 的时间序列特征: (B, N, L, dim_data_time) -> (B, N, dim_emb_time)
    使用 LSTM (或多层 LSTM) 提取时序信息
    """
    def __init__(self, dim_data_time, dim_emb_time, lstm_hidden_size=64, num_layers=1):
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

class AggregatorTrialTime(nn.Module):
    """
    对所有 trial 的表示做 permutation-invariant 的聚合:
    (B, N, dim_emb_trial_time) -> (B, dim_emb_trial_time)
    这里演示一种简单的 DeepSets 思路:
      1) 先对 trial 维度做求和或平均: sum/mean pooling
      2) 再用一个 MLP 进行进一步提取
    """
    def __init__(self, dim_emb_in, dim_emb_out, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_emb_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_emb_out)
        )
        
    def forward(self, x):
        # x shape: (B, N, dim_emb_trial_time)
        # 先做 mean pooling (也可做 sum pooling 或更复杂的 attention)
        x_mean = x.mean(dim=1)  # (B, dim_emb_trial_time)
        # 再经过 MLP
        out = self.net(x_mean)  # (B, dim_emb_out)
        return out

class Decoder(nn.Module):
    """
    将聚合后的表示映射到最终输出: (B, dim_in) -> (B, dim_output)
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

class RegTrialTime(nn.Module):
    """
    组合以上所有模块，构建不包含 subject-level 输入的网络
    """
    def __init__(self,
                 # Trial encoder
                 dim_data_trial, dim_emb_trial,
                 # Time aggregator
                 len_time, dim_data_time, dim_emb_time,
                 # Trial-time aggregator
                 dim_emb_trial_time,
                 # Output
                 dim_output,
                 hidden_size_connect=64,
                 hidden_size_decoder=64,
                 lstm_hidden_size=64,
                 num_layers_lstm=1):
        super().__init__()
        
        print("this model is RegTrialTime")
        
        self.dim_data_trial = dim_data_trial
        self.len_time = len_time
        self.dim_data_time = dim_data_time
        
        # 1. Encoder for trial
        self.encoder_trial = EncoderTrial(dim_data_trial, dim_emb_trial)
        
        # 2. Aggregator for time (deep LSTM)
        self.aggregator_time = AggregatorTime(
            dim_data_time=dim_data_time,
            dim_emb_time=dim_emb_time,
            lstm_hidden_size=lstm_hidden_size,
            num_layers=num_layers_lstm
        )
        
        # 3. Connect to merge trial-emb and time-emb
        self.connect_trial_time = Connect(
            dim_in1=dim_emb_trial,
            dim_in2=dim_emb_time,
            dim_out=dim_emb_trial_time,
            hidden_size=hidden_size_connect
        )
        
        # 4. Aggregator for trial_time (permutation-invariant)
        self.aggregator_trial_time = AggregatorTrialTime(
            dim_emb_in=dim_emb_trial_time,
            dim_emb_out=dim_emb_trial_time,  # 这里可保持相同维度，也可自定义
            hidden_size=hidden_size_connect
        )
        
        # 5. Decoder
        self.decoder = Decoder(
            dim_in=dim_emb_trial_time,
            dim_output=dim_output,
            hidden_size=hidden_size_decoder
        )
        
    def forward(self, x):
        """
        Args:
            trial_x:  (B, N, dim_data_trial)
            time_x:   (B, N, L, dim_data_time)
        Returns:
            out:      (B, dim_output)
        """
        B, N, _ = x.shape
        
        trial_x = x[:, :, :self.dim_data_trial]
        time_x = x[:, :, self.dim_data_trial:].reshape(B, N, self.len_time, self.dim_data_time)   

        # 1. 编码 trial 特征
        emb_trial = self.encoder_trial(trial_x)  # (B, N, dim_emb_trial)
        
        # 2. 时序聚合 -> (B, N, dim_emb_time)
        emb_time = self.aggregator_time(time_x)
        
        # 3. 连接 trial-level embedding 和 time-level embedding
        emb_trial_time = self.connect_trial_time(emb_trial, emb_time)  # (B, N, dim_emb_trial_time)
        
        # 4. 对 N 个 trial 做无序聚合 -> (B, dim_emb_trial_time)
        agg_trial_time = self.aggregator_trial_time(emb_trial_time)
        
        # 5. Decoder 输出
        out = self.decoder(agg_trial_time)  # (B, dim_output)
        
        return out


if __name__ == "__main__":
    # 假设我们有以下超参数
    B = 32      # batch size
    N = 99          # trial 数
    L = 400          # 每个 trial 的时序长度
    dim_data_trial = 16
    dim_data_time = 4
    dim_emb_trial = 16
    dim_emb_time = 20
    dim_emb_trial_time = 24
    dim_output = 3
    
    model = RegTrialTime(
        dim_data_trial=dim_data_trial,
        len_time=L,
        dim_data_time=dim_data_time,
        dim_emb_trial=dim_emb_trial,        
        dim_emb_time=dim_emb_time,
        dim_emb_trial_time=dim_emb_trial_time,
        dim_output=dim_output
    )
    
    # 构造一些随机输入
    x = torch.randn(B, N, dim_data_trial+L*dim_data_time)
    # trial_x = torch.randn(B, N, dim_data_trial)
    # time_x = torch.randn(B, N, L, dim_data_time)
    
    # 前向传播
    out = model(x)
    print("Output shape:", out.shape)  # 期望 (B, dim_output)
