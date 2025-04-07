import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 时间步嵌入函数（Sinusoidal embedding）
def get_timestep_embedding(timesteps, embedding_dim):
    # timesteps: tensor, shape (batch_size,)
    half_dim = embedding_dim // 2
    emb_factor = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float) * -emb_factor)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # shape: (batch_size, embedding_dim)

class ConditionalDiffusion(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, ncon=16, T=100, beta_start=1e-4, beta_end=0.02, time_emb_dim=16):
        super().__init__()
        self.T = T
        self.time_emb_dim = time_emb_dim
        self.output_dim = output_dim
        
        # 定义线性beta调度
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, T))
        alphas = 1 - self.betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        
        # 噪声预测网络
        # 输入：x_t (output_dim 维) + condition (4维) + 时间嵌入 (time_emb_dim)
        input_dim = output_dim + ncon + time_emb_dim
        hidden_dim = 128
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 预测加入的噪声，输出 output_dim 维
        )
    
    def forward(self, x0, condition):
        """
        x0: [batch, output_dim] 真实目标参数
        inputs, masks: 用于从集合中提取条件信息
        """
        batch_size = x0.shape[0]
        
        # 采样时间步 t
        t = torch.randint(0, self.T, (batch_size,), device=x0.device)
        alpha_bar = self.alpha_bars[t].unsqueeze(1)  # [batch, 1]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        # 采样噪声
        noise = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        
        # 获取时间步嵌入
        t_emb = get_timestep_embedding(t, self.time_emb_dim)  # [batch, time_emb_dim]
        
        # 拼接信息输入噪声预测网络
        predictor_input = torch.cat([x_t, condition, t_emb], dim=1)
        noise_pred = self.noise_predictor(predictor_input)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def inverse(self, x_T, condition, deterministic=True):
        """
        逆扩散采样：从 x_T 逐步还原 x0
        deterministic=True 时关闭随机噪声，用于 MAP 估计
        """
        batch_size = x_T.shape[0]
        x_t = x_T

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, device=x_T.device, dtype=torch.long)
            t_emb = get_timestep_embedding(t_tensor, self.time_emb_dim)

            # 预测噪声
            predictor_input = torch.cat([x_t, condition, t_emb], dim=1)
            noise_pred = self.noise_predictor(predictor_input)

            # 计算去噪后的 x_{t-1}
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            coef = beta_t / (torch.sqrt(1 - alpha_bar_t) + 1e-8)
            x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - coef * noise_pred)

            # 仅在非确定性模式下添加噪声
            if not deterministic and t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_prev = x_prev + sigma_t * noise
            x_t = x_prev

        return x_t