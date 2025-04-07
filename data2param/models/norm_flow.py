import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 3. 归一化流的各组件

# (1) DenseCouplingNet：构造全连接网络
class DenseCouplingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=128, num_dense=2, dropout_prob=0.01):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_dense):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# (2) AffineCoupling 模块：采用两段式耦合结构及软截断机制
class AffineCoupling(nn.Module):
    def __init__(self, dim_out, cond_dim, settings_dict):
        super().__init__()
        self.dim_out = dim_out
        self.soft_clamp = settings_dict.get("soft_clamping", None)
        
        # 按照维度对输入进行分割
        self.d1 = dim_out // 2
        self.d2 = dim_out - self.d1
        
        # 第一个耦合子块：从 u2 与 condition 预测用于变换 u1 的尺度和平移参数
        input_dim_net1 = self.d2 + cond_dim
        self.scale_net1 = DenseCouplingNet(input_dim_net1, self.d1, 
                                           hidden_units=settings_dict.get("units", 128),
                                           num_dense=settings_dict.get("num_dense", 2),
                                           dropout_prob=settings_dict.get("dropout_prob", 0.01))
        self.translate_net1 = DenseCouplingNet(input_dim_net1, self.d1,
                                               hidden_units=settings_dict.get("units", 128),
                                               num_dense=settings_dict.get("num_dense", 2),
                                               dropout_prob=settings_dict.get("dropout_prob", 0.01))
        # 第二个耦合子块：从 v1 与 condition 预测用于变换 u2 的尺度和平移参数
        input_dim_net2 = self.d1 + cond_dim
        self.scale_net2 = DenseCouplingNet(input_dim_net2, self.d2,
                                           hidden_units=settings_dict.get("units", 128),
                                           num_dense=settings_dict.get("num_dense", 2),
                                           dropout_prob=settings_dict.get("dropout_prob", 0.01))
        self.translate_net2 = DenseCouplingNet(input_dim_net2, self.d2,
                                               hidden_units=settings_dict.get("units", 128),
                                               num_dense=settings_dict.get("num_dense", 2),
                                               dropout_prob=settings_dict.get("dropout_prob", 0.01))
    
    def _soft_clamp(self, s):
        # 软截断函数： (2*soft_clamp/π)*atan(s/soft_clamp)
        return (2.0 * self.soft_clamp / np.pi) * torch.atan(s / self.soft_clamp)
    
    def forward(self, u1, u2, condition, inverse=False):
        if not inverse:
            return self._forward(u1, u2, condition)
        else:
            return self._inverse(u1, u2, condition)
    
    def _forward(self, u1, u2, condition):
        # 第一部分：变换 u1
        input_net1 = torch.cat([u2, condition], dim=1)
        s1 = self.scale_net1(input_net1)
        if self.soft_clamp is not None:
            s1 = self._soft_clamp(s1)
        t1 = self.translate_net1(input_net1)
        v1 = u1 * torch.exp(s1) + t1
        log_det_J1 = s1.sum(dim=1)
        
        # 第二部分：变换 u2
        input_net2 = torch.cat([v1, condition], dim=1)
        s2 = self.scale_net2(input_net2)
        if self.soft_clamp is not None:
            s2 = self._soft_clamp(s2)
        t2 = self.translate_net2(input_net2)
        v2 = u2 * torch.exp(s2) + t2
        log_det_J2 = s2.sum(dim=1)
        
        v = torch.cat([v1, v2], dim=1)
        log_det_J = log_det_J1 + log_det_J2
        return v, log_det_J
    
    def _inverse(self, v1, v2, condition):
        # 第二部分逆变换：恢复 u2
        input_net2 = torch.cat([v1, condition], dim=1)
        s2 = self.scale_net2(input_net2)
        if self.soft_clamp is not None:
            s2 = self._soft_clamp(s2)
        t2 = self.translate_net2(input_net2)
        u2 = (v2 - t2) * torch.exp(-s2)
        
        # 第一部分逆变换：恢复 u1
        input_net1 = torch.cat([u2, condition], dim=1)
        s1 = self.scale_net1(input_net1)
        if self.soft_clamp is not None:
            s1 = self._soft_clamp(s1)
        t1 = self.translate_net1(input_net1)
        u1 = (v1 - t1) * torch.exp(-s1)
        
        u = torch.cat([u1, u2], dim=1)
        return u

# (3) Permutation 层：固定排列
class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('perm', torch.randperm(dim))
        self.register_buffer('inv_perm', torch.argsort(self.perm))
        
    def forward(self, x, inverse=False):
        if not inverse:
            return x[:, self.perm]
        else:
            return x[:, self.inv_perm]

# (4) ActNorm 层：激活归一化（类似 Glow 中的 ActNorm）
class ActNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.logs = nn.Parameter(torch.zeros(1, num_features))
        self.initialized = False
        self.eps = eps
        
    def initialize_parameters(self, x):
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + self.eps
            self.bias.data.copy_(-mean)
            self.logs.data.copy_(torch.log(1.0 / std))
            self.initialized = True
        
    def forward(self, x, inverse=False):
        if not self.initialized:
            self.initialize_parameters(x)
        if not inverse:
            x = (x + self.bias) * torch.exp(self.logs)
            # 每个样本的对数行列式：所有特征上的 logs 之和
            log_det = self.logs.sum() * torch.ones(x.shape[0], device=x.device)
            return x, log_det
        else:
            x = x * torch.exp(-self.logs) - self.bias
            return x

# (5) CouplingLayerWrapper：包装 AffineCoupling，同时加上 Permutation 与 ActNorm
class CouplingLayerWrapper(nn.Module):
    def __init__(self, latent_dim, cond_dim, coupling_settings, permutation="fixed", use_act_norm=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.d1 = latent_dim // 2
        self.d2 = latent_dim - self.d1
        
        self.coupling = AffineCoupling(dim_out=latent_dim, cond_dim=cond_dim, settings_dict=coupling_settings)
        
        if permutation in ["fixed", "learnable"]:
            self.permutation = Permutation(latent_dim)
        else:
            self.permutation = None
        
        self.use_act_norm = use_act_norm
        if use_act_norm:
            self.act_norm = ActNorm(latent_dim)
        else:
            self.act_norm = None
        
    def forward(self, x, condition, inverse=False):
        log_det_J_total = 0
        if not inverse:
            if self.act_norm is not None:
                x, log_det_act = self.act_norm(x, inverse=False)
                log_det_J_total = log_det_J_total + log_det_act
            if self.permutation is not None:
                x = self.permutation(x, inverse=False)
            # 将输入按维度分为两部分
            x1, x2 = torch.split(x, [self.d1, self.d2], dim=1)
            y, log_det_c = self.coupling(x1, x2, condition, inverse=False)
            log_det_J_total = log_det_J_total + log_det_c
            return y, log_det_J_total
        else:
            # 逆向时：先将输入分割，然后依次逆向恢复，再逆向排列与 ActNorm
            x1, x2 = torch.split(x, [self.d1, self.d2], dim=1)
            u = self.coupling(x1, x2, condition, inverse=True)
            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
            if self.act_norm is not None:
                u = self.act_norm(u, inverse=True)
            return u

# (6) InvertibleNetwork：堆叠多个 CouplingLayerWrapper，再加上条件提取器 SetTransformer
class NormFlow(nn.Module):
    def __init__(self, dim_out=6, cond_dim=64, num_coupling_layers=10, coupling_settings=None):
        super().__init__()

        if coupling_settings is None:
            coupling_settings = {
                "units": 128,
                "num_dense": 2,
                "dropout_prob": 0.01,
                "soft_clamping": 1.9
            }
        self.layers = nn.ModuleList([
            CouplingLayerWrapper(latent_dim=dim_out, cond_dim=cond_dim, coupling_settings=coupling_settings)
            for _ in range(num_coupling_layers)
        ])
    
    def forward(self, targets, condition):
        # 通过 SetTransformer 从输入集合中提取条件向量
        log_det = 0
        z = targets
        for layer in self.layers:
            z, ld = layer(z, condition, inverse=False)
            log_det += ld
        return z, log_det
    
    @torch.no_grad()
    def inverse(self, z, condition):
        x = z
        for layer in reversed(self.layers):
            x = layer(x, condition, inverse=True)
        return x

# 4. 损失函数：负对数似然
def compute_loss(z, log_det):
    d = z.size(1)
    prior_logprob = -0.5 * (z**2).sum(dim=1) - 0.5 * d * np.log(2*np.pi)
    loss = -(prior_logprob + log_det).mean()
    return loss

