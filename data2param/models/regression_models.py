import torch.nn as nn

from .encoders import *
from .aggregators import *
from .decoders import *
from .diffusion_models import *
from .norm_flow import *
from .transformers import *

#%%  
class Reg_subjtrial(nn.Module):
    def __init__(self, 
                 dim_data_subj, dim_emb_subj,
                 dim_data_trial, dim_emb_trial,
                 dim_output,                 
                 agg_num_heads=8, agg_num_layers=4,
                 post_dropout=0.0):
        super().__init__()

        # Encoder        
        self.encoder_subj = MLPDecoder2(dim_data_subj, dim_emb_subj)

        self.encoder_trial = MLPDecoder2(dim_data_trial, dim_emb_trial)
        
        # Aggregator
        self.aggregator = NonPositionAttentionAggregator1(
            hidden_dim=dim_emb_trial,
            num_heads=agg_num_heads, num_layers=agg_num_layers
        )
        
        # Decoder
        self.decoder = MLPDecoder1(dim_emb_subj+dim_emb_trial, dim_output,
                                   dropout=post_dropout)
        
    def forward(self, X_subj, X, mask):
        x_subj_encoded = self.encoder_subj(X_subj)
        x_trial_encoded = self.encoder_trial(X)
        
        x_trial_agg = self.aggregator(x_trial_encoded, mask)
        
        concatenated = torch.cat([x_subj_encoded, x_trial_agg], dim=1)
        Yh = self.decoder(concatenated)
        
        return Yh  
    
class RegDiffusion_subjtrial(Reg_subjtrial):
    def __init__(self,
                 # 新增父类需要的参数 ↓↓↓
                 dim_data_subj, dim_emb_subj,
                 dim_data_trial, dim_emb_trial,
                 dim_output, 
                 agg_num_heads=8, agg_num_layers=4,
                 dropout=0.0,
                 # 新增父类需要的参数 ↑↑↑
                 T=100, ncon=128, beta_start=1e-4, beta_end=0.02, time_emb_dim=128):
        
        # 正确传递所有父类参数 ↓↓↓
        super().__init__(
            dim_data_subj=dim_data_subj,
            dim_emb_subj=dim_emb_subj,
            dim_data_trial=dim_data_trial,
            dim_emb_trial=dim_emb_trial,
            dim_output=dim_output,
            agg_num_heads=agg_num_heads,
            agg_num_layers=agg_num_layers,
            dropout=dropout
        )
        
        self.decoder = MLPDecoder1(dim_emb_subj+dim_emb_trial, ncon,
                                   dropout=dropout)
        
        self.dens_es = ConditionalDiffusion(output_dim=dim_output,
                                            T=T,
                                            ncon=ncon,
                                            time_emb_dim=time_emb_dim,
                                            beta_start=1e-4,
                                            beta_end=0.02)
        
    def forward(self, X_subj, X, mask, labels):
        x_subj_encoded = self.encoder_subj(X_subj)
        x_trial_encoded = self.encoder_trial(X)
        
        x_trial_agg = self.aggregator(x_trial_encoded, mask)
        
        concatenated = torch.cat([x_subj_encoded, x_trial_agg], dim=1)
        condition = self.decoder(concatenated)
        
        loss = self.dens_es(labels, condition)
        
        return loss
    
    def get_estimation_point(self, X_subj, X, mask, x_T):
        x_subj_encoded = self.encoder_subj(X_subj)
        x_trial_encoded = self.encoder_trial(X)
        
        x_trial_agg = self.aggregator(x_trial_encoded, mask)
        
        concatenated = torch.cat([x_subj_encoded, x_trial_agg], dim=1)
        condition = self.decoder(concatenated)
        
        return self.dens_es.inverse(x_T, condition)
     
class RegNormFlow_subjtrial(Reg_subjtrial):
    def __init__(self,
                 # 新增父类需要的参数 ↓↓↓
                 dim_data_subj, dim_emb_subj,
                 dim_data_trial, dim_emb_trial,
                 dim_output, 
                 agg_num_heads=8, agg_num_layers=4,
                 dropout=0.0,
                 # 新增父类需要的参数 ↑↑↑
                 ncon=128, num_coupling_layers=10):
        
        # 正确传递所有父类参数 ↓↓↓
        super().__init__(
            dim_data_subj=dim_data_subj,
            dim_emb_subj=dim_emb_subj,
            dim_data_trial=dim_data_trial,
            dim_emb_trial=dim_emb_trial,
            dim_output=dim_output,
            agg_num_heads=agg_num_heads,
            agg_num_layers=agg_num_layers,
            dropout=dropout
        )
        
        self.decoder = MLPDecoder1(dim_emb_subj+dim_emb_trial, ncon,
                                   dropout=dropout)
        
        self.dens_es = NormFlow(dim_out=dim_output, cond_dim=ncon, num_coupling_layers=num_coupling_layers)
        
    def forward(self, X_subj, X, mask, labels):
        x_subj_encoded = self.encoder_subj(X_subj)
        x_trial_encoded = self.encoder_trial(X)
        
        x_trial_agg = self.aggregator(x_trial_encoded, mask)
        
        concatenated = torch.cat([x_subj_encoded, x_trial_agg], dim=1)
        condition = self.decoder(concatenated)
        
        return self.dens_es(labels, condition)
    
    def get_estimation_point(self, X_subj, X, mask, z):
        x_subj_encoded = self.encoder_subj(X_subj)
        x_trial_encoded = self.encoder_trial(X)
        
        x_trial_agg = self.aggregator(x_trial_encoded, mask)
        
        concatenated = torch.cat([x_subj_encoded, x_trial_agg], dim=1)
        condition = self.decoder(concatenated)
        
        return self.dens_es.inverse(z, condition)
    
class Reg_flat(nn.Module):
    def __init__(self, dim_data, dim_output, dim_middle=256,
                 # encoder_num_blocks=4,
                 agg_hidden_dim=None, agg_num_heads=8, agg_num_layers=4,
                 decoder_num_blocks=4,
                 dropout=0.0, post_dropout=0):
        super().__init__()
        
        if agg_hidden_dim is None:
            agg_hidden_dim = dim_middle

        # Encoder
        # self.encoder = ResidualEncoder(dim_data, dim_middle,
        #                                num_blocks=encoder_num_blocks, dropout=dropout)
        
        self.encoder = MLPDecoder2(dim_input=dim_data, dim_output=dim_middle)
        
        # Aggregator
        self.aggregator = NonPositionAttentionAggregator1(
            hidden_dim=agg_hidden_dim, num_heads=agg_num_heads, num_layers=agg_num_layers
        )
        
        # Decoder
        self.decoder = MLPDecoder1(dim_middle, dim_output,
                                   post_dropout=post_dropout)
        
    def forward(self, X, mask):
        encoded = self.encoder(X)
        aggregated = self.aggregator(encoded, mask)
        Yh = self.decoder(aggregated)
        
        return Yh    

class Reg_trialtime(nn.Module):

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
                 num_layers_lstm=2,
                 agg_num_heads=8,
                 agg_num_layers=2
                 ):
        super().__init__()
        
        print("this model is RegTrialTime")
        
        self.dim_data_trial = dim_data_trial
        self.len_time = len_time
        self.dim_data_time = dim_data_time
        
        # 1. Encoder for trial
        self.encoder_trial = MLPDecoder2(dim_input=dim_data_trial, dim_output=dim_emb_trial)
        
        # 2. Aggregator for time (deep LSTM)
        self.aggregator_time = LSTMAggregator(
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
        self.aggregator_trial_time = NonPositionAttentionAggregator1(
            hidden_dim=dim_emb_trial_time, num_heads=agg_num_heads, num_layers=agg_num_layers
        )
                
        # 5. Decoder
        self.decoder = MLPDecoder1(dim_emb_trial_time, dim_output)
        
    def forward(self, x, mask):
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
        agg_trial_time = self.aggregator_trial_time(emb_trial_time, mask)
        
        # 5. Decoder 输出
        out = self.decoder(agg_trial_time)  # (B, dim_output)
        
        return out

class Classify_subjtrial(Reg_subjtrial):
    def __init__(self,
                 dim_data_subj, dim_emb_subj,
                 dim_data_trial, dim_emb_trial,
                 dim_output,                 
                 agg_num_heads=8, agg_num_layers=4,
                 dropout=0.0):
        super().__init__(dim_data_subj, dim_emb_subj, dim_data_trial, dim_emb_trial, dim_output, agg_num_heads, agg_num_layers, dropout)
        
        self.decoder = MLPDecoder1(dim_emb_subj+dim_emb_trial, dim_output,
                                   dropout=dropout)
        
    def forward(self, X_subj, X, mask):
        x_subj_encoded = self.encoder_subj(X_subj)
        x_trial_encoded = self.encoder_trial(X)
        
        x_trial_agg = self.aggregator(x_trial_encoded, mask)
        
        concatenated = torch.cat([x_subj_encoded, x_trial_agg], dim=1)
        Yh = self.decoder(concatenated)
        return Yh      


class Reg_flat_niid(nn.Module):
    def __init__(self, dim_data, dim_output, dim_middle=256,
                 # encoder_num_blocks=4,
                 agg_hidden_dim=None, agg_num_heads=8, agg_num_layers=4,
                 decoder_num_blocks=4,
                 dropout=0.0, post_dropout=0):
        super().__init__()
        
        if agg_hidden_dim is None:
            agg_hidden_dim = dim_middle

        # Encoder
        # self.encoder = ResidualEncoder(dim_data, dim_middle,
        #                                num_blocks=encoder_num_blocks, dropout=dropout)
        
        # self.encoder = MLPDecoder2(dim_input=dim_data, dim_output=dim_middle)
        self.encoder = ResidualEncoder(dim_data=dim_data, midFeature=dim_middle, num_blocks=4, dropout=0)
        
        # Aggregator
        self.aggregator = Transformer1(
            hidden_dim=agg_hidden_dim, num_heads=agg_num_heads, num_layers=agg_num_layers
        )
        
        # Decoder
        self.decoder = MLPDecoder1(dim_middle, dim_output,
                                   dropout=dropout, post_dropout=post_dropout)
        
    def forward(self, X, mask):
        encoded = self.encoder(X)
        aggregated = self.aggregator(encoded, mask)
        Yh = self.decoder(aggregated)
        
        return Yh    