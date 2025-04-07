import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPDecoder1(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0):
        super().__init__()

        self.max = max(dim_input*2, 256)
        self.decoder = nn.Sequential(
            nn.Linear(dim_input, dim_input * 2),  
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Dropout(dropout),
            nn.Linear(dim_input * 2, self.max),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Dropout(dropout),
            nn.Linear(self.max, self.max),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Dropout(dropout),
            nn.Linear(self.max, self.max//2),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Dropout(dropout),
            nn.Linear(self.max//2, dim_output)
        )
    
    def forward(self, x):
        return self.decoder(x)
    
class MLPDecoder1_plus(nn.Module):
    def __init__(self, dim_input, dim_output, post_dropout=0):
        super().__init__()
        
        # 计算中间层维度
        self.max = max(dim_input*2, 256)
        self.mid = self.max // 2
        
        # 定义基础块
        def make_block(in_dim, out_dim, use_residual=True):
            layers = [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(post_dropout)
            ]
            if use_residual and in_dim == out_dim:
                return ResidualBlock(nn.Sequential(*layers))
            return nn.Sequential(*layers)
        
        # 构建网络
        self.decoder = nn.Sequential(
            # 输入层
            make_block(dim_input, dim_input * 2, use_residual=False),
            
            # 中间层
            make_block(dim_input * 2, self.max),
            make_block(self.max, self.max),
            make_block(self.max, self.max),
            
            # 降维层
            make_block(self.max, self.mid),
            make_block(self.mid, self.mid),
            
            # 输出层
            nn.Linear(self.mid, dim_output)
        )
    
    def forward(self, x):
        return self.decoder(x)

class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        
    def forward(self, x):
        return x + self.layers(x)


class MLPDecoder2(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_input, dim_input * 2),  
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Linear(dim_input * 2, dim_output * 2),
            nn.LeakyReLU(negative_slope=0.01),      
            nn.Linear(dim_output * 2, dim_output)
        )
    
    def forward(self, x):
        return self.model(x)


class MLPClassifier1(nn.Module):
    def __init__(self, dim_input, num_classes, dropout=0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim_input, dim_input * 2),  
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Linear(dim_input * 2, 256),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),       
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        logits = self.classifier(x)
        return logits  # 返回原始logits，通常与nn.CrossEntropyLoss一起使用
        
    def predict_proba(self, x):
        logits = self.classifier(x)
        return F.softmax(logits, dim=1)  # 返回概率
        
    def predict(self, x):
        logits = self.classifier(x)
        return torch.argmax(logits, dim=1)  # 返回预测的类别索引
