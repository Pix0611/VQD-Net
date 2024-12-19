import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
    
    def forward(self, pred, confidence, tgt):
        # pred, confidence, tgt: B*1
        B = pred.shape[0]
        return torch.sum((pred-tgt)**2) / B


class MainModel(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, nhead=4, num_layers=4):
        """
        Args:
            embedding_dim: 输入序列的特征维度
            seq_len: 序列长度
            hidden_dim: 全连接层隐藏层维度
            nhead: Transformer Encoder中多头注意力头数
            num_layers: Transformer Encoder的层数
        """
        super(MainModel, self).__init__()
        
        # Transformer Encoder 部分
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,  # 输入特征维度
            nhead=nhead,            # 多头注意力头数
            dim_feedforward=hidden_dim,  # 前馈网络的维度
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers  # Transformer Encoder的层数
        )
        
        # 全连接部分
        self.score_regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.confidence_regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.loss_func = ModelLoss()

    def forward(self, x, tgt=None):
        """
        Args:
            x: 输入数据，形状为 [batch_size, seq_len, embedding_dim]
        """
        # 将输入从 [batch_size, seq_len, embedding_dim] 转换为 [seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        
        # Transformer 编码，输出形状为 [seq_len, batch_size, embedding_dim]
        x = self.transformer_encoder(x)
        
        # 取序列的最后一个时间步的输出作为特征向量
        x = x[-1, :, :]  # 形状变为 [batch_size, embedding_dim]
        pred = self.score_regressor(x).squeeze()
        # pred = torch.clamp(pred, 0.0, 1.0)
        confidence = self.confidence_regressor(x).squeeze()
        
        if tgt is not None:
            loss_val = self.calc_loss(pred, confidence, tgt)
        else:
            loss_val = 0.0
            
        return pred, confidence, loss_val
    
    def calc_loss(self, pred, confidence, tgt):
        return self.loss_func(pred, confidence, tgt)