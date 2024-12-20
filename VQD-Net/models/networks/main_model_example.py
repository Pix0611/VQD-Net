import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
    
    def forward(self, pred, confidence, tgt):

        B = pred.shape[0]
        return torch.sum((pred-tgt)**2) / B


class MainModel(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, nhead=4, num_layers=4):

        super(MainModel, self).__init__()
        

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        

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


        x = x.permute(1, 0, 2)
        

        x = self.transformer_encoder(x)
        

        x = x[-1, :, :]
        pred = self.score_regressor(x).squeeze()

        confidence = self.confidence_regressor(x).squeeze()
        
        if tgt is not None:
            loss_val = self.calc_loss(pred, confidence, tgt)
        else:
            loss_val = 0.0
            
        return pred, confidence, loss_val
    
    def calc_loss(self, pred, confidence, tgt):
        return self.loss_func(pred, confidence, tgt)