import sys
sys.path.append(".")

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.Transformer import *

class UsdlHead(nn.Module):
    def __init__(self, in_dim, heads_num):
        super().__init__()
        self.linear_projection = nn.Sequential(
            nn.Linear(in_dim, 256),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, heads_num),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.linear_projection(x)
        return x
    


if __name__ == '__main__':
    input = torch.zeros((3, 1024))
    model = UsdlHead(in_dim=1024, heads_num=100)
    output = model(input)
    print(output.shape)