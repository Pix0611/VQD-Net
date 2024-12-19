import sys
sys.path.append(".")

import argparse
from tensorboardX import SummaryWriter
import time
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import logging
import argparse
import random
from data.aqa_dataset import AqaDataset
import json

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True