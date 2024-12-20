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
from models.networks.main_model_example import MainModel
import json


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--config_path",
    help="Config File Path.",
    type=str,
    default="experiments/config/example_with_model.json"
)

parser.add_argument(
    "--reset_log",
    help="Delete old log and tensorboard.",
    action="store_true",
    default=False
)
args = parser.parse_args()

print("Loading Config From {} And Writing into Log File...".format(args.config_path))
config = json.load(open(args.config_path, "r"))

exp_name = args.config_path.split("/")[-1][:-5]
log_path = os.path.join(r"experiments/log", "{}.log".format(exp_name))
tensorboard_path = "experiments/log/tensorboard_{}".format(exp_name)
if args.reset_log:
    os.system("rm {} -f".format(log_path))
    os.system("rm {} -rf".format(tensorboard_path))
logging.basicConfig(filename=log_path, level=logging.INFO, filemode='a')
tensorboard_writer = SummaryWriter(log_dir = tensorboard_path)
logging.info("New Task Started...")
logging.info("Experiment config:")
logging.shutdown()
os.system("cat {}>>{}".format(args.config_path, log_path))
os.system("echo  >> {}".format(log_path))



from experiments.tools.random_seed import setup_seed
setup_seed(config["random_seed"])


cuda_idx = config["gpu_idx"]
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")


model = MainModel().to(device)

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])


dataset_train_ratio = config["train_raio"]
train_labeled_ratio = config["labeled_sample_ratio"]
main_dataset = config["main_dataset"]
sub_dataset = config["sub_dataset"]
B = config["batch_size"]

dataset = AqaDataset(dataset_used=main_dataset, subset=sub_dataset)

total_sample_num = len(dataset)
train_labeled_sample_num = int(total_sample_num*dataset_train_ratio*train_labeled_ratio)
train_unlabeled_sample_num = int(total_sample_num*dataset_train_ratio) - train_labeled_sample_num
test_sample_num = total_sample_num - train_labeled_sample_num - train_unlabeled_sample_num
train_labeled_dataset, train_unlabeled_dataset, test_dataset = random_split(dataset, lengths=[train_labeled_sample_num, train_unlabeled_sample_num, test_sample_num])
logging.info("Nums of samples: Labeled Training: {}, Unlabled Training: {}, Test: {}".format(
    len(train_labeled_dataset), len(train_unlabeled_dataset), len(test_dataset)))

train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=B)
train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=B)
test_loader = DataLoader(test_dataset, batch_size=B)


def supervised_train_one_step():
    model.train()
    for feature, tgt in train_labeled_loader:
        feature = feature.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        pred, confidence, loss = model(feature, tgt)

        loss.backward()
        optimizer.step()
        

def semi_supervised_train_one_step(current_threshold):
    model.train()

    for feature, tgt in train_labeled_loader:
        feature = feature.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        pred, confidence, loss = model(feature, tgt)

        loss.backward()
        optimizer.step()
    

    for feature, _ in train_unlabeled_loader:
        feature = feature.to(device)
        
        optimizer.zero_grad()
        with torch.no_grad():
            pred, confidence, _ = model(feature)
        

        high_confidence_mask = confidence > current_threshold
        high_confidence_features = feature[high_confidence_mask]
        high_confidence_preds = pred[high_confidence_mask]
        

        if len(high_confidence_features) > 0:
            pseudo_labels = high_confidence_preds.detach()
            pred, confidence, pseudo_loss = model(high_confidence_features, pseudo_labels)

            pseudo_loss *= 0.5
            pseudo_loss.backward()
            optimizer.step()    
    
def evaluate(dataloader):
    """
    评估函数，计算模型预测的分数与真实分数的 Spearman 相关性。

    Args:
        dataloader (DataLoader): 需要评估的数据加载器
        model (nn.Module): 已经训练好的模型

    Returns:
        spearman_corr (float): 预测分数和真实分数的 Spearman 相关性
    """
    model.eval()
    true_scores = []
    predicted_scores = []
    loss_val = []

    with torch.no_grad():
        for feature, tgt in dataloader:
            feature = feature.to(device)
            tgt = tgt.to(device)

            pred, confidence, loss = model(feature, tgt)
            

            true_scores.extend(tgt.cpu().numpy())
            predicted_scores.extend(pred.cpu().numpy())
            loss_val.append(loss.item())


    spearman_corr, _ = spearmanr(true_scores, predicted_scores)
    return spearman_corr, sum(loss_val[:-1])/(len(loss_val)-1) if len(loss_val)>1 else loss_val[0]
    


initial_threshold = config["initial_threshold"]
threshold_decay = config["threshold_decay"]
min_threshold = config["min_threshold"]
epochs = config["epochs"]
current_threshold = initial_threshold
for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs}, Threshold: {current_threshold:.2f}")
    

    supervised_train_one_step()
    


    

    if (epoch-1) % 10 == 0:
        spearman_corr_train_labeled, loss_train_labeled  = evaluate(train_labeled_loader)
        logging.info(f"Trainset Labeled Spearman Correlation: {spearman_corr_train_labeled:.4f}")
        
        spearman_corr_train_unlabeled, loss_train_unlabeled  = evaluate(train_unlabeled_loader)    
        logging.info(f"Trainset Unlabeled Spearman Correlation: {spearman_corr_train_unlabeled:.4f}")
        
        spearman_corr_test, loss_test = evaluate(test_loader)
        logging.info(f"Testset Spearman Correlation: {spearman_corr_test:.4f}")
        
        tensorboard_writer.add_scalar("Labeled_Train_Set_Sp-corr", spearman_corr_train_labeled, epoch+1)
        tensorboard_writer.add_scalar("Unlabeled_Train_Set_Sp-corr", spearman_corr_train_unlabeled, epoch+1)
        tensorboard_writer.add_scalar("Test_Set_Sp-corr", spearman_corr_test, epoch+1)
        tensorboard_writer.add_scalar("Labeled_Train_Set_Loss", loss_train_labeled, epoch+1)
        tensorboard_writer.add_scalar("Unlabeled_Train_Set_Loss", loss_train_unlabeled, epoch+1)
        tensorboard_writer.add_scalar("Test_Set_Loss", loss_test, epoch+1)
    

    current_threshold = max(min_threshold, current_threshold * threshold_decay)
