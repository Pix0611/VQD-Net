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

# CONFIG_PATH = r"experiments/config/test0.json"
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--config_path",
    help="Config File Path.",
    type=str,  # 接受字符串参数
    default="experiments/config/example_with_model.json"
)
# 重置日志参数（布尔开关）
parser.add_argument(
    "--reset_log",
    help="Delete old log and tensorboard.",
    action="store_true",  # 布尔标志
    default=False
)
args = parser.parse_args()

print("Loading Config From {} And Writing into Log File...".format(args.config_path))
config = json.load(open(args.config_path, "r"))  # 填写配置文件的路径

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


# 设置随机种子
from experiments.tools.random_seed import setup_seed
setup_seed(config["random_seed"])

# device
cuda_idx = config["gpu_idx"]
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")

# 设置模型和优化器
model = MainModel().to(device)
# criterion = None # -> 用model自带的模块计算loss，不在外部计算
optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

# 加载数据集
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

# 全监督训练阶段
def supervised_train_one_step():
    model.train()
    for feature, tgt in train_labeled_loader:
        feature = feature.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        pred, confidence, loss = model(feature, tgt)
        # loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()
        
# 半监督训练阶段，逐步加入伪标签数据
def semi_supervised_train_one_step(current_threshold):
    model.train()
    # 使用带标签数据进行训练
    for feature, tgt in train_labeled_loader:
        feature = feature.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        pred, confidence, loss = model(feature, tgt)
        # loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()
    
    # 使用无标签数据进行训练
    for feature, _ in train_unlabeled_loader:
        feature = feature.to(device)
        
        optimizer.zero_grad()
        with torch.no_grad():
            pred, confidence, _ = model(feature)
        
        # 选择置信度高于当前阈值的样本
        high_confidence_mask = confidence > current_threshold
        high_confidence_features = feature[high_confidence_mask]
        high_confidence_preds = pred[high_confidence_mask]
        
        # 将高置信度样本加入训练
        if len(high_confidence_features) > 0:
            pseudo_labels = high_confidence_preds.detach()  # 假标签
            pred, confidence, pseudo_loss = model(high_confidence_features, pseudo_labels)
            # pseudo_loss = criterion(pred, pseudo_labels) * 0.5  # 减小伪标签的损失权重
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
    model.eval()  # 设置模型为评估模式
    true_scores = []  # 存储真实分数
    predicted_scores = []  # 存储预测分数
    loss_val = []

    with torch.no_grad():  # 禁用梯度计算以加速评估
        for feature, tgt in dataloader:
            feature = feature.to(device)
            tgt = tgt.to(device)
            # 模型预测
            pred, confidence, loss = model(feature, tgt)
            
            # 记录真实分数和预测分数
            true_scores.extend(tgt.cpu().numpy())  # 将真实值添加到列表
            predicted_scores.extend(pred.cpu().numpy())  # 将预测值添加到列表
            loss_val.append(loss.item()) # 每个batch的loss

    # 计算 Spearman 相关性
    spearman_corr, _ = spearmanr(true_scores, predicted_scores)
    return spearman_corr, sum(loss_val[:-1])/(len(loss_val)-1) if len(loss_val)>1 else loss_val[0]
    

# 主训练循环
initial_threshold = config["initial_threshold"] # 初始置信度阈值
threshold_decay = config["threshold_decay"]  # 每个epoch衰减因子
min_threshold = config["min_threshold"]  # 最低阈值限制
epochs = config["epochs"]
current_threshold = initial_threshold
for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs}, Threshold: {current_threshold:.2f}")
    
    # 阶段1：带标签数据的全监督训练
    supervised_train_one_step()
    
    # 阶段2：半监督训练，逐步加入无标签数据
    # semi_supervised_train_one_step(current_threshold) # 先暂时关闭
    
    # eval: 10epochs/1eval
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
    
    # 动态调整阈值
    current_threshold = max(min_threshold, current_threshold * threshold_decay)
