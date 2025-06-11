import os
import time
from distutils.command.config import config

import torch
from sympy import false
from torch.distributed.checkpoint import optimizer
from torch.utils.data import DataLoader
import sys
from DGCDN import DGCDN
from utils.CreateLogger import create_logger
from utils.DatasetClass import SimpleDataset
from utils.DictObj import DictObj
import yaml
import numpy as np
from datasets.load_DGCDN_data import ReadMIMII
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from utils.SetSeed import set_random_seed

# 强制使用CPU（如果训练时用CPU）或自动检测
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

# 2. 加载保存的检查点
filename_ = r'section01\section01_acc0.7750_auc0.7467_f10.7484_20250610_012422.pth'
# checkpoint_path = r'C:\Users\Malones\Desktop\DGCDN\保存的结果\\'+filename_
checkpoint_path = r'E:\code\myMethod-20250415\checkpoints\\'+filename_
checkpoint = torch.load(checkpoint_path, map_location=device)

configs = checkpoint['configs']

if configs.fan_section == 'sec00' or configs.fan_section == '00' or configs.fan_section == '0' or configs.fan_section == 0:
    datasets_list = ['W', 'X', 'Y', 'Z']
    section = '00'
elif configs.fan_section == 'sec01' or configs.fan_section == '01' or configs.fan_section == '1' or configs.fan_section == 1:
    datasets_list = ['A', 'B', 'C']
    section = '01'
else:
    datasets_list = ['L1', 'L2', 'L3', 'L4']
    section = '02'

# 划分源域和目标域（留一法验证）
dataset_idx = list(range(len(datasets_list)))
tgt_idx = [0]  # 目标域索引（当前实验）
src_idx = [i for i in dataset_idx if i not in tgt_idx]  # 源域索引
datasets_tgt = [datasets_list[i] for i in tgt_idx]  # 目标域数据集名称
datasets_src = [datasets_list[i] for i in src_idx]  # 源域数据集名称
configs.datasets_tgt = datasets_tgt  # 更新配置对象
configs.datasets_src = datasets_src

seed = checkpoint['seed']
bool_acc = checkpoint['bool_acc']
bool_auc = checkpoint['bool_auc']
bool_f1_score = checkpoint['bool_f1']
weights = checkpoint['class_weights']

set_random_seed(seed)
# 3. 初始化模型并加载权重
model = DGCDN(configs, seed, weights).to(device)
model.encoder_m.load_state_dict(checkpoint['encoder_m'])
model.encoder_h.load_state_dict(checkpoint['encoder_h'])
model.decoder.load_state_dict(checkpoint['decoder'])
model.classifer.load_state_dict(checkpoint['classifier'])
model.optimizer.load_state_dict(checkpoint['optimizer'])
model.attention.load_state_dict(checkpoint['attention'])

configs_dict = vars(checkpoint['configs'])  # 通过 __dict__ 属性获取内部字典
print("\n=== Configs 内容 ===")
for key, value in configs_dict.items():
    print(f"{key}: {value}")




# 6. 执行测试（示例测试单个域）
if __name__ == '__main__':
    # 测试配置（根据实际情况修改）

    if configs.fan_section == 'sec00' or configs.fan_section == '00' or configs.fan_section == '0' or configs.fan_section == 0:
        test_domain = 'W'
        test_section = '00'
    elif configs.fan_section == 'sec01' or configs.fan_section == '01' or configs.fan_section == '1' or configs.fan_section == 1:
        test_domain = 'A'
        test_section = '01'
    else:
        test_domain = 'L1'
        test_section = '02'

    model.eval()

    # 划分源域和目标域（留一法验证）
    dataset_idx = list(range(len(datasets_list)))
    tgt_idx = [0]  # 目标域索引（当前实验）
    src_idx = [i for i in dataset_idx if i not in tgt_idx]  # 源域索引
    datasets_tgt = [datasets_list[i] for i in tgt_idx]  # 目标域数据集名称
    datasets_src = [datasets_list[i] for i in src_idx]  # 源域数据集名称
    configs.datasets_tgt = datasets_tgt  # 更新配置对象
    configs.datasets_src = datasets_src
    # 创建训练和测试数据加载器
    datasets_object_src = [ReadMIMII(i, seed, section=section, configs=configs) for i in datasets_src]
    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train,test in train_test_loaders_src]
    test_loaders_src  = [test for train,test in train_test_loaders_src]
    datasets_object_tgt = [ReadMIMII(i, seed, section=section, configs=configs) for i in datasets_tgt]
    train_test_loaders_tgt = [dataset.load_dataloaders() for dataset in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt]  # 目标域测试集

    currtime = str(time.time())[:10]  # 用时间戳创建唯一文件名
    # 创建日志目录
    full_path_log = os.path.join('Output//myMethod//log_files', datasets_list[0])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)  # 自动创建目录

    # 创建报告目录
    full_path_rep = os.path.join('Output//myMethod//TuneReport', datasets_list[0])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)
    logger = create_logger(full_path_log +'//log_file'+currtime)
    model.logger = logger
    acc_results, auc_results, f1_results = model.test_model(test_loaders_tgt + test_loaders_src)
    print(f"准确率: {acc_results[0]:.4f}")
    print(f"AUC: {auc_results[0]:.4f}")
    print(f"F1分数: {f1_results[0]:.4f}")
    # 多域混合、颜色分域，查看健康特征分布
    model.visualize_tsne_mixed_domains(test_loaders_tgt + test_loaders_src, feature_type='health',
                                       title=filename_)

    # 或查看机器特征
    # model.visualize_tsne_mixed_domains(loaders, feature_type='machine')
