import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
# from torch.autograd import Function
# from torchvision import models
from torchsummary import summary

from models.Conv1dBlock import Conv1dBlock
from models.Networks import Network_bearing, Network_fan
from datasets.load_bearing_data import ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA
from datasets.load_fan_data import ReadMIMII, ReadScenarioData

import numpy as np
import yaml
import itertools
import copy
import random
import os
import time
import sys
from builtins import object
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz, hilbert
import pickle
import matplotlib.pyplot as plt
import math

# self-made utils
from utils.DictObj import DictObj
from utils.AverageMeter import AverageMeter
from utils.CalIndex import cal_index
# from utils.SetSeed import set_random_seed
from utils.SetSeed import set_random_seed
from utils.SimpleLayerNorm import LayerNorm
from utils.TuneReport import GenReport
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset, MultiInfiniteDataLoader
# from utils.SignalTransforms import AddGaussianNoise, RandomScale, MakeNoise, Translation
# from utils.LMMD import LMMDLoss
from utils.GradientReserve import grad_reverse

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/CCDG.py


with open(os.path.join(sys.path[0], 'config_files/CCDG_config.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'


class DomainContrastiveLoss(nn.Module):
    '''
    Copied from
    https://github.com/mohamedr002/CCDG
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device

    def forward(self, domains_features, domains_labels, temperature=0.7):
            # masking for the corresponding class labels.
        anchor_feature = domains_features
        anchor_feature = F.normalize(anchor_feature, dim=1)
        labels = domains_labels
        labels= labels.contiguous().view(-1, 1)
        # Generate masking for positive and negative pairs.
        mask = torch.eq(labels, labels.T).float().to(self.device)
        # Applying contrastive via product among the features
        # Measure the similarity between all samples in the batch
        # reiterate fact from Linear Algebra if u and v two vectors are normalised implies cosine similarity
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

        # for numerical stability
        # substract max value from the output
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # create inverted identity matrix with same shape as mask.
        # only diagnoal is zeros and all others are ones
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(self.device), 0)

        # 上边这个完全可以使用： logits_mask = 1-torch.eye(mask.shape[0])

        # mask-out self-contrast cases
        # the diagnoal represent same samples similarity i=j>> we need to remove this
        # remove the true labels mask
        # all ones in mask would be only samples with same labels
        mask = mask * logits_mask

        # compute log_prob and remove the diagnal
        # remove same features from the normalized contrastive matrix
        # The denominoator of the equation samples
        exp_logits = torch.exp(logits) * logits_mask

        # substract the whole multiplications from the negative pair and positive pairs
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        zeros_idx = torch.where(mask_sum == 0)[0]
        mask_sum[zeros_idx] = 1

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = (- 1 * mean_log_prob_pos)
        loss = loss.mean()

        return loss



class CCDG(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = configs.use_domain_weight

        self.domain_contrastive_loss = DomainContrastiveLoss(configs).to(self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1          # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1

        if self.dataset_type=='bearing':
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type=='fan':
            self.model = Network_fan(configs).to(self.device)
        else:
            raise ValueError('The dataset_type should be bearing or fan!')
        self.optimizer = torch.optim.SGD(list(self.model.parameters()), lr=self.lr)
        self.weight_step = None

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)

        fv, logits = self.model(x)

        if  self.weight_step is None:
            self.weight_step = torch.ones(x.shape[0]).to(self.device)

        # ce_loss = F.cross_entropy(logits, labels)
        ce_loss = torch.mean(self.cross_entropy_loss(logits, labels)*self.weight_step)

        assert not torch.any(torch.isnan(fv)) #这个loss是施加在logits上的
        dc_loss = self.domain_contrastive_loss(fv, labels)

        loss = ce_loss + dc_loss
        self.optimizer.zero_grad()
        loss.backward()
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        self.optimizer.step()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = ce_loss.detach().cpu().data.numpy()
        losses['loss_id'] = dc_loss.detach().cpu().data.numpy()

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)

        loss_acc_result = {'loss_total': [], 'loss_ce':[], 'loss_id':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            loss_acc_result['loss_total'].append(losses['loss_total'])
            loss_acc_result['loss_ce'].append(losses['loss_ce'])
            loss_acc_result['loss_id'].append(losses['loss_id'])

            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step==self.steps:
                acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                #################
                # 仅在有多个测试结果时（例如，在旧的section模式下）才更新域权重
                if self.use_domain_weight and len(acc_results) > 1:
                    weight_step = torch.from_numpy(1 - np.array(acc_results[1:])).type(torch.float32)
                    self.weight_step = weight_step.repeat((self.batch_size, 1)).T.flatten(0).to(self.device)
                else:
                    self.weight_step = None  # 在场景模式下重置或不使用
                print("*"*60)
                if auc_results[0] > self.best_auc:
                    print("auc_result[0]:", auc_results[0])
                    self.best_auc = auc_results[0]
                if acc_results[0] > self.best_acc:
                    print("acc_result[0]:", acc_results[0])
                    self.best_acc = acc_results[0]
                if f1_results[0] > self.best_F1_score:
                    print("f1_results[0]:", f1_results[0])
                    self.best_F1_score = f1_results[0]
                if prec_results[0] > self.best_precision:
                    print("prec_results[0]:", prec_results[0])
                    self.best_precision = prec_results[0]
                if (recall_result[0] > self.best_recall) and (recall_result[0] < 1):
                    self.best_recall = recall_result[0]
                    print("best recall:",self.best_recall )
                print("*"*60)

        return loss_acc_result

    def test_model(self, loaders):
        # 将模型设置为评估模式
        self.eval()

        # 初始化用于存储所有加载器结果的列表
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []

        # 使用无梯度的上下文管理器，以提高效率并减少内存使用
        with torch.no_grad():
            # 遍历传入的每一个数据加载器
            for the_loader in loaders:
                # 这是一个健壮性检查，以防传入的列表中包含None
                if the_loader is None:
                    continue

                # 为当前加载器初始化用于累积结果的列表
                y_true_lst, y_pred_lst, y_prob_lst = [], [], []

                # 遍历当前加载器中的所有批次(batch)
                for x, label_fault in the_loader:
                    # 将数据移动到正确的设备
                    x = x.to(self.device)
                    label_fault = label_fault.to(self.device)

                    # 模型前向传播以获取预测结果
                    # 注意：这里的模型输出逻辑需要与您具体模型（CCDG, ADN等）的forward/predict方法一致
                    # 以CCDG为例：
                    _, logits = self.model(x)
                    probabilities = F.softmax(logits, dim=1)
                    y_pred = torch.argmax(logits, dim=1)

                    # 收集真实标签、预测标签和预测概率
                    y_true_lst.extend(label_fault.cpu().numpy())
                    y_pred_lst.extend(y_pred.cpu().numpy())
                    y_prob_lst.extend(probabilities.cpu().numpy())

                # 在处理完一个加载器的所有数据后，将列表转换为Numpy数组
                y_true = np.array(y_true_lst)
                y_pred_labels = np.array(y_pred_lst)
                y_pred_probs = np.array(y_prob_lst)

                # 调用cal_index函数计算各项性能指标
                acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred_labels, y_pred_probs)

                # 将当前加载器的评估结果追加到总结果列表中
                acc_results.append(acc_i)
                auc_results.append(auc_i)
                prec_results.append(prec_i)
                recall_result.append(recall_i)
                f1_results.append(f1_i)

        # 所有测试完成后，将模型恢复到训练模式
        self.train()

        # 返回包含所有测试域结果的列表
        return acc_results, auc_results, prec_results, recall_result, f1_results

    def predict(self, x):
        with torch.no_grad():
            _, logits = self.model(x)
            return torch.max(logits, dim=1)[1]


def main(idx, configs):
    """主函数（实验入口），支持两种数据集模式"""
    # =================================================================
    # ========== 1. Determine Mode and Define Domains =================
    # =================================================================

    # Check if we are in the new scenario mode based on the config value
    is_scenario_mode = str(configs.fan_section).startswith('s')

    datasets_src = []
    datasets_tgt = []

    if is_scenario_mode:
        # --- LOGIC FOR NEW SCENARIO-BASED DATASETS ---
        print(f"INFO: Running in SCENARIO mode for scenario: {configs.fan_section}")
        scenario = configs.fan_section
        scenario_definitions = {
            # --- 原始场景 (3源, 1目标) ---
            's1': {'source': ['id_00', 'id_02', 'id_04'], 'target': ['id_06']},
            's2': {'source': ['id_00', 'id_02', 'id_06'], 'target': ['id_04']},
            's3': {'source': ['id_00', 'id_04', 'id_06'], 'target': ['id_02']},
            's4': {'source': ['id_02', 'id_04', 'id_06'], 'target': ['id_00']},
            # --- 新增场景 (2源, 2目标) ---
            's5': {'source': ['id_00', 'id_02'], 'target': ['id_04', 'id_06']},
            's6': {'source': ['id_00', 'id_04'], 'target': ['id_02', 'id_06']},
            's7': {'source': ['id_00', 'id_06'], 'target': ['id_02', 'id_04']},
            's8': {'source': ['id_02', 'id_04'], 'target': ['id_00', 'id_06']},
            's9': {'source': ['id_02', 'id_06'], 'target': ['id_00', 'id_04']},
            's10': {'source': ['id_04', 'id_06'], 'target': ['id_00', 'id_02']},
            # --- 新增场景 (1源, 3目标) ---
            's11': {'source': ['id_00'], 'target': ['id_02', 'id_04', 'id_06']},
            's12': {'source': ['id_02'], 'target': ['id_00', 'id_04', 'id_06']},
            's13': {'source': ['id_04'], 'target': ['id_00', 'id_02', 'id_06']},
            's14': {'source': ['id_06'], 'target': ['id_00', 'id_02', 'id_04']},
        }
        if scenario not in scenario_definitions:
            raise ValueError(f"Unknown scenario: {scenario}")

        datasets_src = scenario_definitions[scenario]['source']
        datasets_tgt = scenario_definitions[scenario]['target']

        # Instantiate the correct data loader objects
        # Note: You may need to pass 'seed' if the constructor requires it.
        datasets_object_src = [ReadScenarioData(scenario, domain_id, configs=configs) for domain_id in datasets_src]
        datasets_object_tgt = [ReadScenarioData(scenario, domain_id, configs=configs) for domain_id in datasets_tgt]

    else:
        # --- LOGIC FOR OLD SECTION-BASED DATASETS ---
        print(f"INFO: Running in SECTION mode for section: {configs.fan_section}")
        section = str(configs.fan_section).zfill(2)

        # Define the list of domains for the section
        if section == '00':
            datasets_list = ['W', 'X', 'Y', 'Z']
        elif section == '01':
            datasets_list = ['A', 'B', 'C']
        else:  # section '02'
            datasets_list = ['L1', 'L2', 'L3', 'L4']

        # Use the original leave-one-out logic
        tgt_idx = [idx]
        src_idx = [i for i in range(len(datasets_list)) if i not in tgt_idx]

        datasets_tgt = [datasets_list[i] for i in tgt_idx]
        datasets_src = [datasets_list[i] for i in src_idx]

        # Instantiate the correct data loader objects
        # Note: You may need to pass 'seed' if the constructor requires it.
        datasets_object_src = [ReadMIMII(domain, section=section, configs=configs) for domain in datasets_src]
        datasets_object_tgt = [ReadMIMII(domain, section=section, configs=configs) for domain in datasets_tgt]

    # =================================================================
    # ========== 2. Generic Code (Should Work for Both Modes) ========
    # =================================================================

    # Update configs (this is good practice)
    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src

    # Create the dataloaders from the objects
    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src]
    test_loaders_src = [test for train, test in train_test_loaders_src]

    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt]
    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)


    currtime = str(time.time())[:10]
    for i in range(1):
        model = CCDG(configs)

        all_test_loaders = test_loaders_tgt + test_loaders_src
        valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

        # 执行训练
        loss_acc_result = model.train_model(
            train_minibatches_iterator,
            valid_test_loaders  # <-- 正确：传递清理后的列表
        )


        loss_acc_result['loss_total'] = np.array(loss_acc_result['loss_total'])
        loss_acc_result['loss_ce'] = np.array(loss_acc_result['loss_ce'])
        loss_acc_result['loss_id'] = np.array(loss_acc_result['loss_id'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])

        print("===========================================================================================")
        print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

        save_dir = os.path.join('result', 'CCDG')  # 只定义目录路径
        filename = f'section0{configs.fan_section}_best_result.txt'  # 单独定义文件名
        file_path = os.path.join(save_dir, filename)  # 组合完整路径

        # 确保目录存在（仅创建目录）
        os.makedirs(save_dir, exist_ok=True)

        # 添加异常处理
        try:
            with open(file_path, 'a') as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                f.write(f"\nBest ACC: \n{model.best_acc:.4f}")
                f.write(f"\nBest AUC: \n{model.best_auc:.4f}")
                f.write(f"\nBest precision: \n{model.best_precision:.4f}\n")
                f.write(f"\nBest recall: \n{model.best_recall:.4f}\n")
                f.write(f"\nBest F1: \n{model.best_F1_score:.4f}\n")
        except PermissionError as pe:
            print(f"无法写入 {file_path}，错误详情：{str(pe)}")
            # 尝试备用路径（用户主目录）
            home_path = os.path.expanduser("~")
            backup_path = os.path.join(home_path, filename)
            print(f"尝试保存到备用路径：{backup_path}")
            with open(backup_path, 'a') as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                f.write(f"\nBest ACC: \n{model.best_acc:.4f}")
                f.write(f"\nBest AUC: \n{model.best_auc:.4f}")
                f.write(f"\nBest precision: \n{model.best_precision:.4f}\n")
                f.write(f"\nBest recall: \n{model.best_recall:.4f}\n")
                f.write(f"\nBest F1: \n{model.best_F1_score:.4f}\n")


if __name__ == '__main__':
    section_s = 's1','s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section = '00', '01', '02'
    sectionNumber = 3
    sectionsNumber = 14
    for i in range(sectionsNumber):
        run_times = 10
        configs.fan_section = section_s[i]
        for _ in range(run_times):
            print('----------------------------------------------------------', _)
            main( 0, configs)


