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
from scipy.signal import butter, lfilter, freqz, hilbert
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt
import math

from models.Conv1dBlock import Conv1dBlock
from models.Networks import FeatureGenerator_bearing, FaultClassifier_bearing, DomainClassifier_bearing, FeatureGenerator_fan, FaultClassifier_fan, DomainClassifier_fan
from datasets.load_bearing_data import ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA
from datasets.load_fan_data import ReadMIMII, ReadScenarioData

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
# from utils.GradientReserve import grad_reverse

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/DGNIS.py




with open(os.path.join(sys.path[0], 'config_files/DGNIS_config.yaml'), 'r', encoding='utf-8') as f:
    '''从YAML文件加载配置参数'''
    configs = yaml.load(f, Loader=yaml.FullLoader)  # 加载YAML文件
    print(configs)  # 打印配置参数
    configs = DictObj(configs)  # 将字典转换为对象形式

    # 设置计算设备（GPU/CPU）
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'  # 使用GPU


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            # Define positive and negative masks for the current anchor `i`
            positive_mask = mask[i]
            negative_mask = (mask[i] == 0)

            # ======================= START OF FIX =======================
            # Check if both positive (other than self) and negative samples exist.
            # A valid triplet requires at least one of each.
            if torch.sum(positive_mask) > 1 and torch.sum(negative_mask) > 0:
                # Find the hardest positive
                dist_ap.append(dist[i][positive_mask].max().unsqueeze(0))
                # Find the hardest negative
                dist_an.append(dist[i][negative_mask].min().unsqueeze(0))
            # ======================== END OF FIX ========================

        # If no valid triplets were found in the entire batch, return a loss of 0.
        if not dist_ap or not dist_an:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class CoralLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.num_domains = len(configs.datasets_src)

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss

    def cal_overall_coral_loss(self, features):
        '''
        Args:
            features, should be a list with feature vectors from different domains
        '''
        loss = 0
        for i in range(self.num_domains):
            for j in range(i+1, self.num_domains):
                loss += self.forward(features[i*self.batch_size:(i+1)*self.batch_size],
                features[j*self.batch_size:(j+1)*self.batch_size])
                # loss += self.forward(features[i], features[j])

        return loss


class DGNIS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.best_auc = -1  # 最佳验证指标
        self.best_acc = -1  # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        self.configs = configs
        self.device = configs.device
        self.dataset_type =  configs.dataset_type

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size =  configs.batch_size
        self.margin = configs.margin

        self.use_domain_weight = configs.use_domain_weight
        self.domain_weight_scale = configs.domain_weight_scale

        self.num_domains = len(configs.datasets_src)
        if self.dataset_type == 'bearing':
            self.fe_inv = FeatureGenerator_bearing(configs).to(self.device)
            self.fe_dom = FeatureGenerator_bearing(configs).to(self.device)
            self.dc = DomainClassifier_bearing(configs).to(self.device)
            self.fc1 = FaultClassifier_bearing(configs).to(self.device)
            self.fc2 = FaultClassifier_bearing(configs).to(self.device)
            self.fc3 = FaultClassifier_bearing(configs).to(self.device)
            self.fc4 = FaultClassifier_bearing(configs).to(self.device)
            self.fc5 = FaultClassifier_bearing(configs).to(self.device)
            self.fc6 = FaultClassifier_bearing(configs).to(self.device)
            self.fc7 = FaultClassifier_bearing(configs).to(self.device)
            self.fc8 = FaultClassifier_bearing(configs).to(self.device)
        elif self.dataset_type == 'fan':
            self.fe_inv = FeatureGenerator_fan(configs).to(self.device)
            self.fe_dom = FeatureGenerator_fan(configs).to(self.device)
            self.dc = DomainClassifier_fan(configs).to(self.device)
            self.fc1 = FaultClassifier_fan(configs).to(self.device)
            self.fc2 = FaultClassifier_fan(configs).to(self.device)
            self.fc3 = FaultClassifier_fan(configs).to(self.device)
            self.fc4 = FaultClassifier_fan(configs).to(self.device)
            self.fc5 = FaultClassifier_fan(configs).to(self.device)
            self.fc6 = FaultClassifier_fan(configs).to(self.device)
            self.fc7 = FaultClassifier_fan(configs).to(self.device)
            self.fc8 = FaultClassifier_fan(configs).to(self.device)

        self.fcs = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8]
        self.fcs = self.fcs[:self.num_domains]

        self.coral_loss = CoralLoss(configs)
        self.triplet_loss =  TripletLoss(margin=self.margin)

        self.optimizer = torch.optim.Adam(params=list(self.parameters()), lr = self.lr)
        self.lbda_cr = configs.lbda_cr
        self.lbda_tp = configs.lbda_tp

        lr_list = [0.0001/(1+10*p/self.steps)**0.75 for p in range(self.steps+1)]
        lambda_para = lambda step: lr_list[step]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_para)

        self.weight_step = None


    def update(self, minibatches):
        self.train()
        # x = [x.to(self.device) for x, y in minibatches] # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        # labels = [y.to(self.device) for x, y in minibatches]
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)

        fv_inv = self.fe_inv(x)

        # calculate the cross entropy of each domain and add them together
        loss_ce_total = 0
        loss_ce_list = []
        for i in range(self.num_domains):
            fv_inv_i = fv_inv[i*self.batch_size:(i+1)*self.batch_size]
            labels_i = labels[i*self.batch_size:(i+1)*self.batch_size]
            logits_i = self.fcs[i](fv_inv_i)
            cross_entropy_i = F.cross_entropy(logits_i, labels_i)
            loss_ce_list.append(cross_entropy_i)
            loss_ce_total += cross_entropy_i


        if self.use_domain_weight:
            if  self.weight_step is None:
                self.weight_step = torch.ones(x.shape[0]).to(self.device)
            else:
                ce_value_domains =  torch.tensor(loss_ce_list).to(self.device)
                weight_step = 1 + ce_value_domains/loss_ce_total
                self.weight_step =  weight_step.to(self.device)

            loss_ce =0
            for i in range(self.num_domains):
                loss_ce +=  self.weight_step[i]*loss_ce_list[i]
        else:
            loss_ce = loss_ce_total


        # calculate the coral loss
        loss_cr = self.coral_loss.cal_overall_coral_loss(fv_inv)
        # calculate the triplet loss
        loss_tp = self.triplet_loss(fv_inv, labels)
        # calculate the domain classification loss

        # 获取每个源域样本数量
        domain_sizes = [x.shape[0] for x, _ in minibatches]

        # 为每个域构造对应的标签
        dom_labels = torch.cat([
            torch.full((size,), i, dtype=torch.long)
            for i, size in enumerate(domain_sizes)
        ]).to(self.device)

        fv_dom = self.fe_dom(x)
        dom_logits  =self.dc(fv_dom)
        loss_cd = F.cross_entropy(dom_logits, dom_labels)

        total_loss = loss_ce + self.lbda_cr*loss_cr + self.lbda_tp*loss_tp + loss_cd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        loss={}
        loss['ce'] = loss_ce.detach().cpu().data.numpy()
        loss['cr'] = loss_cr.detach().cpu().data.numpy()
        loss['tp'] = loss_tp.detach().cpu().data.numpy()
        loss['cd'] = loss_cd.detach().cpu().data.numpy()

        return loss

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)

        loss_acc_result = {'loss_ce': [], 'loss_cr':[], 'loss_tp':[], 'loss_cd':[], 'acces':[]}

        for step in range(0, self.steps):
            self.train()
            self.current_step = step
            try:
                minibatches_device = next(train_minibatches_iterator)
            except StopIteration:
                print("训练数据用尽，可能是 DataLoader/Sampler 未设置为无限模式。")
                exit(1)
            losses = self.update(minibatches_device)

            loss_acc_result['loss_ce'].append(losses['ce'])
            loss_acc_result['loss_cr'].append(losses['cr'])
            loss_acc_result['loss_tp'].append(losses['tp'])
            loss_acc_result['loss_cd'].append(losses['cd'])
            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step==self.steps:
                acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                loss_acc_result.setdefault('auc', []).append(auc_results)

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

        return loss_acc_result

    def test_model(self, loaders):
        self.eval()
        num_loaders = len(loaders)

        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []

        for i in range(num_loaders):
            the_loader = loaders[i]
            y_pred_labels = []
            y_true = []
            y_pred_probs = []

            for _, (x, label_fault) in enumerate(the_loader):
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)

                with torch.no_grad():
                    fv_inv = self.fe_inv(x)

                    logits_all = []
                    for fc in self.fcs:
                        logits_all.append(fc(fv_inv))

                    logits_all = torch.stack(logits_all, dim=1)

                    fv_dom = self.fe_dom(x)
                    domain_weights = self.dc(fv_dom)
                    domain_weights = F.softmax(domain_weights, dim=1).unsqueeze(2)

                    logits_weighted = logits_all * domain_weights
                    logits = logits_weighted.sum(dim=1)

                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                y_true.extend(label_fault.cpu().numpy())
                y_pred_labels.extend(preds.cpu().numpy())
                y_pred_probs.extend(probs.cpu().numpy())

            # 转为 numpy 数组
            y_true = np.array(y_true)
            y_pred_labels = np.array(y_pred_labels)
            y_pred_probs = np.array(y_pred_probs)

            acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred_labels, y_pred_probs)
            acc_results.append(acc_i)
            auc_results.append(auc_i)
            prec_results.append(prec_i)
            recall_result.append(recall_i)
            f1_results.append(f1_i)
        self.train()

        return acc_results, auc_results, prec_results, recall_result, f1_results

    def predict(self, x):
        self.eval()

        fv_inv = self.fe_inv(x)
        logits=[]
        for fc_i in self.fcs:
            logits.append(fc_i(fv_inv).detach().cpu().data.numpy())

        logits = torch.from_numpy(np.array(logits)).to(self.device)

        # logits = torch.tensor(logits).to(self.device) #(num_domain, num_sample, num_classes)
        logits = torch.permute(logits, [1,0,2]) #(num_sample, num_domain, num_classes)

        fv_dom = self.fe_dom(x)
        logits_dc  = self.dc(fv_dom)
        w = F.softmax(logits_dc, dim=1) #(num_sample, num_domain)
        w = torch.unsqueeze(w, dim=2)
        # print(w.shape)
        # print(logits.shape)

        logits_w = logits * w #(num_sample, num_domain, num_classes)
        logits_w_sum = torch.sum(logits_w, dim=1) #(num_sample, num_classes)

        return torch.max(logits_w_sum, dim=1)[1]


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
    # 创建跨域训练数据迭代器
    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)


    # 初始化日志记录器
    currtime = str(time.time())[:10]  # 用时间戳创建唯一文件名

    for i in range(1):
        model = DGNIS(configs)

        # 模型训练
        all_test_loaders = test_loaders_tgt + test_loaders_src
        valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

        # 执行训练
        loss_acc_result = model.train_model(
            train_minibatches_iterator,
            valid_test_loaders # <-- 正确：传递清理后的列表
        )
        # 结果处理
        loss_acc_result = {
            'loss_ce': np.array(loss_acc_result['loss_ce']),
            'loss_cr': np.array(loss_acc_result['loss_cr']),
            'loss_tp': np.array(loss_acc_result['loss_tp']),
            'loss_cd': np.array(loss_acc_result['loss_cd']),
            'acces': np.array(loss_acc_result['acces']),
            'auc': np.array(loss_acc_result['auc']),  # ← 新增
        }

        print("===========================================================================================")
        print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

        save_dir = os.path.join('result', 'DGNIS')  # 只定义目录路径
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
