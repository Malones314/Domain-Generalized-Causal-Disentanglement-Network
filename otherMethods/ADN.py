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
from models.Networks import FeatureEncoder_adn_bearing, Classifier_adn_bearing, Discriminator_adn_bearing, FeatureEncoder_adn_fan, Classifier_adn_fan, Discriminator_adn_fan
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
# from utils.AverageMeter import AverageMeter
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
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/ADN.py


with open(os.path.join(sys.path[0], 'config_files/ADN_config.yaml'), encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'


class DistanceMetricLoss(nn.Module):
    '''
    B = 20
    D = 1984
    x = torch.randn((B,D))
    configs.num_domains = 4
    domain_labels = torch.tensor([0]*5 +[1]*5 +[2]*5 +[3]*5)
    labels = torch.randint(low = 0, high = configs.num_classes, size=(B,))
    model = DistanceMetricLoss(configs)
    loss = model(x, labels)
    '''

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.num_classes = configs.num_classes
        self.dim_feature = configs.dim_feature
        self.num_domains = configs.num_domains#len(configs.datasets_src)


    def forward(self, x, labels):
        B = x.shape[0]
        C = self.num_classes
        D = self.dim_feature
        P = self.num_domains

        # L_intra: Intra-class distance
        list_category = [[] for i in range(self.num_classes)]
        for i, fv in zip(labels, x):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)

        intra_loss = 0
        centers  = []
        # Loop through all possible classes
        for i in range(self.num_classes):
            # ======================= START OF FIX 1 =======================
            # Only process categories that are actually present in the batch
            if len(list_category[i]) > 0:
                sample_num_class_i = len(list_category[i]) # Number of samples for class i
                fv_i = torch.cat(tuple(list_category[i]), dim=0) # This is now safe
                center_i = torch.mean(fv_i, dim=0, keepdim=True) #(1,D)
                centers.append(center_i)
                # Calculate intra-class distance for this specific class
                intra_i = (fv_i - center_i).abs().sum().div(sample_num_class_i).div(C*D)
                intra_loss = intra_loss + intra_i
            # ======================== END OF FIX 1 ========================

        # L_inter: Inter-class distance
        # ======================= START OF FIX 2 =======================
        # Only calculate inter-class loss if more than one class is present in the batch
        if len(centers) < 2:
            inter_loss = torch.tensor(0.0, device=self.device)
        else:
            self.centers = torch.cat(centers, dim=0).to(self.device) # (num_present_classes, D)
            center_center = self.centers.mean(dim=0, keepdim=True)   # (1,D)
            center_center1 = center_center.expand(len(centers), D) # (num_present_classes, D)
            inter_loss = (center_center1 - self.centers).abs().sum().div(len(centers)*D)
        # ======================== END OF FIX 2 ========================

        total_loss = intra_loss - inter_loss

        return total_loss


#####################
class DGDNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1          # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        configs.num_domains = len(configs.datasets_src)
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type
        self.num_domains = configs.num_domains


        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = configs.use_domain_weight

        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler

        if self.dataset_type== 'bearing':
            self.fe = FeatureEncoder_adn_bearing(configs).to(self.device)
            self.clf = Classifier_adn_bearing(configs).to(self.device)
            self.dcn = Discriminator_adn_bearing(configs).to(self.device)
        elif self.dataset_type== 'fan':
            self.fe = FeatureEncoder_adn_fan(configs).to(self.device)
            self.clf = Classifier_adn_fan(configs).to(self.device)
            self.dcn = Discriminator_adn_fan(configs).to(self.device)

        self.dm_loss = DistanceMetricLoss(configs)

        self.optimizer = torch.optim.Adagrad(list(self.fe.parameters()) + list(self.clf.parameters()) + list(self.dcn.parameters()), lr=self.lr)

        self.cl_loss = nn.CrossEntropyLoss(reduction='none')

        self.lbda_cl = configs.lbda_cl
        self.lbda_dc = configs.lbda_dc
        self.lbda_dm = configs.lbda_dm
        self.weight_step = None

    def adjust_learning_rate(self, step):
        """
        Decay the learning rate based on schedule
        https://github.com/facebookresearch/moco/blob/main/main_moco.py
        """
        lr = self.lr
        if self.configs.cos:  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:  # stepwise lr schedule
            for milestone in self.configs.schedule:
                lr *= 0.5 if step >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        # debug
        # x_shape = [x.shape[0] for x,y in minibatches]
        # print('The shape of X',x_shape)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)
        # domain labels
        domain_labels = []
        for domain_idx, (x_d, _) in enumerate(minibatches):
            domain_labels.extend([domain_idx] * x_d.size(0))
        domain_labels = torch.tensor(domain_labels, dtype=torch.long).to(self.device)

        feature_vectors = self.fe(x)
        logits = self.clf(feature_vectors)
        domain_logits = self.dcn(grad_reverse(feature_vectors))

        if self.use_domain_weight:
            if  self.weight_step is None:
                self.weight_step = torch.ones(x.shape[0]).to(self.device)
            else:
                ce_values  = self.cl_loss(logits, labels)
                ce_values_2d = torch.reshape(ce_values, (self.num_domains, self.batch_size))
                ce_value_domain = torch.mean(ce_values_2d,dim=1)
                ce_value_sum = torch.sum(ce_value_domain)
                weight_step = 1 + ce_value_domain/ce_value_sum
                self.weight_step = weight_step.repeat((self.batch_size,1)).T.flatten(0).to(self.device)
        else:
            self.weight_step = torch.ones(x.shape[0]).to(self.device)

        cl_loss = torch.mean(self.cl_loss(logits, labels)*self.weight_step)
        dc_loss = F.cross_entropy(domain_logits, domain_labels)
        dm_loss = self.dm_loss(feature_vectors, labels)
        total_loss = self.lbda_cl*cl_loss  + self.lbda_dc*dc_loss + self.lbda_dm*dm_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


        loss_cl = cl_loss.detach().cpu().data.numpy()
        loss_dc = dc_loss.detach().cpu().data.numpy()
        loss_dm = dm_loss.detach().cpu().data.numpy()


        losses={}
        losses['cl'] = loss_cl
        losses['dc'] = loss_dc
        losses['dm'] = loss_dm

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)

        # loss_acc_result = {'loss_cc': [], 'loss_ct':[], 'loss_cl':[], 'acces':[]}
        loss_acc_result = {'loss_cl':[], 'loss_dc': [], 'loss_dm':[], 'acces':[]}

        for step in range(1, self.steps+1):
            print(step)
            self.train()
            self.current_step = step
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)
            # self.scheduler.step()
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            loss_acc_result['loss_cl'].append(losses['cl'])
            loss_acc_result['loss_dc'].append(losses['dc'])
            loss_acc_result['loss_dm'].append(losses['dm'])

            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step == self.steps:
                acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
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
        self.eval()
        num_loaders = len(loaders)
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []

        for i in range(num_loaders):
            the_loader = loaders[i]
            y_pred_lst = []
            y_prob_lst = []
            y_true_lst = []

            for j, batched_data in enumerate(the_loader):
                x, label_fault = batched_data
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)

                label_pred, prob_pred = self.predict(x)
                y_pred_lst.extend(label_pred.cpu().numpy())
                y_prob_lst.extend(prob_pred.cpu().numpy())  # prob_pred 是二维的
                y_true_lst.extend(label_fault.cpu().numpy())

            y_true = np.array(y_true_lst)
            y_pred_labels = np.array(y_pred_lst)
            y_pred_probs = np.array(y_prob_lst)

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
        with torch.no_grad():
            fv = self.fe(x)
            logits = self.clf(fv)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds, probs  # 注意：返回的是整个二维概率向量


def main(idx, configs):
    """主函数（实验入口），支持两种数据集模式"""
    # =================================================================
    # ========== 1. Determine Mode and Define Domains =================
    # =================================================================

    # 检查当前是旧的section模式还是新的scenario模式
    is_scenario_mode = str(configs.fan_section).startswith('s')

    if is_scenario_mode:
        # =================================================================
        # ========== 1. 新的数据集加载逻辑 (Scenario-based) ==========
        # =================================================================
        scenario = configs.fan_section

        # 根据场景定义源域和目标域
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
            raise ValueError(f"未知的场景: {scenario}。请在 s1-s14 中选择。")

        datasets_src = scenario_definitions[scenario]['source']
        datasets_tgt = scenario_definitions[scenario]['target']

        # 使用 ReadScenarioData 加载器
        datasets_object_src = [ReadScenarioData(scenario, domain_id, configs) for domain_id in datasets_src]
        datasets_object_tgt = [ReadScenarioData(scenario, domain_id, configs) for domain_id in datasets_tgt]

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



    currtime = str(time.time())[:10]
    for i in range(1):
        model = DGDNN(configs)

        # 组合并清理测试加载器，移除所有None值
        all_test_loaders = test_loaders_tgt + test_loaders_src
        valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

        loss_acc_result = model.train_model(
            train_minibatches_iterator,
            valid_test_loaders,  # <-- 传递清理后的有效加载器列表
        )
        loss_acc_result['loss_cl'] = np.array(loss_acc_result['loss_cl'])
        loss_acc_result['loss_dc'] = np.array(loss_acc_result['loss_dc'])
        loss_acc_result['loss_dm'] = np.array(loss_acc_result['loss_dm'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])


        # # save the loss curve and acc curve


        # update current time
        currtime = str(time.time())[:10]

        print("===========================================================================================")
        print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

        save_dir = os.path.join('result', 'ADN')  # 只定义目录路径
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

# if __name__ == '__main__':
#     section_s = 's1','s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
#     section = '00', '01', '02'
#     sectionNumber = 3
#     sectionsNumber = 14
#     for i in range(sectionsNumber):
#         run_times = 5
#         configs.fan_section = section_s[i]
#         for _ in range(run_times):
#             print('----------------------------------------------------------', _)
#             main( 0, configs)
if __name__ == '__main__':
    section_s = 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section = '00', '01', '02'
    sectionNumber = 3
    sectionsNumber = 10
    for i in range(sectionsNumber):
        run_times = 5
        configs.fan_section = section_s[i]
        for _ in range(run_times):
            print('----------------------------------------------------------', _)
            main( 0, configs)
