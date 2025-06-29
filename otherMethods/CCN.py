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
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz, hilbert
import pickle
import matplotlib.pyplot as plt
import math

from models.Conv1dBlock import Conv1dBlock
from models.Networks import Network_bearing, Network_fan
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

# from utils.LMMD import LMMDLoss
# from utils.GradientReserve import grad_reverse

# run code
# srun -w node5 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/CCN.py




with open(os.path.join(sys.path[0], 'config_files/CCN_config.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs) #yaml库读进来的是字典dict格式，需要用DictObj将dict转换成object
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'

class CausalConsistencyLoss(nn.Module):
    '''
    test code:
    x = torch.tensor([[1,0],[1,1],[3,5],[4,2]]).type(torch.float32)
    y = torch.tensor([0,1,1,0])
    ccl_loss = CausalConsistencyLoss(configs)
    total_causal_consistency_loss = ccl_loss(x,y)
    '''
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.device  = configs.device
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)

    def cal_cos_dis(self, x, y):
        '''
        calculate the cosine dissimilarity
        其实就是把原来的Causal Loss中的欧氏距离换成了余弦距离(余弦不相似度)
        '''
        cos_xy = self.cos_sim(x,y)
        loss = 1-abs(cos_xy)

        return loss

    def forward(self, flatten_features, data_label, weight):
        list_category = [[] for i in range(self.num_classes)]
        list_category_weight = [[] for i in range(self.num_classes)]
        # list_label = [[] for i in range(self.num_classes)] # for debug
        for i, fv, w in zip(data_label, flatten_features, weight):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)
            list_category_weight[i].append(w)
            # list_label[i].append(i) # for debug
        # self.list_label = list_label# for debug

        total_cc_loss = 0
        for i in range(self.num_classes):
            if len(list_category[i])>0:
                fm_i = torch.cat(tuple(list_category[i]), dim=0).to(self.device) # convert the feature vector listinto a single tensor(matrix)
                # print(fm_i.shape)
                # print(list_category_weight[i])
                w_i = torch.tensor(list_category_weight[i]).to(self.device)
                fm_i_mean = torch.mean(fm_i, dim=0, keepdim=True).to(self.device) #Z^{\bar}_{k}
                # causal_i = torch.sum(torch.mean((fm_i - fm_i_mean).pow(2), dim=0))
                cc_loss_i = torch.sum(self.cal_cos_dis(fm_i_mean,fm_i)*w_i).to(self.device)

                total_cc_loss = total_cc_loss + cc_loss_i

        total_cc_loss = total_cc_loss/flatten_features.size(0)

        return total_cc_loss
############
class CollaborativeTrainingLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.device = configs.device
        self.ce_loss = nn.CrossEntropyLoss(reduction='none').to(self.device)


    def forward(self, logits, data_label, weight):
        list_category = [[] for i in range(self.num_classes)]
        # list_label = [[] for i in range(self.num_classes)] # for debug
        list_category_weight = [[] for i in range(self.num_classes)]
        for i, fv,w in zip(data_label, logits, weight):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)
            list_category_weight[i].append(w)

        total_ct_loss = 0
        for i in range(self.num_classes):
            if len(list_category[i])>0:
                logits_i = torch.cat(tuple(list_category[i]), dim=0).to(self.device) # convert the logit list into a single tensor(matrix)
                w_i = torch.tensor(list_category_weight[i]).to(self.device)
                label_i = torch.tensor([i]*logits_i.size(0)).to(self.device)
                ce_i = self.ce_loss(logits_i, label_i)
                ce_i_mean = torch.mean(ce_i).to(self.device)

                ct_loss_i = torch.sum((ce_i - ce_i_mean).pow(2)*w_i).to(self.device)
                total_ct_loss = total_ct_loss + ct_loss_i

        total_ct_loss = torch.sqrt(total_ct_loss/logits.size(0)).to(self.device)

        return total_ct_loss

class CCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1          # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type # bearing or fan
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = configs.use_domain_weight
        if self.dataset_type=='bearing':
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type=='fan':
            self.model = Network_fan(configs).to(self.device)
        else:
            raise ValueError('The dataset_type should be bearing or fan!')
        self.optimizer = torch.optim.Adagrad(list(self.model.parameters()), lr=self.lr)
        zz = [1]*25+[0.5]*150+[0.1]*400
        # self.lambda_func = lambda step: 0.95**np.floor(step/5)
        # self.lambda_func = lambda step: 1-step/self.steps
        self.lambda_func = lambda step: zz[step]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lambda_func )

        self.cc_loss = CausalConsistencyLoss(configs)
        self.ct_loss = CollaborativeTrainingLoss(configs)
        self.cl_loss = nn.CrossEntropyLoss(reduction='none')

        self.lbda_cc = configs.lbda_cc
        self.lbda_ct = configs.lbda_ct
        self.num_domains = len(configs.datasets_src)
        self.weight_step = None



    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        # debug
        # x_shape = [x.shape[0] for x,y in minibatches]
        # print('The shape of X',x_shape)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)

        feature_vectors, logits = self.model(x)
        if  self.weight_step is None:
            self.weight_step = torch.ones(x.shape[0]).to(self.device)
        else:
            ce_values = self.cl_loss(logits, labels)  # shape [total_batch_size]
            batch_size_total = ce_values.shape[0]
            batch_sizes = [x.shape[0] for x, _ in minibatches]

            # 切分每个 domain 的 ce 值
            ce_values_2d = torch.split(ce_values, batch_sizes)
            ce_value_domain = torch.tensor([v.mean().item() for v in ce_values_2d]).to(self.device)
            ce_value_sum = torch.sum(ce_value_domain)
            weight_step = 1 + ce_value_domain / ce_value_sum  # shape [num_domains]

            # 为每个 domain 的样本生成对应权重
            self.weight_step = torch.cat([
                weight_step[i].repeat(batch_sizes[i]) for i in range(self.num_domains)
            ]).to(self.device)

        cc_loss = self.cc_loss(feature_vectors, labels, self.weight_step)
        ct_loss = self.ct_loss(logits, labels, self.weight_step)

        cl_loss = torch.mean(self.cl_loss(logits, labels)*self.weight_step)


        total_loss  = cl_loss + self.lbda_cc*cc_loss +  self.lbda_ct*ct_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_cc = cc_loss.detach().cpu().data.numpy()
        loss_ct = ct_loss.detach().cpu().data.numpy()
        loss_cl = cl_loss.detach().cpu().data.numpy()


        losses={}
        losses['cc']=loss_cc
        losses['ct']=loss_ct
        losses['cl']=loss_cl

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)

        loss_acc_result = {'loss_cc': [], 'loss_ct':[], 'loss_cl':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)
            # self.scheduler.step()

            loss_acc_result['loss_cc'].append(losses['cc'])
            loss_acc_result['loss_ct'].append(losses['ct'])
            loss_acc_result['loss_cl'].append(losses['cl'])

            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step==self.steps:
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
            y_pred_labels = []
            y_true = []
            y_pred_probs = []
            for j, batched_data in enumerate(the_loader):
                x, label_fault = batched_data
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)

                with torch.no_grad():
                    _, logits = self.model(x)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                y_pred_labels.extend(preds.cpu().numpy())
                y_true.extend(label_fault.cpu().numpy())
                y_pred_probs.extend(probs.cpu().numpy())

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

    def predict(self,x):
        # print(x.shape)
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
    # 创建跨域训练数据迭代器
    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)



    for i in range(1):
        model = CCN(configs)


        # 执行训练
        # 组合并清理测试加载器，移除所有None值
        all_test_loaders = test_loaders_tgt + test_loaders_src
        valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

        # 执行训练
        loss_acc_result = model.train_model(
            train_minibatches_iterator,
            valid_test_loaders,  # <-- 正确：传递清理后的列表
        )
        loss_acc_result['loss_cc'] = np.array(loss_acc_result['loss_cc'])
        loss_acc_result['loss_ct'] = np.array(loss_acc_result['loss_ct'])
        loss_acc_result['loss_cl'] = np.array(loss_acc_result['loss_cl'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])


        print("===========================================================================================")
        print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

        save_dir = os.path.join('result', 'CCN')  # 只定义目录路径
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

