# 导入必要的库
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
import torch.nn.functional as F  # 神经网络函数式接口
import torchvision  # 计算机视觉相关工具（未直接使用）
from torch.autograd import Variable  # 自动求导变量（旧版PyTorch）
from torchsummary import summary  # 模型结构可视化工具
# from torch.autograd import Function
# from torchvision import models

# 导入科学计算和数据处理库
import numpy as np  # 数值计算库
import yaml  # YAML配置文件解析
import itertools  # 迭代工具
import copy  # 对象复制
import random  # 随机数生成
import os  # 操作系统接口
import time  # 时间相关功能
import sys  # 系统相关参数和函数
from builtins import object
import scipy.io as sio  # MATLAB文件读写
from scipy.interpolate import interp1d  # 一维插值
from scipy.signal import butter, lfilter, freqz, hilbert  # 信号处理工具
import pickle  # 对象序列化
import matplotlib.pyplot as plt  # 绘图库
import math  # 数学函数

# 导入自定义模块
from models.Conv1dBlock import Conv1dBlock  # 自定义1D卷积块
from models.Networks import Network_bearing, Network_fan  # 轴承和风扇网络结构
from datasets.load_bearing_data import ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA  # 轴承数据集加载器
from datasets.load_fan_data import ReadMIMII, ReadScenarioData

# self-made utils
# 导入自定义工具类
from utils.DictObj import DictObj  # 字典转对象工具
from utils.AverageMeter import AverageMeter  # 平均值计算器
from utils.CalIndex import cal_index  # 性能指标计算
# from utils.SetSeed import set_random_seed
from utils.SetSeed import set_random_seed
from utils.SimpleLayerNorm import LayerNorm  # 自定义层归一化
from utils.TuneReport import GenReport  # 报告生成器
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset, MultiInfiniteDataLoader  # 自定义数据加载器
# from utils.SignalTransforms import AddGaussianNoise, RandomScale, MakeNoise, Translation
# from utils.LMMD import LMMDLoss
from utils.GradientReserve import grad_reverse  # 梯度反转层（未直接使用）

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/DDGFD.py

# 加载配置文件
with open(os.path.join(sys.path[0], 'config_files/DDGFD_config.yaml')) as f:
    '''从YAML文件加载配置参数'''
    '''
    link: https://zetcode.com/python/yaml/#:~:text=YAML%20natively%20supports%20three%20basic,for%20YAML%3A%20PyYAML%20and%20ruamel.
    '''
    configs = yaml.load(f, Loader=yaml.FullLoader)  # 加载YAML配置
    print(configs)  # 打印原始配置
    configs = DictObj(configs)  # 转换为对象形式

    # 设置计算设备（GPU/CPU）
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'  # 使用GPU



class InstanceDiscriLoss(nn.Module):
    """实例判别损失（Instance Discrimination Loss）
       测试代码：
       x = torch.randn((5,128))
       y = torch.tensor([1,2,0,1,2])
       the_id_loss = InstanceDiscriLoss(configs)
       print(the_id_loss(x,y))
       """

    '''
    Instance-based discriminative loss
    test code:
    x = torch.randn((5,128))
    y = torch.tensor([1,2,0,1,2])
    the_id_loss = InstanceDiscriLoss(configs)
    print(the_id_loss(x,y))
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device  # 计算设备
        self.m0 = configs.m0  # 类内距离阈值
        self.m1 = configs.m1  # 类间距离阈值

    def forward(self, x, labels):
        """前向传播计算损失"""
        # print('x:', x)
        x = F.normalize(x, dim=1)  # 特征归一化（L2归一化）
        x1 = torch.unsqueeze(x, 0)  # 扩展维度 (1,B,D)
        x2 = torch.unsqueeze(x, 1)  # 扩展维度 (B,1,D)
        labels = labels.contiguous().view(-1, 1)  # 调整标签形状 (B,1)

        # 计算欧氏距离矩阵 (B,B)
        dx = torch.pow(torch.sum(torch.pow(x2-x1, 2), dim=2)+1e-9,0.5)

        # 生成同类掩码矩阵
        mask = torch.eq(labels, labels.T).float().to(self.device)  # (B,B)
        mask_opposite = 1 - mask  # 异类样本掩码

        # print('mask:', mask)

        # 初始化零矩阵用于损失计算
        zero_matrix = torch.zeros_like(dx).to(self.device)

        # 计算阈值化距离
        m0_dx = dx - self.m0  # 类内距离与阈值差
        m1_dx = self.m1 - dx  # 类间阈值与距离差

        # 计算样本数量（防止除零）
        n1 = torch.sum(mask)
        n1 = 1 if n1 == 0 else n1
        n2 = torch.sum(mask_opposite)
        n2 = 1 if n2 == 0 else n2

        # 计算类内损失（拉近同类样本）
        loss1 = torch.sum(mask*torch.max(m0_dx, zero_matrix))/n1
        # 计算类间损失（推开异类样本）
        loss2 = torch.sum(mask_opposite*torch.max(m1_dx, zero_matrix))/n2

        loss = loss1+loss2

        return loss



class DDGFD(nn.Module):
    """深度域泛化故障诊断网络（Deep Domain Generalized Fault Diagnosis）"""
    def __init__(self, configs):
        super().__init__()
        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1          # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type # 数据类型（bearing/fan）

        # 训练参数设置
        self.checkpoint_freq = configs.checkpoint_freq  # 验证频率
        self.steps = configs.steps  # 总训练步数
        self.lr = configs.lr  # 初始学习率
        self.batch_size = configs.batch_size  # 批量大小

        # 实例判别损失模块
        self.instance_discri_loss = InstanceDiscriLoss(configs).to(self.device)

        # 根据数据类型选择网络结构
        if self.dataset_type == 'bearing':
            self.model = Network_bearing(configs).to(self.device)  # 轴承诊断网络
        elif self.dataset_type == 'fan':
            self.model = Network_fan(configs).to(self.device)  # 风扇诊断网络
        else:
            raise ValueError("数据集类型应为轴承(bearing)或风扇(fan)!")

        # 优化器设置（随机梯度下降）
        self.optimizer = torch.optim.SGD(list(self.model.parameters()), lr=self.lr)

    def update(self, minibatches):
        """参数更新步骤"""
        # 合并多源域数据
        x = torch.cat([x for x, y in minibatches]) # 特征数据 the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        labels = torch.cat([y for x, y in minibatches])# 标签数据
        x      = x.to(self.device)
        labels = labels.to(self.device)

        # 前向传播
        fv, logits = self.model(x)  # 获取特征向量和分类logits

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, labels)

        assert not torch.any(torch.isnan(fv))
        # 计算实例判别损失
        id_loss = self.instance_discri_loss(fv, labels)

        # 总损失（联合优化）
        loss = ce_loss + id_loss

        # 反向传播与优化
        self.optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()
        self.optimizer.step()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = ce_loss.detach().cpu().data.numpy()
        losses['loss_id'] = id_loss.detach().cpu().data.numpy()

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders):
        """模型训练流程"""
        self.to(self.device)  # 确保模型在正确设备上

        # 初始化损失记录容器
        loss_acc_result = {'loss_total': [], 'loss_ce':[], 'loss_id':[], 'acces':[]}

        # 训练循环
        for step in range(1, self.steps+1):
            print(step)
            self.train()  # 训练模式
            self.current_step = step

            # 获取数据并更新参数
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            # 记录损失值
            loss_acc_result['loss_total'].append(losses['loss_total'])
            loss_acc_result['loss_ce'].append(losses['loss_ce'])
            loss_acc_result['loss_id'].append(losses['loss_id'])


            # 定期验证  显示train_accuracy和test_accuracy
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

    def predict(self, x):
        """执行预测"""
        with torch.no_grad():
            _, logits = self.model(x)  # 获取分类logits
            return torch.max(logits, dim=1)[1]  # 返回预测类别


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

    # 训练模型
    for i in range(1):  # 可扩展为多次随机种子实验
        model = DDGFD(configs)

        # 执行训练
        all_test_loaders = test_loaders_tgt + test_loaders_src
        valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

        # 执行训练
        loss_acc_result = model.train_model(
            train_minibatches_iterator,
            valid_test_loaders,  # <-- 正确：传递清理后的列表
        )
        # 处理结果数据
        loss_acc_result['loss_total'] = np.array(loss_acc_result['loss_total'])
        loss_acc_result['loss_ce'] = np.array(loss_acc_result['loss_ce'])
        loss_acc_result['loss_id'] = np.array(loss_acc_result['loss_id'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])

        currtime = str(time.time())[:10]

        print("===========================================================================================")
        print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

        save_dir = os.path.join('result', 'DDGFD')  # 只定义目录路径
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
    # section_s = 's1','s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section_s = 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section = '00', '01', '02'
    sectionNumber = 3
    sectionsNumber = 12
    for i in range(sectionsNumber):
        run_times = 10
        configs.fan_section = section_s[i]
        for _ in range(run_times):
            print('----------------------------------------------------------', _)
            main( 0, configs)
