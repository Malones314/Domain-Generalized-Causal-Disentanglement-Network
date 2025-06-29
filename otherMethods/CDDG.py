# 导入必要的库
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
import torch.nn.functional as F  # 神经网络函数式接口
import torchvision  # 计算机视觉工具库（未直接使用）
from torch.autograd import Variable  # 自动求导变量（旧版PyTorch）
# from torch.autograd import Function
# from torchvision import models
from torchsummary import summary  # 模型结构可视化工具

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
from sklearn.metrics import confusion_matrix  # 混淆矩阵计算
import pickle  # 对象序列化
import matplotlib.pyplot as plt  # 绘图库
import math  # 数学函数
from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score
# 导入自定义模块
from models.Conv1dBlock import Conv1dBlock  # 自定义1D卷积块
from models.Networks import (  # CDDG网络组件
    Encoder_cddg_bearing, Decoder_cddg_bearing, Classifier_cddg_bearing,
    Encoder_cddg_fan, Decoder_cddg_fan, Classifier_cddg_fan
)
from datasets.load_bearing_data import (  # 轴承数据集加载器
    ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA
)
from datasets.load_fan_data import ReadMIMII, ReadScenarioData

# 导入自定义工具类
from utils.DictObj import DictObj  # 字典转对象工具
from utils.AverageMeter import AverageMeter  # 平均值计算器
from utils.CalIndex import cal_index  # 性能指标计算
from utils.SetSeed import set_random_seed
from utils.SimpleLayerNorm import LayerNorm  # 自定义层归一化
from utils.TuneReport import GenReport  # 报告生成器
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset, MultiInfiniteDataLoader  # 自定义数据加载器
from utils.losses import focal_loss

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/CDDG.py

# 加载配置文件
with open(os.path.join(sys.path[0], 'config_files/CDDG_config.yaml'), 'r', encoding='utf-8') as f:
    '''从YAML文件加载配置参数'''
    configs = yaml.load(f, Loader=yaml.FullLoader)  # 加载YAML文件
    print(configs)  # 打印配置参数
    configs = DictObj(configs)  # 将字典转换为对象形式

    # 设置计算设备（GPU/CPU）
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'  # 使用GPU


class CDDG(nn.Module):
    """条件域解耦生成网络（Conditional Domain Disentanglement Generative Network）"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs  # 配置参数
        self.device = torch.device(
            configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")  # 强制设备类型
        self.dataset_type = configs.dataset_type  # 数据集类型（bearing/fan）

        # 网络参数设置
        self.num_classes = configs.num_classes  # 分类类别数
        self.batch_size = configs.batch_size  # 批量大小
        self.steps = configs.steps  # 总训练步数，默认200
        self.checkpoint_freq = configs.checkpoint_freq  # 验证频率 每更新checkpoint_freq个batch,对test dataloader进行推理1次,默认100
        self.lr = configs.lr  # 初始学习率
        self.num_domains = len(configs.datasets_src)  # 源域数量
        # 网络组件初始化后立即移动到设备
        self.encoder_m = Encoder_cddg_fan().to(self.device)
        self.encoder_h = Encoder_cddg_fan().to(self.device)
        self.decoder = Decoder_cddg_fan().to(self.device)
        self.classifer = Classifier_cddg_fan(self.num_classes).to(self.device)
        # 根据数据集类型初始化网络组件
        if self.dataset_type == 'bearing':
            # 轴承数据专用组件
            self.encoder_m = Encoder_cddg_bearing()  # 机器特征编码器
            self.encoder_h = Encoder_cddg_bearing()  # 健康状态编码器
            self.decoder = Decoder_cddg_bearing()    # 信号解码器
            self.classifer = Classifier_cddg_bearing(self.num_classes)  # 分类器
        elif self.dataset_type == 'fan':
            # 风扇数据专用组件
            self.encoder_m = Encoder_cddg_fan()
            self.encoder_h = Encoder_cddg_fan()
            self.decoder = Decoder_cddg_fan()
            self.classifer = Classifier_cddg_fan(self.num_classes)

        # 优化器设置（联合优化所有组件）
        self.optimizer = torch.optim.Adam(
            list(self.encoder_m.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifer.parameters()),
            lr=self.lr
        )

        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1          # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1

        # 损失函数权重参数
        self.w_rc = configs.w_rc  # 重构损失权重
        self.w_rr = configs.w_rr  # 冗余减少损失权重
        self.w_ca = configs.w_ca  # 因果聚合损失权重

        # 域权重相关参数
        self.weight_step = None  # 动态域权重
        self.use_domain_weight = configs.use_domain_weight  # 是否使用动态域权重

        # 学习率调度参数
        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler
        self.gamma = configs.gamma  # 学习率衰减系数

        self.grad_norms = AverageMeter()
        self.grad_clip = configs.grad_clip

    def forward_penul_fv(self, x):
        """获取倒数第二层健康特征向量（用于可视化）"""
        _, fh_vec = self.encoder_h(x)  # 编码健康特征(设备是否处于健康状态)
        fv = self.classifer.forward1(fh_vec)  # 通过分类器的中间层
        return fv

    def forward_zd_fv(self, x):
        """获取机器域特征向量"""
        _, fm_vec = self.encoder_m(x)  # 编码机器特征  fh_vec:(B,D)
        return fm_vec


    def adjust_learning_rate(self, step):
        """动态调整学习率（支持余弦退火和阶梯式衰减）"""
        """
        Decay the learning rate based on schedule
        https://github.com/facebookresearch/moco/blob/main/main_moco.py
        """
        lr = self.lr
        if self.configs.cos:  # 余弦退火策略  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:  # stepwise lr schedule 阶梯式衰减策略
            for milestone in self.configs.schedule:
                lr *= self.gamma if step >= milestone else 1.0
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            # print(lr)
            param_group["lr"] = lr

    def cal_reconstruction_loss(self, x, x_rec):
        """计算信号重构损失（MSE），裁剪输入与重构信号至相同长度后计算"""
        # 假设 x 的形状为 (B, C, L1)，x_rec 的形状为 (B, C, L2)
        # 使用较小的长度计算损失
        L = min(x.shape[2], x_rec.shape[2])
        x_cropped = x[:, :, :L]
        x_rec_cropped = x_rec[:, :, :L]
        return (x_rec_cropped - x_cropped).pow(2).mean()

    def cal_reduce_redundancy_loss(self, fm_vec, fh_vec):
        """计算特征冗余减少损失"""
        '''
        zz = torch.load('fm_fh_tensor.pt',map_location=torch.device('cpu') )
        fm_vec = zz[0]
        fh_vec = zz[1]
        '''
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]
        # debug
        # torch.save([fm_vec, fh_vec],'fm_fh_tensor.pt')
        #注意这里，原来是在dim=1上标准化，这个是错误的，这里一列才是一个vector，所以应该是dim=0标准化

        # 特征归一化（按特征维度）
        # fm_vec = F.normalize(fm_vec, p=2, dim=0) #(B,D) 机器特征归一化，L2范数，按列归一化
        # fh_vec = F.normalize(fh_vec, p=2, dim=0) #(B,D) 健康特征归一化
        fm_vec = F.normalize(fm_vec, p=2, dim=1) #(B,D) 按样本维度归一化
        fh_vec = F.normalize(fh_vec, p=2, dim=1) #(B,D) 按样本维度归一化

        # 计算自相似矩阵
        sim_fm_vec = torch.matmul(fm_vec.T, fm_vec) #(D,D) 机器特征相似矩阵
        sim_fh_vec = torch.matmul(fh_vec.T, fh_vec) #(D,D) 健康特征相似矩阵
        # 经过normalize之后，上边两个矩阵的对角线本身就是1（不同于Barlow Twins, 这里是两个相同向量的内积）

        E = torch.eye(D).to(self.device) #单位矩阵
        denominator = torch.sum(1 - E) + 1e-8  # 防止除零

        # 计算冗余损失
        loss_fm =  ((1-E)*sim_fm_vec).pow(2).sum()/denominator  # 机器特征冗余
        loss_fh =  ((1-E)*sim_fh_vec).pow(2).sum()/denominator  # 健康特征冗余
        loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()  # 跨特征冗余
        # loss = loss_fmh

        loss = loss_fm + loss_fh + loss_fmh
        # loss =   loss_fm + loss_fh

        return loss

    def cal_causal_aggregation_loss(self, fm_vec, fh_vec, labels, domain_labels):
        """改进后的因果聚合损失函数"""
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        # 添加数值稳定性处理
        fm_vec = F.normalize(fm_vec, p=2, dim=1, eps=1e-8)
        fh_vec = F.normalize(fh_vec, p=2, dim=1, eps=1e-8)

        # 健康状态对比
        labels = labels.contiguous().view(-1, 1)
        mask_fh = torch.eq(labels, labels.T).float().to(self.device)
        sim_fh = torch.mm(fh_vec, fh_vec.t()) / D

        # 分母保护
        pos_count = torch.sum(mask_fh) + 1e-8
        neg_count = torch.sum(1 - mask_fh) + 1e-8
        loss_fh = -(mask_fh * sim_fh).sum() / pos_count + ((1 - mask_fh) * sim_fh).sum() / neg_count

        # 机器域对比
        domain_labels = domain_labels.contiguous().view(-1, 1)
        mask_fm = torch.eq(domain_labels, domain_labels.T).float().to(self.device)
        sim_fm = torch.mm(fm_vec, fm_vec.t()) / D

        # 分母保护
        pos_count_d = torch.sum(mask_fm) + 1e-8
        neg_count_d = torch.sum(1 - mask_fm) + 1e-8
        loss_fm = -(mask_fm * sim_fm).sum() / pos_count_d + ((1 - mask_fm) * sim_fm).sum() / neg_count_d

        # 数值截断
        total_loss = torch.clamp(loss_fh + loss_fm, -1e3, 1e3)
        return total_loss

    def update(self, minibatches):
        """改进的更新方法，支持类别加权的交叉熵损失"""

        xs, ys, domain_labels = [], [], []

        # 确保所有数据移动到模型所在设备
        for domain_idx, (x, y) in enumerate(minibatches):
            x = x.to(self.device)
            y = y.to(self.device)
            xs.append(x)
            ys.append(y)
            domain_labels.append(torch.full((x.size(0),), domain_idx, device=self.device))

        x = torch.cat(xs)
        y = torch.cat(ys)
        domain_labels = torch.cat(domain_labels)

        # 前向计算
        output = self.forward(x, y, domain_labels)

        # ----------------------------------------
        # 类别加权 CrossEntropyLoss
        with torch.no_grad():
            class_sample_counts = torch.bincount(y, minlength=self.num_classes).float()
            class_weights = 1.0 / (class_sample_counts + 1e-8)
            class_weights = class_weights / class_weights.sum()  # 归一化
            class_weights = class_weights.to(self.device)


        logits = self.classifer(output['fh_vec'])
        loss_cl = focal_loss(logits, y, alpha=0.5, gamma=2.0, reduction='none')

        # ----------------------------------------
        # 如果使用 domain 权重，再加一层权重
        if self.use_domain_weight and self.weight_step is not None:
            loss_cl = (loss_cl * self.weight_step).mean()
        else:
            loss_cl = loss_cl.mean()

        # 组合损失
        loss = self.w_rc * output['loss_rc'] + \
               self.w_rr * output['loss_rr'] + \
               self.w_ca * output['loss_ca'] + \
               loss_cl

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        # 返回 detach 后的损失项
        losses = {
            'rc': output['loss_rc'].detach().cpu().item(),
            'rr': output['loss_rr'].detach().cpu().item(),
            'ca': output['loss_ca'].detach().cpu().item(),
            'cl': loss_cl.detach().cpu().item()
        }
        return losses

    # 修改后的 forward 方法（添加 domain_labels 参数）
    def forward(self, x, labels, domain_labels=None):
        """前向传播过程"""
        output = {}
        B = x.shape[0]  # 总批次大小

        if domain_labels is None:
            domain_labels = np.repeat(np.array(list(range(self.num_domains))), self.batch_size)
            domain_labels = torch.from_numpy(domain_labels).type(torch.int64).to(self.device)

        # 双编码器特征提取
        fm_map, fm_vec = self.encoder_m(x)  # 机器特征
        fh_map, fh_vec = self.encoder_h(x)  # 健康特征

        # 特征融合与信号重构
        fmh_map = torch.cat([fm_map, fh_map], dim=1)
        x_rec = self.decoder(fmh_map)

        # 健康状态分类
        logits = self.classifer(fh_vec)

        # 计算各项损失
        loss_rc = self.cal_reconstruction_loss(x, x_rec)
        loss_rr = self.cal_reduce_redundancy_loss(fm_vec, fh_vec)
        loss_ca = self.cal_causal_aggregation_loss(fm_vec, fh_vec, labels, domain_labels)
        loss_cl = F.cross_entropy(logits, labels)

        # 动态域权重计算：根据每个域的平均交叉熵损失生成权重，直接利用 domain_labels 索引
        if self.use_domain_weight:
            ce_values = F.cross_entropy(logits, labels, reduction='none')
            # 计算每个域的平均损失
            weight_list = []
            for d in range(self.num_domains):
                mask = (domain_labels == d)
                if mask.sum() > 0:
                    avg_loss = ce_values[mask].mean()
                else:
                    avg_loss = torch.tensor(0.0, device=self.device)
                weight_list.append(avg_loss)
            weight_step = torch.stack(weight_list)  # shape: (num_domains,)
            ce_value_sum = weight_step.sum() + 1e-8  # 防止除零
            weight_step = 1 + weight_step / ce_value_sum
            # 为每个样本分配其所属域的权重
            self.weight_step = weight_step[domain_labels]
        else:
            self.weight_step = torch.ones(B).to(self.device)

        # 加权分类损失
        loss_cl = torch.mean(F.cross_entropy(logits, labels, reduction='none') * self.weight_step)

        output.update({
            'loss_rc': loss_rc,
            'loss_rr': loss_rr,
            'loss_ca': loss_ca,
            'loss_cl': loss_cl,
            'fh_vec': fh_vec  # 添加这一项
        })
        return output


    def train_model(self, train_minibatches_iterator, test_loaders):
        """模型训练流程"""
        self.to(self.device)  # 移至指定设备
        print("train_model begin")
        # 初始化记录容器
        # loss_acc_result = {'loss_cc': [], 'loss_ct':[], 'loss_cl':[], 'acces':[]}
        loss_acc_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl':[], 'acces':[]}

        # 训练循环
        for step in range(1, self.steps+1):
            self.train()  # 训练模式
            self.current_step = step
            print(step)
            # 获取数据并更新参数
            minibatches_device = next(train_minibatches_iterator)

            losses = self.update(minibatches_device)
            # self.scheduler.step()
            # print("losses = self.update(minibatches_device)")

            # 学习率调整
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            # 记录损失
            loss_acc_result['loss_rc'].append(losses['rc'])
            loss_acc_result['loss_rr'].append(losses['rr'])
            loss_acc_result['loss_ca'].append(losses['ca'])
            loss_acc_result['loss_cl'].append(losses['cl'])


            # 显示train_accuracy和test_accuracy 定期验证
            if step % self.checkpoint_freq == 0 or step == self.steps:
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
                # 梯度范数监控
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

        return loss_acc_result
    def test_model(self, loaders):
        self.eval()
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []
        with torch.no_grad():
            for loader in loaders:
                y_pred_lst = []
                y_prob_lst = []
                y_true_lst = []
                for x, label_fault in loader:
                    x = x.to(self.device)
                    label_fault = label_fault.to(self.device)
                    y_logits = self.classifer(self.encoder_h(x)[1])
                    y_probs = torch.softmax(y_logits, dim=1)  # 保持二维结构 (batch, 2)
                    y_preds = torch.argmax(y_logits, dim=1)

                    y_prob_lst.append(y_probs.detach().cpu().numpy())  # 形状 (batch, 2)
                    y_pred_lst.extend(y_preds.detach().cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                # 修改3：合并概率数组
                y_true = np.array(y_true_lst)
                y_pred = np.array(y_pred_lst)
                y_prob = np.vstack(y_prob_lst)  # 形状变为 (n_samples, 2)
                acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred, y_prob)
                acc_results.append(acc_i)
                auc_results.append(auc_i)
                prec_results.append(prec_i)
                recall_result.append(recall_i)
                f1_results.append(f1_i)
        self.train()

        return acc_results, auc_results, prec_results, recall_result, f1_results

    def predict(self, x):
        '''
        预测样本的标签
        '''
        _, fh_vec= self.encoder_h(x)  # 提取健康特征
        y_pred = self.classifer(fh_vec)  # 返回预测类别

        return torch.max(y_pred, dim=1)[1]


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
        if scenario == 's1':
            datasets_list = ['id_06', 'id_00', 'id_02', 'id_04']
        elif scenario == 's2':
            datasets_list = ['id_04', 'id_00', 'id_02', 'id_06']
        elif scenario == 's3':
            datasets_list = ['id_02', 'id_00', 'id_04', 'id_06']
        elif scenario == 's4':
            datasets_list = ['id_00', 'id_02', 'id_04', 'id_06']

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

    # 创建日志目录

    # 创建报告目录
    full_path_rep = os.path.join('Output//CDDG//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)

    # 初始化日志记录器
    currtime = str(time.time())[:10]  # 用时间戳创建唯一文件名


    # 训练循环（支持多次随机种子实验）
    for i in range(1):  # 通常这里可以改为多个循环进行多次实验
        model = CDDG(configs)  # 初始化模型

        # 组合并清理测试加载器，移除所有None值
        all_test_loaders = test_loaders_tgt + test_loaders_src
        valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

        # 模型训练
        loss_acc_result = model.train_model(
            train_minibatches_iterator,
            valid_test_loaders  # <-- 传递清理后的列表
        )

        # 结果处理
        loss_acc_result = {
            'loss_rc': np.array(loss_acc_result['loss_rc']),
            'loss_rr': np.array(loss_acc_result['loss_rr']),
            'loss_ca': np.array(loss_acc_result['loss_ca']),
            'loss_cl': np.array(loss_acc_result['loss_cl']),
            'acces': np.array(loss_acc_result['acces']),
            'auc': np.array(loss_acc_result['auc']),  # ← 新增
        }



        print("===========================================================================================")
        print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

        save_dir = os.path.join('result', 'CDDG')  # 只定义目录路径
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
