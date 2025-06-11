###########################################################################
# 在训练嵌入模型时采用平衡样本权重
# 将子聚类AdaCos损失函数替换为AdaProj损失函数
# 面对测试集和训练集中故障和正常数据分配不均，故障数据少，正常数据多。重新设计加权损失函数，添加dropout、CBAM注意力机制增强
#
# 20250425
# 读取不同域的数据时，按照数量赋予不同权重（load_DGCDN_data）
# 根据不同域数量调整safe_batch_size（load_DGCDN_data.py）
# 20250426
# 增加保存训练模型情况
# 固定并保存随机种子
# 20250430
# 自动计算权重，改进权重计算
# 20250501
# 使用自定义预测阈值
# 20250502
# 设置entropy_loss_weight（最少熵损失项，增强模型信心）优化损失函数
#########################################################################

# 导入必要的库
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络模块

import torch.nn.functional as F  # 神经网络函数式接口

# 导入科学计算和数据处理库
import numpy as np  # 数值计算库
import yaml  # YAML配置文件解析

import os  # 操作系统接口
import sys  # 系统相关参数和函数

# 导入自定义模块
from models.Networks import (  # DGCDN网络组件
    Encoder_DGCDN, Decoder_DGCDN, Classifier_DGCDN
)

# 导入自定义工具类
from utils.DictObj import DictObj  # 字典转对象工具
from utils.AverageMeter import AverageMeter  # 平均值计算器
from utils.CalIndex import cal_index  # 性能指标计算

# 加载配置文件
with open(os.path.join(sys.path[0], 'config_files/DGCDN.yaml'), 'r', encoding='utf-8') as f:
    '''从YAML文件加载配置参数'''
    configs = yaml.load(f, Loader=yaml.FullLoader)  # 加载YAML文件
    # print(configs)  # 打印配置参数
    configs = DictObj(configs)  # 将字典转换为对象形式

    # 设置计算设备（GPU/CPU）
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'  # 使用GPU

class DGCDN(nn.Module):
    """条件域解耦生成网络（Conditional Domain Disentanglement Generative Network）"""
    def __init__(self, configs, seed, class_weights=None):
        super().__init__()
        self.current_step = None
        self.model_version = "v_20250426"
        self.configs = configs  # 配置参数
        self.device = torch.device(
            configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")  # 强制设备类型
        self.dataset_type = configs.dataset_type  # 数据集类型（bearing/fan）
        self.seed = seed
        self.configs.seed = seed
        self.eps = configs.eps
        self.schedule = configs.schedule

        # 网络参数设置
        self.num_classes = configs.num_classes  # 分类类别数
        self.batch_size = configs.batch_size  # 批量大小
        self.steps = configs.steps  # 总训练步数
        self.checkpoint_freq = configs.checkpoint_freq  # 验证频率 每更新checkpoint_freq个batch,对test dataloader进行推理1次,默认100
        self.lr = configs.lr  # 初始学习率
        self.num_domains = len(configs.datasets_src)  # 源域数量
        # 网络组件初始化后立即移动到设备
        self.encoder_m = Encoder_DGCDN().to(self.device)
        self.encoder_h = Encoder_DGCDN().to(self.device)
        self.decoder = Decoder_DGCDN().to(self.device)
        self.classifer = Classifier_DGCDN(self.num_classes).to(self.device)

        self.use_entropy_loss = configs.use_entropy_loss
        self.entropy_loss_weight = configs.entropy_loss_weight

        # 增加 Dropout 层，为避免过拟合，将在分类前对健康特征进行 dropout
        self.dropout = nn.Dropout(p=configs.dropout)

        # 注意力机制超参数
        self.cbam_reduction = configs.cbam_reduction
        self.cbam_kernel_size = configs.cbam_kernel_size
        self.use_residual = configs.use_residual
########################################################################################################################

        # CBAM 注意力：根据分类器第一个 Linear 层自动推断特征维度，避免 Encoder dummy 前向
        channels = None
        for m in self.classifer.modules():
            if isinstance(m, nn.Linear):
                channels = m.in_features
                break
        if channels is None:
            raise ValueError("无法从 Classifier 中推断特征维度")
########################################################################################################################
        # 初始化 CBAM 注意力模块
        self.attention = CBAM1D(
            channels=channels,
            reduction=self.cbam_reduction,
            kernel_size=self.cbam_kernel_size,
            use_residual = self.use_residual
        )
########################################################################################################################
        self.focal_loss_gamma = configs.focal_loss_gamma
        # ===== 设置加权 FocalLoss（或切换为 CrossEntropyLoss）=====
        if class_weights is not None:
            self.focal_loss = FocalLoss(gamma=self.focal_loss_gamma, weight=class_weights)
        else:
            self.focal_loss = FocalLoss(gamma=self.focal_loss_gamma)
########################################################################################################################

        # 早停机制
        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1
        self.best_F1_score = -1
        self.early_stop_counter = 0      # 未提升计数
        self.early_stop = configs.early_stop          # 早停标志
        self.best_model_state = None     # 最佳模型状态
        self.early_stopping_patience = configs.early_stopping_patience   # 早停容忍步数
        # 优化器设置（联合优化所有组件）
        self.optimizer = torch.optim.Adam(
            list(self.encoder_m.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifer.parameters()),
            lr=self.lr,
            weight_decay=1e-4  # 加入 L2 正则化
        )

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

    def test_model(self, loaders):
        self.eval()
        freeze_bn_stats(self)
        acc_results = []
        auc_results = []
        f1_results = []

        with torch.no_grad():
            for idx_loader, loader in enumerate(loaders):
                y_pred_lst = []
                y_prob_lst = []  # 改为存储二维概率
                y_true_lst = []

                for x, label_fault in loader:
                    x = x.to(self.device)
                    label_fault = label_fault.to(self.device)

                    # 修改1：获取完整概率矩阵
                    y_logits = self.classifer(self.encoder_h(x)[1])
                    y_probs = torch.softmax(y_logits, dim=1)  # 保持二维结构 (batch, n_classes)

                    # 修改2：使用append代替extend
                    y_prob_lst.append(y_probs.detach().cpu().numpy())  # 形状保持为 (batch, 2)

                    # 保持原有预测逻辑
                    threshold = 0.6
                    y_preds = (y_probs[:, 1] > threshold).long()
                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                # 修改3：合并概率数组
                y_true = np.array(y_true_lst)
                y_pred = np.array(y_pred_lst)
                y_prob = np.vstack(y_prob_lst)  # 形状变为 (n_samples, 2)

                # 修改4：传递二维概率矩阵
                acc_i, auc_i, _, _, f1_i = cal_index(y_true, y_pred, y_prob)

                acc_results.append(acc_i)
                auc_results.append(auc_i)
                f1_results.append(f1_i)
                # print(Counter(y_true_lst))
                # sns.histplot(y_prob_lst, bins=20)
                # plt.title('目标域预测正类概率分布')
                # plt.show()

        self.train()

        # 打印结果
        self.logger.info(f"[Test Result] Accuracy: {acc_results}")
        self.logger.info(f"[Test Result] AUC: {auc_results}")
        self.logger.info(f"[Test Result] F1-SCORE: {f1_results}")

        return acc_results, auc_results, f1_results

    def predict(self, x):
        '''
        预测样本的标签
        '''
        _, fh_vec= self.encoder_h(x)  # 提取健康特征
        fh_vec = self.dropout(fh_vec)  # 保持与前向一致
        y_pred = self.classifer(fh_vec)  # 返回预测类别

        return torch.max(y_pred, dim=1)[1]


########################################################################################################################
def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

########################################################################################################################
class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention1D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.BatchNorm1d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.size()
        avg_pool = torch.mean(x, dim=2)  # (B, C)
        max_pool = torch.max(x, dim=2)[0]  # (B, C)
        avg_out = self.mlp(avg_pool)  # (B, C)
        max_out = self.mlp(max_pool)  # (B, C)
        att = torch.sigmoid(avg_out + max_out).view(B, C, 1)  # (B, C, 1)
        return x * att

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)

        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B,1,L)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B,1,L)
        cat = torch.cat([avg_pool, max_pool], dim=1)  # (B,2,L)
        att = self.sigmoid(self.conv(cat))  # (B,1,L)
        return x * att

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7, use_residual=False):
        super(CBAM1D, self).__init__()
        self.channel_att = ChannelAttention1D(channels, reduction)
        self.spatial_att = SpatialAttention1D(kernel_size)
        self.use_residual = use_residual

    def forward(self, x):
        # x: (B, C) or (B, C, L)
        is_vector = False
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (B, C, 1)
            is_vector = True

        out = self.channel_att(x)
        out = self.spatial_att(out)

        if self.use_residual:
            out = out + x  # 加残差连接（必须维度匹配）

        if is_vector:
            out = out.squeeze(2)  # 回到 (B, C)
        return out
########################################################################################################################
class FocalLoss(nn.Module):
    def __init__(self, gamma, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
