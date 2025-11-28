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
import uuid
from fileinput import filename

# 导入必要的库
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络模块
from sklearn.utils import compute_class_weight
import torch.nn.functional as F  # 神经网络函数式接口
# 导入科学计算和数据处理库
import numpy as np  # 数值计算库
import yaml  # YAML配置文件解析
import random  # 随机数生成
import os  # 操作系统接口
import time  # 时间相关功能
import sys  # 系统相关参数和函数
import scipy.io as sio  # MATLAB文件读写
import seaborn as sns
import matplotlib.pyplot as plt
import math  # 数学函数

# 导入自定义模块
from models.Networks import (  # DGCDN网络组件
    Encoder_DGCDN, Decoder_DGCDN, Classifier_DGCDN
)

from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData  # 风扇数据集加载器

# 导入自定义工具类
from utils.CalIndex import cal_index  # 性能指标计算
from utils.CreateLogger import create_logger  # 日志创建器
from utils.TuneReport import GenReport  # 报告生成器
from utils.DatasetClass import MultiInfiniteDataLoader
import csv
from sklearn.model_selection import ParameterGrid
from torch.cuda.amp import GradScaler, autocast

import types

def to_namespace(d: dict):
    """
    递归地将字典及其嵌套的字典转换为 types.SimpleNamespace。
    """
    if not isinstance(d, dict):
        return d

    # 遍历字典，递归地转换嵌套的字典或列表中的字典
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = to_namespace(val)
        elif isinstance(val, (list, tuple)):
            # 同样处理列表/元组中的字典
            d[key] = [to_namespace(x) if isinstance(x, dict) else x for x in val]

    return types.SimpleNamespace(**d)

# 加载配置文件
# (请确保 to_namespace 函数已经定义在某处)
with open(os.path.join(sys.path[0], 'config_files/DGCDN.yaml'), 'r', encoding='utf-8') as f:
    '''从YAML文件加载配置参数'''
    configs_dict = yaml.load(f, Loader=yaml.FullLoader)  # 加载YAML文件，得到一个字典
    # print(configs_dict)
    configs = to_namespace(configs_dict)  # <--- 使用辅助函数转换为 SimpleNamespace

    # 设置计算设备（GPU/CPU），后面的代码完全不需要改变！
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'  # 使用GPU


class DGCDN(nn.Module):
    """条件域解耦生成网络（Conditional Domain Disentanglement Generative Network）"""

    def __init__(self, configs, seed, class_weights=None):
        super().__init__()
        self.current_step = None
        self.model_version = "v_20250602"
        self.configs = configs  # 配置参数
        self.device = torch.device(
            configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")  # 强制设备类型
        self.dataset_type = configs.dataset_type  # 数据集类型（bearing/fan）
        self.seed = seed
        self.configs.seed = seed
        self.eps = configs.eps
        self.schedule = configs.schedule

################################123123#####################################
        # self.t_sne = configs.t_sne
        self.t_sne = True

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
        ############123123################
        # self.use_attention = configs.use_attention
        self.use_attention = True
        ########################################################################################################################

        # CBAM 注意力：根据分类器第一个 Linear 层自动推断特征维度，避免 Encoder dummy 前向
        channels = None
        for m in self.classifer.modules():
            if isinstance(m, nn.Linear):
                channels = m.in_features
                break
        if channels is None:
            raise ValueError("无法从 Classifier 中推断特征维度")

        # #####################################123123###################################################################################
        if self.use_attention:
            self.attention = CBAM1D(
                channels=channels,
                reduction=self.cbam_reduction,
                kernel_size=self.cbam_kernel_size,
                use_residual=self.use_residual
            )
            # print("注意力机制 (CBAM) 已启用。")
        else:
            self.attention = None  # 显式设置为空
            # print("注意力机制 (CBAM) 已禁用。")

        # 初始化 CBAM 注意力模块
        # #######################################################################################################################
        # 根据配置参数条件性地初始化 CBAM 注意力模块

        # if self.configs.use_attention:
        #     self.attention = CBAM1D(
        #         channels=channels,
        #         reduction=self.cbam_reduction,
        #         kernel_size=self.cbam_kernel_size,
        #         use_residual=self.use_residual
        #     )
        #     print("注意力机制 (CBAM) 已启用。")
        # else:
        #     self.attention = None  # 显式设置为空
        #     print("注意力机制 (CBAM) 已禁用。")

        #######################################################################################################################
        self.focal_loss_gamma = configs.focal_loss_gamma
        # ===== 设置加权 FocalLoss（或切换为 CrossEntropyLoss）=====
        if class_weights is not None:
            self.focal_loss = FocalLoss(gamma=self.focal_loss_gamma, weight=class_weights)
        else:
            self.focal_loss = FocalLoss(gamma=self.focal_loss_gamma)
        ########################################################################################################################

        # 早停机制
        self.best_auc = -1          # 最佳验证指标
        self.best_acc = -1          # 最佳成功率
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        self.early_stop_counter = 0  # 未提升计数
        self.early_stop = configs.early_stop  # 早停标志
        self.best_model_path = None  # 最佳模型状态
        self.early_stopping_patience = configs.early_stopping_patience  # 早停容忍步数
        # 优化器设置（联合优化所有组件）
        self.optimizer = torch.optim.Adam(
            list(self.encoder_m.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifer.parameters()),
            lr=self.lr,
            weight_decay=1e-4  # 加入 L2 正则化
        )

        # 仅在 CUDA 可用时启用 GradScaler
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

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
        lr = self.lr
        if not self.configs.cos:
            m = int(self.configs.schedule)
            if step % m == 0:
                lr *= self.gamma
        else:
            lr *= 0.5 * (1 + math.cos(math.pi * step / self.steps))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

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

        # 特征归一化
        fm_vec = F.normalize(fm_vec, p=2, dim=1)  # (B,D) 按样本维度归一化
        fh_vec = F.normalize(fh_vec, p=2, dim=1)  # (B,D) 按样本维度归一化

        # 计算自相似矩阵
        sim_fm_vec = torch.matmul(fm_vec.T, fm_vec)  # (D,D) 机器特征相似矩阵
        sim_fh_vec = torch.matmul(fh_vec.T, fh_vec)  # (D,D) 健康特征相似矩阵
        # 经过normalize之后，上边两个矩阵的对角线本身就是1（不同于Barlow Twins, 这里是两个相同向量的内积）

        E = torch.eye(D).to(self.device)  # 单位矩阵
        denominator = torch.sum(1 - E) + float(self.eps)  # 防止除零

        # 计算冗余损失
        loss_fm = ((1 - E) * sim_fm_vec).pow(2).sum() / denominator  # 机器特征冗余
        loss_fh = ((1 - E) * sim_fh_vec).pow(2).sum() / denominator  # 健康特征冗余
        loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()  # 跨特征冗余

        loss = loss_fm + loss_fh + loss_fmh

        return loss

    def cal_causal_aggregation_loss(self, fm_vec, fh_vec, labels, domain_labels):
        """改进后的因果聚合损失函数"""
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        # 添加数值稳定性处理
        fm_vec = F.normalize(fm_vec, p=2, dim=1, eps=float(self.eps))
        fh_vec = F.normalize(fh_vec, p=2, dim=1, eps=float(self.eps))

        # 健康状态对比
        labels = labels.contiguous().view(-1, 1)
        mask_fh = torch.eq(labels, labels.T).float().to(self.device)
        sim_fh = torch.mm(fh_vec, fh_vec.t()) / D

        # 分母保护
        pos_count = torch.sum(mask_fh) + float(self.eps)
        neg_count = torch.sum(1 - mask_fh) + float(self.eps)
        loss_fh = -(mask_fh * sim_fh).sum() / pos_count + ((1 - mask_fh) * sim_fh).sum() / neg_count

        # 机器域对比
        domain_labels = domain_labels.contiguous().view(-1, 1)
        mask_fm = torch.eq(domain_labels, domain_labels.T).float().to(self.device)
        sim_fm = torch.mm(fm_vec, fm_vec.t()) / D

        # 分母保护
        pos_count_d = torch.sum(mask_fm) + float(self.eps)
        neg_count_d = torch.sum(1 - mask_fm) + float(self.eps)
        loss_fm = -(mask_fm * sim_fm).sum() / pos_count_d + ((1 - mask_fm) * sim_fm).sum() / neg_count_d

        # 添加调试代码
        if torch.isnan(loss_fh).any() or torch.isinf(loss_fh).any():
            print(f"!!! loss_fh is NaN or Inf: {loss_fh}")
        if torch.isnan(loss_fm).any() or torch.isinf(loss_fm).any():
            print(f"!!! loss_fm is NaN or Inf: {loss_fm}")

        # 数值截断
        total_loss = torch.clamp(loss_fh + loss_fm, -1e3, 1e3)
        return total_loss


    def update(self, minibatches):
        """改进的更新方法，支持类别加权的交叉熵损失（已集成AMP）"""

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

        self.optimizer.zero_grad()

        # -> 修改：使用 autocast 上下文管理器包裹前向传播和损失计算
        with autocast(enabled=self.use_amp):
            # 前向计算
            output = self.forward(x, y, domain_labels)

            # 组合损失
            if self.use_entropy_loss:
                loss = self.w_rc * output['loss_rc'] + \
                       self.w_rr * output['loss_rr'] + \
                       self.w_ca * output['loss_ca'] + \
                       output['loss_cl'] + \
                       self.entropy_loss_weight * output['loss_entropy']
            else:
                loss = self.w_rc * output['loss_rc'] + \
                       self.w_rr * output['loss_rr'] + \
                       self.w_ca * output['loss_ca'] + \
                       output['loss_cl']

        # -> 修改：使用 GradScaler 进行反向传播和优化器更新
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # 在 unscale 梯度后进行梯度裁剪（可选但推荐）
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:  # 如果不使用 AMP，则执行原始流程
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

        # 返回 detach 后的损失项 (这部分逻辑不变)
        if self.use_entropy_loss:
            losses = {
                'rc': output['loss_rc'].detach().cpu().item(),
                'rr': output['loss_rr'].detach().cpu().item(),
                'ca': output['loss_ca'].detach().cpu().item(),
                'cl': output['loss_cl'].detach().cpu().item(),
                'loss_entropy': output['loss_entropy'].detach().cpu().item()
            }
        else:
            losses = {
                'rc': output['loss_rc'].detach().cpu().item(),
                'rr': output['loss_rr'].detach().cpu().item(),
                'ca': output['loss_ca'].detach().cpu().item(),
                'cl': output['loss_cl'].detach().cpu().item()
            }

        return losses

    # 修改后的 forward 方法（添加 domain_labels 参数）
    def forward(self, x, labels, domain_labels=None):
        """前向传播过程"""
        output = {}
        B = x.shape[0]  # 总批次大小

        # if domain_labels is None:
        #     domain_labels = torch.from_numpy(
        #         np.repeat(np.array(list(range(self.num_domains))), self.batch_size)
        #     ).type(torch.int64).to(self.device)
        ########################################################################################################################
        #
        # 双编码器特征提取
        fm_map, fm_vec = self.encoder_m(x)  # 机器特征
        fh_map, fh_vec = self.encoder_h(x)  # 健康特征

        # 特征融合与信号重构
        fmh_map = torch.cat([fm_map, fh_map], dim=1)
        x_rec = self.decoder(fmh_map)

        # 在分类前对健康特征使用 Dropout 避免过拟合
        fh_vec = self.dropout(fh_vec)
        ########################################################################################################################
        # 如果启用了注意力机制，则应用它
        if self.configs.use_attention and self.attention is not None:
            fh_vec = self.attention(fh_vec)
        ########################################################################################################################

        # 健康状态分类
        logits = self.classifer(fh_vec)

        # 计算各项损失
        loss_rc = self.cal_reconstruction_loss(x, x_rec)
        loss_rr = self.cal_reduce_redundancy_loss(fm_vec, fh_vec)
        loss_ca = self.cal_causal_aggregation_loss(fm_vec, fh_vec, labels, domain_labels)
        if self.use_entropy_loss:
            probs = torch.softmax(logits, dim=1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

        # 动态域权重计算：根据每个域的平均交叉熵损失生成权重，直接利用 domain_labels 索引
        if self.use_domain_weight:
            # focal_loss_fn = FocalLoss(self.focal_loss_gamma, reduction='none')
            # # 计算每个样本的损失
            # ce_values = focal_loss_fn(logits, labels)            # 计算每个域的平均损失

            ce_values = FocalLoss(self.focal_loss_gamma,
                                  weight=self.focal_loss.weight,
                                  reduction='none')(logits, labels)

            weight_list = []
            for d in range(self.num_domains):
                mask = (domain_labels == d)
                if mask.sum() > 0:
                    avg_loss = ce_values[mask].mean()
                else:
                    avg_loss = torch.tensor(0.0, device=self.device)
                weight_list.append(avg_loss)
            weight_step = torch.stack(weight_list)  # shape: (num_domains,)

            ce_value_sum = weight_step.mean() + float(self.eps)  # 使用平均损失作为基准
            weight_step = 1 + (weight_step / ce_value_sum)  # 标准化缩放
            # 为每个样本分配其所属域的权重
            self.weight_step = weight_step[domain_labels]
        else:
            self.weight_step = torch.ones(B, device=self.device)

        # —— 使用带权重的 FocalLoss ——
        loss_cl = torch.mean(
            FocalLoss(self.focal_loss_gamma,
                      weight=self.focal_loss.weight,
                      reduction='none')(logits, labels)
            * self.weight_step
        )
        output.update({
            'loss_rc': loss_rc,
            'loss_rr': loss_rr,
            'loss_ca': loss_ca,
            'loss_cl': loss_cl,
            'fh_vec': fh_vec,
            'loss_entropy': loss_entropy
        })
        return output

    # 模型保存
    def save_checkpoint(self, current_time, step):

        # 创建保存目录
        if self.use_attention:
            save_dir = 'checkpoints\\section' + str(configs.fan_section)
        else:
            save_dir = 'checkpoints\\xiaorongshiyan\\without_attention\\' + 'section' + str(configs.fan_section)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"section{configs.fan_section}_acc{self.best_acc:.4f}_auc{self.best_auc:.4f}pre_{self.best_precision}rec_{self.best_recall}_f1{self.best_F1_score:.4f}_{current_time}.pth"

        filename = os.path.join(save_dir, filename)
        # 保存模型状态
        # 先创建一个包含所有通用参数的字典
        checkpoint_data = {
            'encoder_m': self.encoder_m.state_dict(),
            'encoder_h': self.encoder_h.state_dict(),
            'decoder': self.decoder.state_dict(),
            'classifier': self.classifer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_F1': self.best_F1_score,
            'class_weights': self.focal_loss.weight,
            'seed': self.seed,
            'configs': self.configs
        }

        # 如果注意力机制被使用，则添加其参数到字典中
        if self.configs.use_attention and self.attention is not None:
            checkpoint_data['attention'] = self.attention.state_dict()

        # 保存最终的字典
        torch.save(checkpoint_data, filename)
        print(f"step:{step}, model is located at: {filename}")
        return filename

    def _visualize_tsne_base(self, features_2d, labels, title, save_path, metrics_text=None):
        """
        私有的基础 t-SNE 绘图函数。
        :param features_2d: (np.array) 经过 t-SNE 降维后的2D特征。
        :param labels: (np.array) 用于着色的标签。
        :param title: (str) 图表标题。
        :param save_path: (str) 图片保存路径。
        :param metrics_text: (str, optional) 要在图上显示的额外文本（如聚类指标）。
        """
        plt.figure(figsize=(10, 8))
        palette = sns.color_palette("hsv", len(set(labels)))
        sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1],
                        hue=labels, palette=palette, s=25, alpha=0.8, legend='full')
        plt.title(title, fontsize=22)
        plt.xlabel("t-SNE Dimension 1", fontsize=22)
        plt.ylabel("t-SNE Dimension 2", fontsize=22)
        # 新增代码
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        ###########################123123123#############################################################
        # 将图例放在图外，防止遮挡数据点
        plt.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc='upper left',
                   fontsize=22, title_fontsize=24)

        # 将指标分数显示在图上
        if metrics_text:
            plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=10,
                        bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})

        # 保存文件并关闭图形，避免阻塞和内存泄漏
        if save_path:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"t-SNE plot saved to {save_path}")

        # plt.show() # 在自动化脚本中应避免使用 plt.show()
        plt.close()

    def visualize_tsne(self, loader, feature_type, title, save_path=None):
        """
        t-SNE可视化单个域的特征分布（按类别着色）。
        """
        from sklearn.manifold import TSNE

        self.eval()
        freeze_bn_stats(self)

        features, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                fv = self.encoder_h(x)[1] if feature_type == 'health' else self.encoder_m(x)[1]
                features.append(fv.cpu())
                labels.append(y.cpu())

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        if len(features) < 5:
            self.logger.warning(f"Skipping t-SNE for '{title}': Not enough samples ({len(features)}).")
            return

        # 动态调整 perplexity
        n_samples = features.shape[0]
        perplexity = min(30.0, float(n_samples - 1))

        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
        features_2d = tsne.fit_transform(features)

        self._visualize_tsne_base(features_2d, labels, title, save_path)

    # 在 DGCDN.py 中找到并替换此函数

    def visualize_tsne_mixed_domains(self, loaders, feature_type, title, save_path=None):
        """
        t-SNE可视化多个域的特征分布（按实际域名着色），并计算聚类指标。
        """
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        self.eval()
        freeze_bn_stats(self)

        # --- 1. 获取与加载器顺序一致的域名列表 ---
        # 这个列表将作为我们图例的标签
        domain_names = self.configs.datasets_tgt + self.configs.datasets_src

        features, domain_ids_numeric, domain_ids_str = [], [], []
        with torch.no_grad():
            # --- 2. 遍历加载器，同时记录数字索引和域名字符串 ---
            for domain_idx, loader in enumerate(loaders):
                # 获取当前加载器对应的域名字符串，例如 'id_00'
                domain_name = domain_names[domain_idx]

                for x, _ in loader:
                    x = x.to(self.device)
                    fv = self.encoder_h(x)[1] if feature_type == 'health' else self.encoder_m(x)[1]
                    features.append(fv.cpu())

                    # 数字索引用于后续的聚类指标计算
                    domain_ids_numeric.extend([domain_idx] * fv.size(0))
                    # 域名字符串用于t-SNE图的图例
                    domain_ids_str.extend([domain_name] * fv.size(0))

        features = torch.cat(features).numpy()
        domain_ids_numeric = np.array(domain_ids_numeric)
        # domain_ids_str 是一个字符串列表，例如 ['id_00', 'id_00', ..., 'id_06', 'id_06']

        if len(features) < 5:
            self.logger.warning(f"Skipping t-SNE for '{title}': Not enough samples ({len(features)}).")
            return

        # 动态调整 perplexity
        n_samples = features.shape[0]
        perplexity = min(30.0, float(n_samples - 1))

        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
        features_2d = tsne.fit_transform(features)

        # --- 3. 使用数字索引计算聚类指标 ---
        metrics_text = ""
        try:
            if len(set(domain_ids_numeric)) > 1:
                sil_score = silhouette_score(features, domain_ids_numeric)
                ch_score = calinski_harabasz_score(features, domain_ids_numeric)
                db_score = davies_bouldin_score(features, domain_ids_numeric)
                # print(f"Silhouette: {sil_score:.3f} (higher is better) | "
                #       f"Calinski-Harabasz: {ch_score:.1f} (higher is better) | "
                #       f"Davies-Bouldin: {db_score:.3f} (lower is better)")
                self.logger.info(f"[t-SNE Metrics for '{title}'] {metrics_text}")
        except ValueError as e:
            self.logger.warning(f"Could not calculate t-SNE metrics for '{title}': {e}")

        # --- 4. 将域名字符串列表传递给绘图函数 ---
        # 这样图例就会显示 'id_00', 'id_02' 等实际名称
        self._visualize_tsne_base(features_2d, domain_ids_str, title, save_path, metrics_text)

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        """模型训练流程"""
        self.logger = logger
        self.to(self.device)  # 移至指定设备
        print("train_model begin")
        # 初始化记录容器
        if self.use_entropy_loss:
            all_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl': [], 'loss_entropy': [], 'acces': [],
                          'auc': [],
                          'f1-score': []}
        else:
            all_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl': [], 'acces': [], 'auc': [],
                          'f1-score': []}
        current_time = time.strftime("%Y%m%d_%H%M%S")
        run_id = uuid.uuid4().hex[:8]
        # 训练循环
        for step in range(1, self.steps + 1):
            # print("step:",step)
            self.train()  # 训练模式
            self.current_step = step
            self.logger.info(f"================Step {step}({run_id})================")

            # 获取数据并更新参数
            minibatches_device = next(train_minibatches_iterator)

            # ============ 调试代码（打印每个域当前 batch 的标签分布和数据统计） ============
            # for d, (x, y) in enumerate(minibatches_device):
            # y_cpu = y.cpu().detach()
            # unique, counts = torch.unique(y_cpu, return_counts=True)
            # x_mean = x.mean().item()
            # x_std = x.std().item()

            losses = self.update(minibatches_device)
            all_result['loss_rc'].append(losses['rc'])
            all_result['loss_rr'].append(losses['rr'])
            all_result['loss_ca'].append(losses['ca'])
            all_result['loss_cl'].append(losses['cl'])

            if self.use_entropy_loss:
                all_result['loss_entropy'].append(losses['loss_entropy'])

            # 学习率调整
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            if self.use_entropy_loss:
                self.logger.info(
                    'loss_rc_train: {:.4f} \t loss_rr_train: {:.4f} \t loss_ca_train: {:.4f} \t loss_cl_train: {:.4f} \t loss_entropy_train: {:.4f}'.format(
                        losses['rc'], losses['rr'], losses['ca'], losses['cl'], losses['loss_entropy']))
            else:
                self.logger.info(
                    'loss_rc_train: {:.4f} \t loss_rr_train: {:.4f} \t loss_ca_train: {:.4f} \t loss_cl_train: {:.4f}'.format(
                        losses['rc'], losses['rr'], losses['ca'], losses['cl']))

            # 显示train_accuracy和test_accuracy 定期验证
            if step % self.checkpoint_freq == 0 or step == self.steps or step == 1:
                acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                all_result['acces'].append(acc_results)
                all_result.setdefault('auc', []).append(auc_results)
                # 记录准确率
                self.logger.info("--- Individual Target Domain Results ---")
                for i, domain in enumerate(self.configs.datasets_tgt):
                    self.logger.info(
                        f"  - Domain '{domain}': ACC={acc_results[i]:.4f}, AUC={auc_results[i]:.4f}, F1={f1_results[i]:.4f}")

                print("*"*60)
                print(f"  - step: {step}")

                # 由于测试域可能不止一个，故而我们探求每个测试域上的平均值
                avg_acc = np.mean(acc_results) if acc_results else -1.0
                avg_auc = np.mean(auc_results) if auc_results else -1.0
                avg_prec = np.mean(prec_results) if prec_results else -1.0
                avg_recall = np.mean(recall_result) if recall_result else -1.0
                avg_f1 = np.mean(f1_results) if f1_results else -1.0

                self.logger.info("--- Metrics Across Target Domains ---")
                self.logger.info(f"  - ACC: {avg_acc:.4f}")
                self.logger.info(f"  - AUC: {avg_auc:.4f}")
                self.logger.info(f"  - Precision: {avg_prec:.4f}")
                self.logger.info(f"  - Recall: {avg_recall:.4f}")
                self.logger.info(f"  - F1-Score: {avg_f1:.4f}")

                print(f"  - ACC: {avg_acc:.4f}")
                print(f"  - AUC: {avg_auc:.4f}")
                print(f"  - Precision: {avg_prec:.4f}")
                print(f"  - Recall: {avg_recall:.4f}")
                print(f"  - F1-Score: {avg_f1:.4f}")

                save_model_flag = False  # 用于决定是否保存模型的标志

                # 使用平均AUC判断
                if avg_auc > self.best_auc:
                    self.logger.info(f"New best average AUC found: {avg_auc:.4f} (previously: {self.best_auc:.4f})")
                    self.best_auc = avg_auc
                    save_model_flag = True

                # 使用平均ACC判断
                if avg_acc > self.best_acc:
                    self.logger.info(f"New best average ACC found: {avg_acc:.4f} (previously: {self.best_acc:.4f})")
                    self.best_acc = avg_acc
                    save_model_flag = True

                # 使用平均Precision判断
                if avg_prec > self.best_precision:
                    self.logger.info(
                        f"New best average Precision found: {avg_prec:.4f} (previously: {self.best_precision:.4f})")
                    self.best_precision = avg_prec
                    save_model_flag = True

                # 使用平均Recall判断
                if avg_recall > self.best_recall:
                    self.logger.info(
                        f"New best average Recall found: {avg_recall:.4f} (previously: {self.best_recall:.4f})")
                    self.best_recall = avg_recall
                    save_model_flag = True

                # 使用平均F1分数判断
                if avg_f1 > self.best_F1_score:
                    self.logger.info(
                        f"New best average F1-Score found: {avg_f1:.4f} (previously: {self.best_F1_score:.4f})")
                    self.best_F1_score = avg_f1
                    save_model_flag = True

                # 5. 如果任何一个平均指标达到了历史最佳，则保存模型
                if save_model_flag:
                    print("*" * 60)
                    print(f"New best model found based on AVERAGE metrics (Step: {self.current_step})")
                    print(f"Best AVG ACC: {self.best_acc:.4f}")
                    print(f"Best AVG AUC: {self.best_auc:.4f}")
                    print(f"Best AVG Precision: {self.best_precision:.4f}")  # <-- 新增打印
                    print(f"Best AVG Recall: {self.best_recall:.4f}")  # <-- 新增打印
                    print(f"Best AVG F1-SCORE: {self.best_F1_score:.4f}")
                    print("*" * 60)
                    if self.best_model_path and os.path.exists(self.best_model_path):
                        print(f"Delete old model: {self.best_model_path}")
                        os.remove(self.best_model_path)

                    # 2. 保存新的最佳模型，并获取其路径
                    new_best_path = self.save_checkpoint( current_time, step)

                    # 3. 更新最佳模型的路径记录
                    self.best_model_path = new_best_path


                if self.early_stop:
                    # 早停逻辑判断
                    if save_model_flag:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        self.logger.info(
                            f'ACC not improved ({self.early_stop_counter}/{self.early_stopping_patience})')
                        self.logger.info(
                            f'AUC not improved ({self.early_stop_counter}/{self.early_stopping_patience})')
                        self.logger.info(
                            f'F1-SCORE not improved ({self.early_stop_counter}/{self.early_stopping_patience})')

                        if self.early_stop_counter >= self.early_stopping_patience:
                            self.logger.info("Early stopping triggered!")
                            print("Early stopping triggered!")
                            return all_result
            # 梯度范数监控
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logger.info(f'梯度范数: {total_norm:.4f}')

            # NaN值检测
            for name, param in self.named_parameters():
                if torch.isnan(param).any():
                    logger.error(f"参数 {name} 包含NaN值!")
                    raise ValueError("参数包含NaN值")

        return all_result

    def test_model(self, loaders):
        self.eval()
        freeze_bn_stats(self)
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []

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
                    if str(self.configs.fan_section).startswith('s') is True:
                        threshold = 0.5
                    y_preds = (y_probs[:, 1] > threshold).long()
                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                # 修改3：合并概率数组
                y_true = np.array(y_true_lst)
                y_pred = np.array(y_pred_lst)
                y_prob = np.vstack(y_prob_lst)  # 形状变为 (n_samples, 2)

                # 修改4：传递二维概率矩阵
                acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred, y_prob)

                acc_results.append(acc_i)
                auc_results.append(auc_i)
                prec_results.append(prec_i)
                recall_result.append(recall_i)
                f1_results.append(f1_i)

                # print(Counter(y_true_lst))
                # sns.histplot(y_prob_lst, bins=20)
                # plt.title('目标域预测正类概率分布')
                # plt.show()

        self.train()

        # 打印结果
        self.logger.info(f"[Test Result] Accuracy: {acc_results}")
        self.logger.info(f"[Test Result] AUC: {auc_results}")
        self.logger.info(f"[Test Result] Precision: {prec_results}")
        self.logger.info(f"[Test Result] Recall: {recall_result}")
        self.logger.info(f"[Test Result] F1-SCORE: {f1_results}")

        # t_sne = True
        t_sne = False
        # ==================== BEGIN: t-SNE VISUALIZATION ====================
        if t_sne : # 选择是否进行t-SNE可视化
            # 创建一个本次测试专用的图像保存目录
            plot_dir = os.path.join('Output//myMethod//tsne_plots', f"section{self.configs.fan_section}_seed{self.seed}_step{self.current_step}")
            os.makedirs(plot_dir, exist_ok=True)

            # 1. 可视化目标域 (按类别着色)
            self.logger.info("Generating t-SNE plots for the target domain (colored by class)...")
            target_loader = loaders[0]
            tgt_domain_name = self.configs.datasets_tgt[0]

            self.visualize_tsne(target_loader, feature_type='health',
                                title=f'Health Features on Target "{tgt_domain_name}" (by Class)',
                                save_path=os.path.join(plot_dir, f'target_{tgt_domain_name}_health_by_class.png'))

            self.visualize_tsne(target_loader, feature_type='machine',
                                title=f'Machine Features on Target "{tgt_domain_name}" (by Class)',
                                save_path=os.path.join(plot_dir, f'target_{tgt_domain_name}_machine_by_class.png'))

            # 2. 可视化所有域 (按域着色)
            self.logger.info("Generating t-SNE plots for all domains (colored by domain)...")

            self.visualize_tsne_mixed_domains(loaders, feature_type='health',
                                              title=f'Health Features - All Domains (by Domain ID)',
                                              save_path=os.path.join(plot_dir, 'all_domains_health_by_domain.png'))

            self.visualize_tsne_mixed_domains(loaders, feature_type='machine',
                                              title=f'Machine Features - All Domains (by Domain ID)',
                                              save_path=os.path.join(plot_dir, 'all_domains_machine_by_domain.png'))

        # ===================== END: t-SNE VISUALIZATION =====================

        return acc_results, auc_results, prec_results, recall_result, f1_results
        # return acc_results, auc_results, f1_results

    def predict(self, x):
        '''
        预测样本的标签
        '''
        _, fh_vec = self.encoder_h(x)  # 提取健康特征
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


# ========== 自动计算类别权重函数 ==========
# def compute_class_weights_from_dataloader(loaders, num_classes):
#     """
#     支持多个 DataLoader 列表的自动权重计算（如多源域）
#     """
#     all_labels = []
#     for loader in loaders:
#         for _, labels in loader:
#             all_labels.extend(labels.cpu().numpy())
#
#     weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.arange(num_classes),
#         y=all_labels
#     )
#     return torch.tensor(weights, dtype=torch.float32)
def compute_class_weights_from_dataloader(loaders, num_classes):
    """
    支持多个 DataLoader 列表的自动权重计算（如多源域）。
    新增了对训练集中只存在部分类别的特殊情况的处理。
    """
    all_labels = []
    if not loaders:  # 如果加载器列表为空，返回默认权重
        return torch.ones(num_classes, dtype=torch.float32)

    for loader in loaders:
        # 确保加载器不为空
        if loader is not None and len(loader.dataset) > 0:
            for _, labels in loader:
                all_labels.extend(labels.cpu().numpy())

    # 如果遍历后没有收集到任何标签，返回默认权重
    if not all_labels:
        # print("[Warning] 无法从训练加载器中收集到任何标签，使用默认权重。")
        return torch.ones(num_classes, dtype=torch.float32)

    unique_labels = np.unique(all_labels)

    # 核心修复：检查是否所有预期的类别都存在于标签中
    if len(unique_labels) < num_classes:
        # print(f"[Info] 训练集中只找到 {len(unique_labels)} 个类别（期望 {num_classes} 个）。")
        # print("[Info] 这在源域只包含正常样本的设置下是正常现象。将使用默认的平衡权重 [1.0, 1.0, ...]。")
        # 当训练数据不包含所有类别时，无法计算有意义的权重，返回均匀权重
        weights = torch.ones(num_classes, dtype=torch.float32)
    else:
        # 只有当所有类别都存在时，才使用sklearn计算权重
        # print("[Info] 训练集中找到所有期望的类别，正在计算类别权重...")
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=all_labels
        )
        weights = torch.tensor(weights, dtype=torch.float32)

    return weights

# ===========================================


def main(idx, seed, configs):
    """主函数（实验入口），支持两种数据集模式"""
    # 创建日志和报告目录 (这部分逻辑不变)
    log_dir_name = f"section_{configs.fan_section}" if not str(configs.fan_section).startswith(
        's') else f"scenario_{configs.fan_section}"
    full_path_log = os.path.join('Output//myMethod//log_files', log_dir_name, f"tgt_{idx}")
    os.makedirs(full_path_log, exist_ok=True)
    full_path_rep = os.path.join('Output//myMethod//TuneReport', log_dir_name, f"tgt_{idx}")
    os.makedirs(full_path_rep, exist_ok=True)

    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, 'log_file' + currtime))

    datasets_src = []
    datasets_tgt = []

    # 检查当前是旧的section模式还是新的scenario模式
    is_scenario_mode = str(configs.fan_section).startswith('s')

    if is_scenario_mode:
        # =================================================================
        # ========== 1. 新的数据集加载逻辑 (Scenario-based) ==========
        # =================================================================
        logger.info(f"启动 Scenario 模式，当前场景: {configs.fan_section}")
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
        datasets_object_src = [ReadScenarioData(scenario, domain_id, seed, configs) for domain_id in datasets_src]
        datasets_object_tgt = [ReadScenarioData(scenario, domain_id, seed, configs) for domain_id in datasets_tgt]

    else:
        # =================================================================
        # ========== 2. 旧的数据集加载逻辑 (Section-based) ============
        # =================================================================
        logger.info(f"启动 Section 模式，当前 Section: {configs.fan_section}")
        section = str(configs.fan_section).zfill(2)  # 确保是 '00', '01' 等格式

        # 根据 section 定义域列表
        if section == '00':
            datasets_list = ['W', 'X', 'Y', 'Z']
        elif section == '01':
            datasets_list = ['A', 'B', 'C']
        elif section == '02':
            datasets_list = ['L1', 'L2', 'L3', 'L4']
        else:
            raise ValueError(f"未知的 Section: {section}。请在 0, 1, 2 中选择。")

        # 使用留一法划分源域和目标域
        dataset_idx_list = list(range(len(datasets_list)))
        tgt_idx_list = [idx]  # main函数传入的参数决定哪个是目标域
        src_idx_list = [i for i in dataset_idx_list if i not in tgt_idx_list]

        datasets_tgt = [datasets_list[i] for i in tgt_idx_list]
        datasets_src = [datasets_list[i] for i in src_idx_list]

        # 使用 ReadMIMII 加载器
        datasets_object_src = [ReadMIMII(domain, seed, section, configs) for domain in datasets_src]
        datasets_object_tgt = [ReadMIMII(domain, seed, section, configs) for domain in datasets_tgt]

    # 更新配置对象（方便其他地方引用）
    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src
    logger.info(f"源域 (Source Domains): {datasets_src}")
    logger.info(f"目标域 (Target Domains): {datasets_tgt}")

    # =================================================================
    # ========== 3. 通用的数据加载和模型训练流程 (无需修改) =========
    # =================================================================

    # 创建训练和测试数据加载器
    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src if train is not None]
    test_loaders_src = [test for train, test in train_test_loaders_src if test is not None]

    # 自动计算类别权重
    class_weights = compute_class_weights_from_dataloader(train_loaders_src, configs.num_classes).to(configs.device)

    # 加载目标域数据加载器
    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

    # 创建跨域训练数据迭代器
    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)

    # 初始化模型
    model = DGCDN(configs, seed, class_weights=class_weights)  #

    # 记录配置参数
    for k, v in sorted(vars(configs).items()):
        logger.info('\t{}: {}'.format(k, v))

    # 模型训练
    all_result = model.train_model(
        train_minibatches_iterator,
        test_loaders_tgt + test_loaders_src,  # 测试集包含目标域和所有源域
        logger
    )

    # 结果处理与保存 (这部分逻辑不变)
    all_result = {
        'loss_rc': np.array(all_result['loss_rc']),
        'loss_rr': np.array(all_result['loss_rr']),
        'loss_ca': np.array(all_result['loss_ca']),
        'loss_cl': np.array(all_result['loss_cl']),
        'acces': np.array(all_result['acces']),
        'auc': np.array(all_result['auc']),
    }

    sio.savemat(os.path.join(full_path_log, 'loss_all_result' + currtime + '.mat'), all_result)

    gen_report = GenReport(full_path_rep)
    gen_report.write_file(configs=configs, test_item=None, loss_acc_result=all_result)
    gen_report.save_file(currtime)

    # if model.best_acc > 0:
    #     print("#" * 80)
    #     print(f'Best ACC: {model.best_acc:.4f}')
    #     print(f'Best AUC: {model.best_auc:.4f}')
    #     print(f'Best Precision: {model.best_precision:.4f}')
    #     print(f'Best Recall: {model.best_recall:.4f}')
    #     print(f'Best F1-SCORE: {model.best_F1_score:.4f}')
    #     print("#" * 80)

    #网格搜索
    # 修改为:
    if model.best_acc > 0:  # 确保模型至少有一次有效的评估
        print(f"Run finished for seed {seed}. "
              f"Best ACC: {model.best_acc:.4f}, Best AUC: {model.best_auc:.4f}, "
              f"Best PRE: {model.best_precision:.4f}, Best REC: {model.best_recall:.4f}, "
              f"Best F1: {model.best_F1_score:.4f}")
        # 创建一个包含所有最佳指标的字典
        best_metrics = {
            'accuracy': model.best_acc,
            'auc': model.best_auc,
            'precision': model.best_precision,
            'recall': model.best_recall,
            'f1_score': model.best_F1_score
        }
        return best_metrics  # <-- 返回一个字典
    else:
        # 如果训练或评估失败，返回一个包含失败值的字典
        print(f"Run failed or did not yield a valid score for seed {seed}.")
        return {'accuracy': 0.0, 'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    # 1. 定义您想要搜索的超参数网格
    param_grid = {
        # 'dropout': [0.6],
        'dropout': [0.3],
        'w_ca': [0, 0.01, 0.1, 1, 5, 10],
        # 'w_ca': [0],
        'w_rc': [0, 0.01, 0.1, 1, 5, 10],
        # 'w_rc': [0],
        'w_rr': [0, 0.01, 0.1, 1, 5, 10],
        # 'w_rr': [0],
    }

    # 2. 定义实验设置
    # scenarios_to_test = ['s1', 's5', 's11'] # 定义要为哪些场景运行网格搜索
    # scenarios_to_test = ['s1'] # 定义要为哪些场景运行网格搜索
    # scenarios_to_test = ['s2', 's3', 's6','s7','s10','s11','s13','s14','s4','s5','s8', 's9', 's12']
    # scenarios_to_test = ['00','01', '02', 's1', 's5', 's11'] # 定义要为哪些场景运行网格搜索
    scenarios_to_test = ['00','01', '02'] # 定义要为哪些场景运行网格搜索
    N_SEEDS_PER_COMBO = 3 # 为每个超参数组合运行多少次不同随机种子的实验，以获得更稳健的结果
    output_csv_file = f'grid_search_full_metrics_{time.strftime("%Y%m%d%H%M%S")}.csv'

    # ========================== START OF MODIFICATION ==========================
    # 3. 更新CSV文件的表头，以包含所有指标
    param_keys = list(param_grid.keys())
    csv_headers = param_keys + ['scenario', 'seed', 'acc', 'auc', 'precision', 'recall',
                                'f1_score']
    # =========================== END OF MODIFICATION ===========================

    # 4. 设置并开始网格搜索
    grid = ParameterGrid(param_grid)
    print(f"Starting Grid Search. Total combinations: {len(grid)}")
    print(f"Results will be saved to: {output_csv_file}")

    # 创建CSV文件并写入新表头
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    run_counter = 0
    # 循环遍历每个场景
    for scenario in scenarios_to_test:
        print(f"\n{'=' * 30} TESTING SCENARIO: {scenario} {'=' * 30}")
        configs.fan_section = scenario

        # 循环遍历每一种超参数组合
        for params in grid:
            # 为当前组合运行多次随机种子实验
            for i in range(N_SEEDS_PER_COMBO):
                run_counter += 1
                seed = int(time.time()) + run_counter
                set_random_seed(seed)

                # 更新配置对象
                for key, value in params.items():
                    setattr(configs, key, value)

                print(f"\n--- [Run {run_counter}] Scenario: {scenario}, Seed: {seed}, Params: {params} ---")

                # 执行训练并获取包含所有分数的字典
                best_metrics_dict = main(0, seed, configs)

                # ========================== START OF MODIFICATION ==========================
                # 5. 从返回的字典中获取所有指标
                acc_score = best_metrics_dict.get('accuracy', 0.0)
                auc_score = best_metrics_dict.get('auc', 0.0)
                precision_score = best_metrics_dict.get('precision', 0.0)
                recall_score = best_metrics_dict.get('recall', 0.0)
                f1_score = best_metrics_dict.get('f1_score', 0.0)

                # 6. 准备包含所有指标的完整数据行
                result_row = list(params.values()) + [
                    scenario,
                    seed,
                    f"{acc_score:.4f}",
                    f"{auc_score:.4f}",
                    f"{precision_score:.4f}",
                    f"{recall_score:.4f}",
                    f"{f1_score:.4f}"
                ]
                # =========================== END OF MODIFICATION ===========================

                # 7. 将完整结果写入CSV文件
                with open(output_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result_row)

                print(f"    Run finished. Result logged to {output_csv_file}")

    print(f"\n{'=' * 30} GRID SEARCH COMPLETE {'=' * 30}")
    print(f"All results have been saved to {output_csv_file}")