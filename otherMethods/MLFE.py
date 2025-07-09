# reproduce_MLFE_paper.py

#########################################################################
#
# 本脚本旨在复现论文《Multilevel feature encoder for transfer learning-based
# fault detection on acoustic signal》(简称 MLFE)
#
# 核心复现点:
# 1. 严格遵循 'PropellerDANN.py' 和 'DGCDN.py' 的数据加载与实验框架。
#    - 支持 'scenario' 和 'section' 模式。
#    - 使用 MultiInfiniteDataLoader 进行源域训练。
#    - 采用相同的日志和多轮次运行结构。
# 2. 实现 MLFE 论文中描述的核心模型和特征工程。
#    - Feature Engineering: 频率掩码, 频域统计特征, K-Means聚类特征。
#    - Model Architecture: FourierTransformEncoder,
#      FrequencyDomainStatisticalEncoder, LearnableEnsembleModel。
#
#########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import yaml
import random
from sklearn.cluster import KMeans
from scipy.stats import iqr, entropy
from sklearn.metrics import confusion_matrix


# 导入 DGCDN 项目的工具和数据加载器
from utils.DictObj import DictObj
from utils.CreateLogger import create_logger
from utils.CalIndex import cal_index  # 假设这个函数能计算MCC等指标
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.DatasetClass import MultiInfiniteDataLoader


# --- MLFE 论文模型组件定义 ---

class FourierTransformEncoder(nn.Module):
    """
    MLFE 论文中的傅里叶变换编码器 (基于Transformer)
    详情请见论文 3.9.1 节
    """

    def __init__(self, input_dim, model_dim, num_heads, num_layers, patch_size, dropout=0.1):
        super(FourierTransformEncoder, self).__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (input_dim // patch_size), model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection_header = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2)  # 输出为2分类的logits
        )

    # In class FourierTransformEncoder:

    def forward(self, x):
        # x shape: (B, N) where N is sequence length
        # Patching
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)  # (B, num_patches, patch_size)
        x = self.patch_embedding(x)  # (B, num_patches, model_dim)

        # --- FIX START ---
        # 动态对齐序列长度，以处理数据加载器返回不同长度信号的情况
        num_patches_runtime = x.shape[1]
        num_patches_init = self.positional_encoding.shape[1]  # 这是初始化时固定的长度 (32)

        # 如果运行时的patch数与初始化的不匹配，则进行对齐
        if num_patches_runtime != num_patches_init:
            # 如果运行时序列更长，则截断
            if num_patches_runtime > num_patches_init:
                x = x[:, :num_patches_init, :]
            # 如果运行时序列更短，则用0填充
            else:
                padding = torch.zeros(x.shape[0], num_patches_init - num_patches_runtime, x.shape[2], device=x.device)
                x = torch.cat([x, padding], dim=1)
        # --- FIX END ---

        # Add positional encoding
        # 现在 x 的形状保证是 (B, 32, model_dim)，可以安全地与 positional_encoding 相加
        x += self.positional_encoding

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Global Average Pooling over patches
        x = x.mean(dim=1)  # (B, model_dim)

        # Projection Header
        output = self.projection_header(x)  # (B, 2)
        return output


class FrequencyDomainStatisticalEncoder(nn.Module):
    """
    MLFE 论文中的频域统计编码器
    详情请见论文 3.9.2 节
    """

    def __init__(self, input_dim, hidden_dim=256):
        super(FrequencyDomainStatisticalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # 输出为2分类的logits
        )

    def forward(self, x):
        return self.encoder(x)


class LearnableEnsembleModel(nn.Module):
    """
    MLFE 论文中的可学习集成模型
    详情请见论文 3.9.3 节
    """

    def __init__(self):
        super(LearnableEnsembleModel, self).__init__()
        # a, b, c are learnable parameters as described in Eq. (21)
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, fte_output, fse_output):
        # fte_output: (B, 2), fse_output: (B, 2)
        # Eq. (21)
        combined_logits = self.a * fte_output + self.b * fse_output + self.c
        return combined_logits


def extract_mlfe_features(x_batch, configs, kmeans_model=None):
    """
    实现MLFE论文中最重要的特征工程步骤
    Input: x_batch (Tensor): 原始一维信号, shape (B, L)
    Output:
        - masked_fft (Tensor): 掩码后的傅里叶变换
        - stat_features (Tensor): 频域统计特征
        - cluster_features (Tensor): 基于聚类的特征
    """
    x_np = x_batch.cpu().numpy()

    # --- FIX START ---
    # 数据加载器返回的是3D张量 (B, C, L), C是通道数 (通常是1)
    # 在解包前需要移除通道维度，将其变为2D数组 (B, L)
    if x_np.ndim == 3:
        x_np = x_np.squeeze(axis=1)  # 移除第二个维度(axis=1)
    # --- FIX END ---

    B, L = x_np.shape  # <--- 现在这里可以正常工作了

    # 1. 傅里叶变换
    fft_raw = np.fft.fft(x_np, axis=1)
    fft_abs = np.abs(fft_raw[:, :L // 2])  # 只取正频率部分

    # 2. 频率掩码 (Frequency Mask) - 低通滤波
    # 论文中设置为2000Hz, 采样率16kHz。这里按比例保留低频分量
    # 假设采样率为16000, 我们保留到2000Hz, 即保留前 1/8 的频率
    cutoff_idx = fft_abs.shape[1] // (16000 // (2 * configs.mask_freq))
    masked_fft = np.zeros_like(fft_abs)
    masked_fft[:, :cutoff_idx] = fft_abs[:, :cutoff_idx]

    # 3. 频域统计特征 (Frequency Statistical Features)
    # 论文中对整个频段分片(sliding window)计算, 这里为简化起见，对每个样本的整个频段计算
    stat_features_list = []
    for i in range(B):
        sample_fft = masked_fft[i, :]
        mean = np.mean(sample_fft)
        std = np.std(sample_fft)
        mad = np.median(np.abs(sample_fft - np.median(sample_fft)))
        maximum = np.max(sample_fft)
        minimum = np.min(sample_fft)
        energy = np.sum(sample_fft ** 2)
        sample_iqr = iqr(sample_fft)
        # 计算熵需要先归一化为概率分布
        prob_dist = sample_fft / np.sum(sample_fft)
        sample_entropy = entropy(prob_dist)

        # AR coefficients can be complex to compute, here we use a placeholder or simplified version
        # For simplicity, we'll stack the other features
        stat_features_list.append([mean, std, mad, maximum, minimum, energy, sample_iqr, sample_entropy])

    stat_features = np.array(stat_features_list)

    # 4. 基于聚类的特征 (Features based on Clustering)
    # 注意: 论文中k-means模型应在整个训练集上预训练得到固定的质心。
    # 这里为复现流程，我们做一个简化：如果提供了预训练模型，就使用它；否则动态拟合(这不完全符合论文)
    if kmeans_model is None:
        # A DUMMY/SIMPLIFIED version for demonstration
        kmeans_model = KMeans(n_clusters=configs.k_clusters, random_state=0, n_init=10).fit(stat_features)

    cluster_labels = kmeans_model.predict(stat_features)
    cluster_distances = kmeans_model.transform(stat_features)  # (B, k) distances to each centroid

    cluster_features_list = []
    for i in range(B):
        dist_to_centroids = cluster_distances[i]
        # 使用到每个质心的距离作为特征
        cluster_features_list.append(dist_to_centroids)

    cluster_features = np.array(cluster_features_list)

    # 转换为Tensor并移动到设备
    device = x_batch.device
    masked_fft_tensor = torch.tensor(masked_fft, dtype=torch.float32, device=device)
    # 将统计特征和聚类特征合并
    combined_stat_cluster_features = np.concatenate((stat_features, cluster_features), axis=1)
    stat_cluster_tensor = torch.tensor(combined_stat_cluster_features, dtype=torch.float32, device=device)

    return masked_fft_tensor, stat_cluster_tensor


class MLFE(nn.Module):
    """ 论文核心模型 """

    def __init__(self, configs, kmeans_model=None):
        super(MLFE, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")
        self.kmeans_model = kmeans_model  # 预训练的KMeans模型

        # 动态获取输入维度
        # 假设原始信号长度为1024, 傅里叶变换后取一半
        fft_len = configs.signal_len // 2

        # 统计特征8个 + 聚类特征k个
        stat_cluster_input_dim = 8 + configs.k_clusters

        # 初始化模型组件
        self.fte = FourierTransformEncoder(
            input_dim=fft_len,
            model_dim=configs.fte.model_dim,
            num_heads=configs.fte.num_heads,
            num_layers=configs.fte.num_layers,
            patch_size=configs.fte.patch_size
        ).to(self.device)

        self.fse = FrequencyDomainStatisticalEncoder(
            input_dim=stat_cluster_input_dim,
            hidden_dim=configs.fse.hidden_dim
        ).to(self.device)

        self.ensemble = LearnableEnsembleModel().to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.fte.parameters()) + list(self.fse.parameters()) + list(self.ensemble.parameters()),
            lr=configs.lr, weight_decay=1e-4
        )

        # 性能追踪
        self.best_mcc = -1.0
        self.best_auc = -1.0
        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_recall = -1.0
        self.best_precision = -1.0
        self.early_stop_counter = 0

    def forward(self, masked_fft, stat_cluster_features):
        fte_logits = self.fte(masked_fft)
        fse_logits = self.fse(stat_cluster_features)
        final_logits = self.ensemble(fte_logits, fse_logits)
        return final_logits

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        for step in range(1, self.configs.steps + 1):
            self.train()

            # 严格按照参考代码的输入方式：从源域获取数据
            source_minibatches = next(train_minibatches_iterator)

            all_xs_src, all_ys_src = [], []
            for xs_src_batch, ys_src_batch in source_minibatches:
                all_xs_src.append(xs_src_batch.to(self.device))
                all_ys_src.append(ys_src_batch.to(self.device))

            xs_src, ys_src = torch.cat(all_xs_src), torch.cat(all_ys_src)

            # --- MLFE核心特征提取 ---
            masked_fft, stat_cluster_features = extract_mlfe_features(xs_src, self.configs, self.kmeans_model)

            # --- 模型前向传播与损失计算 ---
            self.optimizer.zero_grad()
            final_logits = self.forward(masked_fft, stat_cluster_features)

            # MLFE 论文只提及了分类，所以我们用标准的交叉熵损失
            loss = F.cross_entropy(final_logits, ys_src)

            loss.backward()
            self.optimizer.step()

            # --- 定期验证与追踪最佳指标 ---
            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:
                logger.info(f"Step [{step}/{self.configs.steps}] Train Loss: {loss.item():.4f}")

                # 在目标域上进行测试
                acc, auc, prec, recall, f1, mcc = self.test_model(test_loaders)
                avg_acc, avg_f1, avg_auc, avg_prec, avg_recall = np.mean(acc), np.mean(f1), np.mean(auc), np.mean(prec), np.mean(recall)

                logger.info(f"Validation -> Avg ACC: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}, Avg MCC: {avg_auc:.4f}")

                if avg_acc > self.best_acc:
                    logger.info(f"New best ACC found: {avg_acc:.4f} (previously {self.best_acc:.4f})")
                    self.best_auc = avg_auc if (avg_auc > self.best_auc) else self.best_auc
                    self.best_acc = avg_acc if (avg_acc > self.best_acc) else self.best_acc
                    self.best_f1 = avg_f1   if (avg_f1 > self.best_f1) else self.best_f1
                    self.best_recall = avg_recall   if (avg_recall > self.best_recall) else self.best_recall
                    self.best_precision = avg_prec  if (avg_prec > self.best_precision) else self.best_precision
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.configs.early_stopping_patience and self.configs.early_stop:
                        logger.info("Early stopping triggered!")
                        break
        # 在训练结束后，返回本次运行的最佳结果字典
        return {
            'best_f1': self.best_f1,
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision,
            'best_recall': self.best_recall
        }

    # 替换 MLFE.py 中原有的 test_model 函数
    def test_model(self, loaders):
        self.eval()
        all_acc, all_auc, all_prec, all_recall, all_f1, all_mcc = [], [], [], [], [], []

        # 严格按照参考代码的测试方式：只评估目标域
        num_tgt_domains = len(self.configs.datasets_tgt)
        target_loaders = loaders[:num_tgt_domains]

        with torch.no_grad():
            for loader in target_loaders:
                if loader is None: continue
                y_pred_lst, y_prob_lst, y_true_lst = [], [], []

                for x, label_fault in loader:
                    x = x.to(self.device)

                    # --- 特征提取和预测 ---
                    masked_fft, stat_cluster_features = extract_mlfe_features(x, self.configs, self.kmeans_model)
                    final_logits = self.forward(masked_fft, stat_cluster_features)

                    final_probs = F.softmax(final_logits, dim=1)
                    y_prob_lst.append(final_probs.cpu().numpy())
                    y_preds = torch.argmax(final_probs, dim=1)
                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                if not y_true_lst: continue
                y_true, y_pred, y_prob = np.array(y_true_lst), np.array(y_pred_lst), np.vstack(y_prob_lst)

                # 计算指标
                acc, auc, prec, recall, f1 = cal_index(y_true, y_pred, y_prob)

                # --- FIX START ---
                # 使用 scikit-learn 的 standard 方法来计算混淆矩阵
                # 确保标签是二分类的 (0 和 1)
                labels = np.unique(y_true)
                if len(labels) == 2:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
                    # 计算MCC
                    mcc_numerator = (tp * tn) - (fp * fn)
                    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    mcc = mcc_numerator / (mcc_denominator + 1e-8)  # avoid division by zero
                else:  # 如果只有一个类别，无法计算有意义的混淆矩阵和MCC
                    mcc = 0
                    # --- FIX END ---

                all_acc.append(acc)
                all_auc.append(auc)
                all_prec.append(prec)
                all_recall.append(recall)
                all_f1.append(f1)
                all_mcc.append(mcc)

        self.train()
        return all_acc, all_auc, all_prec, all_recall, all_f1, all_mcc


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(idx, seed, configs):
    """主函数，负责执行单次实验"""
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'
    log_dir_name = f"{dir_prefix}_{configs.fan_section}"
    full_path_log = os.path.join('Output/MLFE_reproduction/log_files', log_dir_name,
                                 f"tgt_{idx if not is_scenario else 'all'}")
    os.makedirs(full_path_log, exist_ok=True)
    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, f'log_file_{currtime}'))

    # --- 数据加载逻辑 (严格复用 PropellerDANN.py 的逻辑) ---
    datasets_src, datasets_tgt = [], []
    if is_scenario:
        logger.info(f"启动 Scenario 模式，当前场景: {configs.fan_section}")
        scenario_definitions = {
            's1': {'source': ['id_00', 'id_02', 'id_04'], 'target': ['id_06']},
            's2': {'source': ['id_00', 'id_02', 'id_06'], 'target': ['id_04']},
            's3': {'source': ['id_00', 'id_04', 'id_06'], 'target': ['id_02']},
            's4': {'source': ['id_02', 'id_04', 'id_06'], 'target': ['id_00']},
            's5': {'source': ['id_00', 'id_02'], 'target': ['id_04', 'id_06']},
            's6': {'source': ['id_00', 'id_04'], 'target': ['id_02', 'id_06']},
            's7': {'source': ['id_00', 'id_06'], 'target': ['id_02', 'id_04']},
            's8': {'source': ['id_02', 'id_04'], 'target': ['id_00', 'id_06']},
            's9': {'source': ['id_02', 'id_06'], 'target': ['id_00', 'id_04']},
            's10': {'source': ['id_04', 'id_06'], 'target': ['id_00', 'id_02']},
            's11': {'source': ['id_00'], 'target': ['id_02', 'id_04', 'id_06']},
            's12': {'source': ['id_02'], 'target': ['id_00', 'id_04', 'id_06']},
            's13': {'source': ['id_04'], 'target': ['id_00', 'id_02', 'id_06']},
            's14': {'source': ['id_06'], 'target': ['id_00', 'id_02', 'id_04']},
        }
        if configs.fan_section not in scenario_definitions: raise ValueError(f"未知的场景: {configs.fan_section}。")
        datasets_src = scenario_definitions[configs.fan_section]['source']
        datasets_tgt = scenario_definitions[configs.fan_section]['target']
        datasets_object_src = [ReadScenarioData(configs.fan_section, domain, seed, configs) for domain in datasets_src]
        datasets_object_tgt = [ReadScenarioData(configs.fan_section, domain, seed, configs) for domain in datasets_tgt]
    else:
        logger.info(f"启动 Section 模式，当前 Section: {configs.fan_section}")
        section = str(configs.fan_section).zfill(2)

        datasets_map = {
            '00': {'source': ['X', 'Y', 'Z'], 'target': ['W']},
            '01': {'source': ['B', 'C'], 'target': ['A']},
            '02': {'source': ['L2', 'L3', 'L4'], 'target': ['L1']},
        }

        if section not in datasets_map: raise ValueError(f"未知的 Section: {section}。")

        datasets_tgt = datasets_map[configs.fan_section]['target']
        datasets_src = datasets_map[configs.fan_section]['source']
        datasets_object_src = [ReadMIMII(domain, seed, section, configs) for domain in datasets_src]
        datasets_object_tgt = [ReadMIMII(domain, seed, section, configs) for domain in datasets_tgt]

    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src
    logger.info(f"Source Domains: {datasets_src}")
    logger.info(f"Target Domains: {datasets_tgt}")

    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src if train is not None]
    test_loaders_src = [test for train, test in train_test_loaders_src if test is not None]

    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

    if not train_loaders_src:
        logger.error("没有可用的源域训练数据加载器，无法继续训练。")
        return

    # K-Means 模型预训练 (理想情况下)
    # 此处应加载所有源域训练数据来训练一个K-Means模型
    # 为简化流程，我们将此步骤设为可选，并在特征提取函数中处理None的情况
    kmeans_model = None  # Placeholder

    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)
    model = MLFE(configs, kmeans_model=kmeans_model)

    for k, v in sorted(vars(configs).items()):
        logger.info(f'\t{k}: {v}')

    # --- 接收训练结果 ---
    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger
    )

    # ======================= START: 新增的文件写入逻辑 =======================
    # 在 main 函数中处理文件写入
    if best_results.get('best_f1', -1) > -1:
        save_dir = f'checkpoints/MLFE/{dir_prefix}_{configs.fan_section}'
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"section{configs.fan_section}_best_result.txt"
        result_filepath = os.path.join(save_dir, result_filename)

        file_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 使用 'a' (追加) 模式，为多次运行记录结果
            with open(result_filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{file_timestamp}] (seed: {seed})\n")
                f.write("Best ACC:\n")
                f.write(f"{best_results['best_acc']:.4f}\n")
                f.write("Best AUC:\n")
                f.write(f"{best_results['best_auc']:.4f}\n")
                f.write("Best precision:\n")
                f.write(f"{best_results['best_precision']:.4f}\n")
                f.write("Best recall:\n")
                f.write(f"{best_results['best_recall']:.4f}\n")
                f.write("Best F1:\n")
                f.write(f"{best_results['best_f1']:.4f}\n")
                f.write("-" * 40 + "\n\n")
            logger.info(f"Appended best results of run to {result_filepath}")
        except Exception as e:
            logger.error(f"Failed to write results to file: {e}")
    # ======================== END: 新增的文件写入逻辑 ========================


if __name__ == '__main__':
    # 1. 加载配置文件
    with open(os.path.join(sys.path[0], 'config_files/MLFE.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    # 2. 为MLFE模型添加特定配置 (这些可以移到YAML文件中)
    configs.signal_len = 1024  # 假设输入信号长度
    configs.mask_freq = 2000  # 频率掩码的截止频率
    configs.k_clusters = 9  # K-Means的簇数量 (来自论文)

    configs.fte = DictObj({
        'model_dim': 128,
        'num_heads': 4,
        'num_layers': 3,
        'patch_size': 16  # fft_len (512) 必须能被 patch_size 整除
    })
    configs.fse = DictObj({
        'hidden_dim': 256
    })

    # 3. 设置设备
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(f"Using device: {configs.device}")

    # 4. 实验运行循环 (与参考代码一致)
    # 重点测试场景 's1', 这与MLFE论文的实验设置最匹配
    # ---- 示例：对场景各运行10次 ----
    # scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    scenarios_to_test = ['s14']

    run_times_per_scenario = 2

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")
        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            # 使用一个唯一的、可复现的种子
            seed = int(time.time()) + i
            set_random_seed(seed)
            # 对于scenario模式，第一个参数'idx'不起作用，可以固定为0
            main(0, seed, configs)