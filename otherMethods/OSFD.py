# OSFD_reproduction.py

#########################################################################
#
# 本脚本旨在复现论文《Open-Set Fault Diagnosis Based on 1D-ResNet
# With Fusion of Cross-Class and Extreme Information...》
#
# 核心复现点:
# 1. 严格遵循 'VibrMamba.py' 的数据加载与实验框架。
#    - 支持 'scenario' 和 'section' 模式。
#    - 使用 MultiInfiniteDataLoader 进行源域训练。
#    - 采用相同的日志和多轮次运行结构。
# 2. 实现 OSFD 论文中描述的核心模型和检测逻辑。
#    - 模型架构: 1D-ResNet。
#    - OSFD方法: FCEI (融合集体和极端信息)。
#    - 评估指标: AUROC, FPR95，用于衡量开集识别能力。
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
from scipy.spatial.distance import mahalanobis, cosine
from sklearn.metrics import roc_auc_score

# 导入 DGCDN/MLFE 项目的工具和数据加载器
from utils.DictObj import DictObj
from utils.CreateLogger import create_logger
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.DatasetClass import MultiInfiniteDataLoader


# --- OSFD 论文模型组件定义 ---

class ResidualBlock(nn.Module):
    """ 1D-ResNet 的残差块 """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果维度或步长变化，需要一个 shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class OSFD_1D_ResNet(nn.Module):
    """
    OSFD 论文核心模型 (1D-ResNet) 及 FCEI 检测逻辑
    """

    def __init__(self, configs):
        super(OSFD_1D_ResNet, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")

        num_blocks = configs.model.num_res_blocks
        d_model = configs.model.d_model
        num_classes = configs.model.num_classes

        self.in_channels = d_model
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)

        # 根据论文描述堆叠残差块
        self.layer1 = self._make_layer(d_model, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(d_model * 2, num_blocks=num_blocks - 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model * 2, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=configs.lr, weight_decay=1e-4)

        # FCEI 所需参数
        self.feature_centers = None
        self.shared_cov_inv = None
        self.FCDM = None
        self.alpha = configs.osfd.alpha

        # 性能追踪
        self.best_auc = -1.0
        self.early_stop_counter = 0

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def get_features(self, x):
        """ 提取最终全连接层之前的特征 """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        features = torch.flatten(out, 1)
        return features

    def forward(self, x, return_features=False):
        features = self.get_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

    @torch.no_grad()
    def _calculate_osfd_params(self, train_loaders_src):
        """ 训练后，计算FCEI所需的特征中心和FCDM """
        self.eval()
        self.logger.info("Calculating feature centers and FCDM for OSFD...")

        all_features = []
        all_labels = []

        for loader in train_loaders_src:
            for x, y in loader:
                x = x.to(self.device)
                _, features = self.forward(x, return_features=True)
                all_features.append(features.cpu())
                all_labels.append(y.cpu())

        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)

        # 1. 计算每个类的特征中心
        num_classes = self.configs.model.num_classes
        feature_dim = all_features.shape[1]
        self.feature_centers = torch.zeros(num_classes, feature_dim)

        for i in range(num_classes):
            class_features = all_features[all_labels == i]
            if len(class_features) > 0:
                self.feature_centers[i] = class_features.mean(dim=0)

        # 2. 计算共享的逆协方差矩阵 (用于马氏距离)
        # 使用整体特征的协方差以增加鲁棒性
        cov_matrix = np.cov(all_features.numpy(), rowvar=False)
        self.shared_cov_inv = torch.tensor(np.linalg.pinv(cov_matrix), dtype=torch.float32)

        # 3. 计算 FCDM (Feature Center Distance Matrix)
        self.FCDM = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    self.FCDM[i, j] = 0
                else:
                    center_i = self.feature_centers[i].numpy()
                    center_j = self.feature_centers[j].numpy()
                    # 使用封装的马氏距离计算
                    self.FCDM[i, j] = self._mahalanobis_dist(center_i, center_j)

        self.logger.info("OSFD parameters (feature centers, FCDM) calculated successfully.")

    def _mahalanobis_dist(self, u, v):
        """ 计算两个向量之间的马氏距离 """
        return mahalanobis(u, v, self.shared_cov_inv.numpy())

    @torch.no_grad()
    def get_fcei_score(self, x):
        """ 对输入样本计算 FCEI 分数 """
        self.eval()
        logits, features = self.forward(x, return_features=True)
        probs = F.softmax(logits, dim=1)
        msp, pred_class = torch.max(probs, dim=1)

        fcei_scores = []
        for i in range(x.shape[0]):
            feat_i = features[i].cpu().numpy()
            pred_cls_i = pred_class[i].item()
            msp_i = msp[i].item()

            # 计算当前样本到所有类中心的马氏距离向量 d_x
            dist_vec = torch.zeros(self.configs.model.num_classes)
            for j in range(self.configs.model.num_classes):
                center_j = self.feature_centers[j].numpy()
                dist_vec[j] = self._mahalanobis_dist(feat_i, center_j)

            # 获取FCDM中对应的行 d_i
            fcdm_row = self.FCDM[pred_cls_i]

            # 计算相似度分数 DS(x)
            # 1 - cosine 是因为cosine距离范围是[0, 2]，相似度范围是[-1, 1]
            # 我们希望向量越相似，得分越高，所以用 1 - 距离
            ds_score = 1.0 - cosine(dist_vec.numpy(), fcdm_row.numpy())

            # 线性组合得到 FCEI score
            fcei_score = self.alpha * ds_score + (1 - self.alpha) * msp_i
            fcei_scores.append(fcei_score)

        return torch.tensor(fcei_scores)

    def train_model(self, train_minibatches_iterator, id_test_loaders, ood_test_loader, logger):
        self.logger = logger
        self.to(self.device)

        for step in range(1, self.configs.steps + 1):
            print(step)
            self.train()

            source_minibatches = next(train_minibatches_iterator)
            xs_src, ys_src = [], []
            for xs, ys in source_minibatches:
                xs_src.append(xs.to(self.device))
                ys_src.append(ys.to(self.device))
            xs_src, ys_src = torch.cat(xs_src), torch.cat(ys_src)

            self.optimizer.zero_grad()
            final_logits = self.forward(xs_src)
            loss = F.cross_entropy(final_logits, ys_src)
            loss.backward()
            self.optimizer.step()

            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:
                logger.info(f"Step [{step}/{self.configs.steps}] Train Loss: {loss.item():.4f}")

                # 在训练分类器时，我们仍然可以先计算ID数据的分类准确率
                id_acc = self._test_classification(id_test_loaders)
                logger.info(f"Validation on ID data -> Avg ACC: {np.mean(id_acc):.4f}")

        # 训练结束后，计算FCEI所需参数
        train_loaders_src = [l for it in train_minibatches_iterator.infinite_iters for l in it.loader.loaders]
        self._calculate_osfd_params(train_loaders_src)

        # 最后，进行OSFD评估
        logger.info("--- Final Open-Set Fault Diagnosis Evaluation ---")
        auroc, fpr95 = self.test_osfd(id_test_loaders, ood_test_loader)
        logger.info(f"OSFD Results -> AUROC: {auroc:.4f}, FPR95: {fpr95:.4f}")

        # 使用AUROC作为早停和最佳模型判断依据
        if auroc > self.best_auc:
            logger.info(f"New best AUROC found: {auroc:.4f} (previously {self.best_auc:.4f})")
            self.best_auc = auroc

        return {'best_auroc': auroc, 'best_fpr95': fpr95}

    @torch.no_grad()
    def _test_classification(self, loaders):
        self.eval()
        all_acc = []
        for loader in loaders:
            if loader is None: continue
            correct, total = 0, 0
            for x, y_true in loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                logits = self.forward(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_true).sum().item()
                total += y_true.size(0)
            if total > 0:
                all_acc.append(correct / total)
        return all_acc

    @torch.no_grad()
    def test_osfd(self, id_loaders, ood_loader):
        self.eval()

        # 1. 获取 ID 样本的 FCEI 分数
        id_scores = []
        for loader in id_loaders:
            if loader is None: continue
            for x, _ in loader:
                x = x.to(self.device)
                scores = self.get_fcei_score(x)
                id_scores.extend(scores.cpu().numpy())

        # 2. 获取 OOD 样本的 FCEI 分数
        ood_scores = []
        if ood_loader:
            for x, _ in ood_loader:
                x = x.to(self.device)
                scores = self.get_fcei_score(x)
                ood_scores.extend(scores.cpu().numpy())

        if not id_scores or not ood_scores:
            self.logger.warning("Could not perform OSFD test: ID or OOD data is missing.")
            return 0.0, 1.0

        # 3. 计算 AUROC 和 FPR95
        scores = np.concatenate([id_scores, ood_scores])
        # ID 样本标签为0，OOD样本标签为1。FCEI分数越高越像ID，所以OOD是正类。
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

        # FCEI分数是“已知度”，分数越低越可能是OOD。所以OOD的预测分数为 -scores
        auroc = roc_auc_score(labels, -scores)

        # 计算FPR95
        tpr_threshold = 0.95
        fpr, tpr, thresholds = roc_curve(labels, -scores)
        fpr95_idx = np.where(tpr >= tpr_threshold)[0]
        fpr95 = fpr[fpr95_idx[0]] if len(fpr95_idx) > 0 else 1.0

        return auroc, fpr95


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(idx, seed, configs):
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'
    log_dir_name = f"{dir_prefix}_{configs.fan_section}"
    full_path_log = os.path.join('Output/OSFD_reproduction/log_files', log_dir_name,
                                 f"tgt_{idx if not is_scenario else 'all'}")
    os.makedirs(full_path_log, exist_ok=True)
    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, f'log_file_{currtime}'))

    # --- 数据加载逻辑 (严格复用，并进行OOD适配) ---
    datasets_src_orig, datasets_tgt = [], []
    if is_scenario:
        logger.info(f"启动 Scenario 模式，当前场景: {configs.fan_section}")
        # (Scenario definitions are the same as in VibrMamba.py)
        # ... [Full scenario_definitions dictionary here] ...
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
        datasets_src_orig = scenario_definitions[configs.fan_section]['source']
        datasets_tgt = scenario_definitions[configs.fan_section]['target']
    else:  # Section模式
        logger.info(f"启动 Section 模式，当前 Section: {configs.fan_section}")
        section = str(configs.fan_section).zfill(2)
        datasets_map = {'00': {'source': ['X', 'Y', 'Z'], 'target': ['W']},
                        '01': {'source': ['B', 'C'], 'target': ['A']},
                        '02': {'source': ['L2', 'L3', 'L4'], 'target': ['L1']}, }
        if section not in datasets_map: raise ValueError(f"未知的 Section: {section}。")
        datasets_src_orig = datasets_map[configs.fan_section]['source']
        datasets_tgt = datasets_map[configs.fan_section]['target']

    # --- OSFD 适配: 从源域中划分一个作为OOD数据集 ---
    if not datasets_src_orig:
        logger.error("原始源域为空，无法进行训练和OOD划分。")
        return

    ood_domain = datasets_src_orig.pop()  # 将最后一个源域作为OOD
    datasets_src = datasets_src_orig

    logger.info(f"Training Source Domains (ID): {datasets_src}")
    logger.info(f"OOD Domain: {ood_domain}")
    logger.info(f"Target Domains (ID for test): {datasets_tgt}")

    # 加载数据集对象
    if is_scenario:
        datasets_object_src = [ReadScenarioData(configs.fan_section, domain, seed, configs) for domain in datasets_src]
        ood_dataset_object = ReadScenarioData(configs.fan_section, ood_domain, seed, configs)
        datasets_object_tgt = [ReadScenarioData(configs.fan_section, domain, seed, configs) for domain in datasets_tgt]
    else:
        section = str(configs.fan_section).zfill(2)
        datasets_object_src = [ReadMIMII(domain, seed, section, configs) for domain in datasets_src]
        ood_dataset_object = ReadMIMII(ood_domain, seed, section, configs)
        datasets_object_tgt = [ReadMIMII(domain, seed, section, configs) for domain in datasets_tgt]

    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src if train is not None]

    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt_id = [test for train, test in train_test_loaders_tgt if test is not None]

    _, ood_test_loader = ood_dataset_object.load_dataloaders()

    if not train_loaders_src:
        logger.error("没有可用的源域训练数据加载器，无法继续训练。")
        return

    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)
    model = OSFD_1D_ResNet(configs)

    for k, v in sorted(vars(configs).items()):
        logger.info(f'\t{k}: {v}')

    # 传递ID和OOD加载器给训练函数
    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt_id, ood_test_loader, logger
    )

    # 严格复用文件写入逻辑，保存OSFD指标
    if best_results.get('best_auroc', -1) > -1:
        save_dir = f'checkpoints/OSFD_repro/{dir_prefix}_{configs.fan_section}'
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"section{configs.fan_section}_best_osfd_result.txt"
        result_filepath = os.path.join(save_dir, result_filename)

        file_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(result_filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{file_timestamp}] (seed: {seed})\n")
                f.write("Best AUROC:\n")
                f.write(f"{best_results['best_auroc']:.4f}\n")
                f.write("Best FPR95:\n")
                f.write(f"{best_results['best_fpr95']:.4f}\n")
                f.write("-" * 40 + "\n\n")
            logger.info(f"Appended best OSFD results of run to {result_filepath}")
        except Exception as e:
            logger.error(f"Failed to write results to file: {e}")


if __name__ == '__main__':
    # 1. 加载配置文件
    try:
        with open(os.path.join(sys.path[0], 'config_files/OSFD.yaml'), 'r', encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            configs = DictObj(configs)
    except FileNotFoundError:
        print("错误: 请确保 'config_files/OSFD_repro.yaml' 文件存在。")
        sys.exit(1)

    # 2. 设置设备
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
        # 动态导入 roc_curve，因为它可能不在主环境中
        try:
            from sklearn.metrics import roc_curve
        except ImportError:
            print("警告: scikit-learn 未安装，FPR95计算将被跳过。请运行 'pip install scikit-learn'")


            # 创建一个假的 roc_curve 函数
            def roc_curve(y_true, y_score):
                return np.array([0., 1.]), np.array([0., 1.]), np.array([0, 1])
        globals()['roc_curve'] = roc_curve
        print("Using device: cuda")
    else:
        configs.device = 'cpu'
        print("Using device: cpu")

    # 3. 实验运行循环 (与参考代码一致)
    scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    run_times_per_scenario = 5

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")

        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            seed = int(time.time()) + i
            set_random_seed(seed)
            main(0, seed, configs)