# DDNEL.py

#########################################################################
#
# 本脚本旨在复现论文《Fault Diagnosis of Unseen Modes in Chemical Process
# via Fusing Invariance and Specificity》(简称 DDNEL)
#
# 核心复现点:
# 1. 严格遵循 'MLFE.py' 的数据加载与实验框架。
#    - 支持 'scenario' 和 'section' 模式。
#    - 使用 MultiInfiniteDataLoader 进行源域训练。
#    - 采用相同的日志、结果保存和多轮次运行结构。
# 2. 实现 DDNEL 论文中描述的核心模型和训练策略。
#    - 架构: Shared Net, Domain-Specific Modules, Domain-Invariant Module.
#    - 训练: 两阶段优化 (特有模块优化 + 不变模块优化)。
#      - 损失函数: 交叉熵, 熵最大化损失, LMMD损失。
#    - 推理: 基于熵的加权融合策略。
#
#########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import yaml
import random

# 导入 DGCDN 项目的工具和数据加载器 (与MLFE.py保持一致)
from utils.DictObj import DictObj
from utils.CreateLogger import create_logger
from utils.CalIndex import cal_index
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.DatasetClass import MultiInfiniteDataLoader


# --- DDNEL 辅助函数定义 ---

def entropy_loss(logits):
    """ 计算熵损失，用于最大化预测不确定性 (论文公式1, 3) """
    p = F.softmax(logits, dim=1)
    # 最小化负熵 = 最大化熵
    return -torch.mean(torch.sum(p * F.log_softmax(p, dim=1), dim=1))


def lmmd_loss(features_src, features_tgt, labels_src, labels_tgt, num_classes=2, kernel_mul=2.0, kernel_num=5,
              fix_sigma=None):
    """
    计算局部最大均值差异 (Local Maximum Mean Discrepancy) (论文公式4)
    用于对齐不同源域之间同类样本的分布。
    """
    # ... [LMMD 实现，这是一个标准但较长的函数，为简洁起见，此处省略]
    # 在实际项目中，通常会从一个库中导入或使用一个经过验证的实现。
    # 下方提供一个简化版的实现框架。
    batch_size = features_src.size(0)
    total_loss = 0.0

    for c in range(num_classes):
        # 筛选出属于类别c的样本
        src_c = features_src[labels_src == c]
        tgt_c = features_tgt[labels_tgt == c]

        if src_c.shape[0] == 0 or tgt_c.shape[0] == 0:
            continue  # 如果某个类别没有样本，则跳过

        # --- MMD 计算 ---
        def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            n_samples = int(source.size()[0]) + int(target.size()[0])
            total = torch.cat([source, target], dim=0)
            total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance = ((total0 - total1) ** 2).sum(2)
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
            return sum(kernel_val)

        len_src_c = src_c.size(0)
        len_tgt_c = tgt_c.size(0)
        kernels = guassian_kernel(src_c, tgt_c, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

        XX = kernels[:len_src_c, :len_src_c]
        YY = kernels[len_src_c:, len_src_c:]
        XY = kernels[:len_src_c, len_src_c:]

        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        total_loss += loss

    return total_loss / num_classes


# --- DDNEL 模型组件定义 ---

class SharedBackbone(nn.Module):
    """ 共享骨干网络 G """

    def __init__(self, configs):
        super(SharedBackbone, self).__init__()
        dims = [configs.backbone.input_dim] + configs.backbone.hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, x):
        # 展平输入信号
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class SubModule(nn.Module):
    """
    可复用的子模块，用于构建域特有模块(G_Di)和域不变模块(G_C)
    包含一个特征提取器F和一个分类器D
    """

    def __init__(self, configs, input_dim):
        super(SubModule, self).__init__()
        # 特征提取器 F
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, configs.submodule.feature_dim),
            nn.BatchNorm1d(configs.submodule.feature_dim),
            nn.ReLU(inplace=True)
        )
        # 分类器 D
        self.classifier = nn.Sequential(
            nn.Linear(configs.submodule.feature_dim, configs.submodule.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(configs.submodule.classifier_hidden, configs.submodule.num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return features, logits


class DDNEL(nn.Module):
    """ DDNEL 论文核心模型 """

    def __init__(self, configs, num_source_domains):
        super(DDNEL, self).__init__()
        self.configs = configs
        self.num_source_domains = num_source_domains
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")

        # 1. 初始化网络组件
        self.backbone = SharedBackbone(configs).to(self.device)
        backbone_output_dim = self.backbone.output_dim

        # 域特有模块 G_Di (每个源域一个)
        self.specific_modules = nn.ModuleList(
            [SubModule(configs, backbone_output_dim) for _ in range(num_source_domains)]
        ).to(self.device)

        # 域不变模块 G_C
        self.invariant_module = SubModule(configs, backbone_output_dim).to(self.device)

        # 2. 定义分离的优化器 (对应论文中的两阶段训练)
        # 优化器1: 优化骨干网络G和所有特有模块G_Di
        self.optimizer_spec = optim.Adam(
            list(self.backbone.parameters()) + list(self.specific_modules.parameters()),
            lr=configs.lr
        )
        # 优化器2: 优化骨干网络G和不变模块G_C
        self.optimizer_inv = optim.Adam(
            list(self.backbone.parameters()) + list(self.invariant_module.parameters()),
            lr=configs.lr
        )

        # 3. 性能追踪 (与MLFE.py保持一致)
        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_auc = -1.0
        self.best_precision = -1.0
        self.best_recall = -1.0
        self.early_stop_counter = 0

    def forward(self, x):
        """
        前向传播函数，主要用于测试/推理阶段的熵驱动融合
        """
        # 1. 提取共享特征
        shared_features = self.backbone(x)

        # 2. 从所有模块获取预测
        all_logits = []
        # 域不变模块的预测
        _, inv_logits = self.invariant_module(shared_features)
        all_logits.append(inv_logits)
        # 域特有模块的预测
        for i in range(self.num_source_domains):
            _, spec_logits = self.specific_modules[i](shared_features)
            all_logits.append(spec_logits)

        # 3. 熵驱动的加权融合 (论文公式 7, 8)
        entropies = []
        for logit in all_logits:
            p = F.softmax(logit, dim=1)
            # 计算每个样本的熵，然后取batch均值
            ent = -torch.sum(p * torch.log(p + 1e-8), dim=1)
            entropies.append(ent)

        # entropies: list of tensors, shape (B,)
        entropies_tensor = torch.stack(entropies, dim=1)  # (B, num_modules)

        # 计算权重: w = softmax(1/entropy)
        weights = F.softmax(1.0 / (entropies_tensor + 1e-8), dim=1).unsqueeze(2)  # (B, num_modules, 1)

        # 组合所有logits: (B, num_modules, num_classes)
        all_logits_tensor = torch.stack(all_logits, dim=1)

        # 加权平均
        final_logits = torch.sum(all_logits_tensor * weights, dim=1)  # (B, num_classes)

        return final_logits

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()

        for step in range(1, self.configs.steps + 1):
            self.train()

            # 从多源数据加载器获取数据
            source_minibatches = next(train_minibatches_iterator)

            all_xs_src, all_ys_src = [], []
            for xs, ys in source_minibatches:
                all_xs_src.append(xs.to(self.device))
                all_ys_src.append(ys.to(self.device))

            # --- 阶段1: 训练域特有模块 (G_Di) ---
            self.optimizer_spec.zero_grad()

            # ✅ Forward pass for Phase 1
            shared_features_p1 = [self.backbone(x) for x in all_xs_src]

            loss_spec_total = 0
            for i in range(self.num_source_domains):
                # L_ci
                _, logits_i_on_i = self.specific_modules[i](shared_features_p1[i])
                loss_ci = criterion(logits_i_on_i, all_ys_src[i])

                # L_u
                loss_u = 0
                for j in range(self.num_source_domains):
                    if i == j: continue
                    _, logits_i_on_j = self.specific_modules[i](shared_features_p1[j])
                    loss_u += entropy_loss(logits_i_on_j)

                loss_spec_total += loss_ci + self.configs.lambda_entropy * loss_u

            loss_spec_total.backward()  # First backward pass, graph is freed
            self.optimizer_spec.step()

            # --- 阶段2: 训练域不变模块 (G_C) ---
            self.optimizer_inv.zero_grad()

            # 冻结特有模块的参数
            for module in self.specific_modules:
                module.eval()

            # ✅ New, independent forward pass for Phase 2
            xs_src_cat = torch.cat(all_xs_src, dim=0)
            ys_src_cat = torch.cat(all_ys_src, dim=0)
            shared_features_p2_cat = self.backbone(xs_src_cat)  # Used for L_c_inv and L_uc

            # Re-compute features per domain for LMMD
            shared_features_p2_per_domain = [self.backbone(x) for x in all_xs_src]

            # 提取不变特征
            invariant_features_cat, invariant_logits = self.invariant_module(shared_features_p2_cat)

            # 1. 分类损失 L_c
            loss_c_inv = criterion(invariant_logits, ys_src_cat)

            # 2. 对抗特有模块的熵损失 L_uc
            loss_uc = 0
            for i in range(self.num_source_domains):
                # Note: We use the concatenated features here as L_uc is calculated on all source data
                _, logits_spec_on_inv = self.specific_modules[i](shared_features_p2_cat)
                loss_uc += entropy_loss(logits_spec_on_inv)

            # 3. LMMD损失
            loss_lmmd = 0
            inv_features_per_domain = [self.invariant_module(sf)[0] for sf in shared_features_p2_per_domain]
            for i in range(self.num_source_domains):
                for j in range(i + 1, self.num_source_domains):
                    loss_lmmd += lmmd_loss(
                        inv_features_per_domain[i], inv_features_per_domain[j],
                        all_ys_src[i], all_ys_src[j]
                    )

            loss_inv_total = loss_c_inv + self.configs.lambda_entropy * loss_uc + self.configs.lambda_lmmd * loss_lmmd
            loss_inv_total.backward()  # Second backward pass on a new graph
            self.optimizer_inv.step()

            # --- 定期验证与日志 ---
            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:
                print(step)
                logger.info(f"Step [{step}/{self.configs.steps}] "
                            f"Loss_Spec: {loss_spec_total.item():.4f}, "
                            f"Loss_Inv: {loss_inv_total.item():.4f}")

                acc, auc, prec, recall, f1, _ = self.test_model(test_loaders)
                avg_acc, avg_f1, avg_auc, avg_prec, avg_recall = np.mean(acc), np.mean(f1), np.mean(auc), np.mean(
                    prec), np.mean(recall)

                logger.info(f"Validation -> Avg ACC: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}, Avg AUC: {avg_auc:.4f}")

                # 更新最佳指标 (与MLFE.py一致)
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

        # 返回最佳结果字典
        return {
            'best_f1': self.best_f1, 'best_acc': self.best_acc, 'best_auc': self.best_auc,
            'best_precision': self.best_precision, 'best_recall': self.best_recall
        }

    def test_model(self, loaders):
        self.eval()
        all_acc, all_auc, all_prec, all_recall, all_f1, all_mcc = [], [], [], [], [], []

        # 只评估目标域
        num_tgt_domains = len(self.configs.datasets_tgt)
        target_loaders = loaders[:num_tgt_domains]

        with torch.no_grad():
            for loader in target_loaders:
                if loader is None: continue
                y_pred_lst, y_prob_lst, y_true_lst = [], [], []

                for x, label_fault in loader:
                    x = x.to(self.device)
                    final_logits = self.forward(x)  # 使用融合后的结果进行预测

                    final_probs = F.softmax(final_logits, dim=1)
                    y_prob_lst.append(final_probs.cpu().numpy())
                    y_preds = torch.argmax(final_probs, dim=1)
                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                if not y_true_lst: continue
                y_true, y_pred, y_prob = np.array(y_true_lst), np.array(y_pred_lst), np.vstack(y_prob_lst)

                acc, auc, prec, recall, f1 = cal_index(y_true, y_pred, y_prob)
                all_acc.append(acc);
                all_auc.append(auc);
                all_prec.append(prec)
                all_recall.append(recall);
                all_f1.append(f1)
                # MCC计算可以保持原样或暂时省略
                all_mcc.append(0)  # Placeholder

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
    # --- 日志和路径设置 (与MLFE.py完全一致) ---
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'
    log_dir_name = f"{dir_prefix}_{configs.fan_section}"
    full_path_log = os.path.join('Output/DDNEL_reproduction/log_files', log_dir_name,
                                 f"tgt_{idx if not is_scenario else 'all'}")
    os.makedirs(full_path_log, exist_ok=True)
    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, f'log_file_{currtime}'))

    # --- 数据加载逻辑 (与MLFE.py完全一致) ---
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

    # --- 模型初始化与训练 ---
    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)
    model = DDNEL(configs, num_source_domains=len(train_loaders_src))

    for k, v in sorted(vars(configs).items()):
        logger.info(f'\t{k}: {v}')

    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger
    )

    # --- 文件写入逻辑 (与MLFE.py完全一致) ---
    if best_results.get('best_f1', -1) > -1:
        save_dir = f'checkpoints/DDNEL/{dir_prefix}_{configs.fan_section}'
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"section{configs.fan_section}_best_result.txt"
        result_filepath = os.path.join(save_dir, result_filename)
        file_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(result_filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{file_timestamp}] (seed: {seed})\n")
                f.write(f"Best ACC: {best_results['best_acc']:.4f}\n")
                f.write(f"Best AUC: {best_results['best_auc']:.4f}\n")
                f.write(f"Best precision: {best_results['best_precision']:.4f}\n")
                f.write(f"Best recall: {best_results['best_recall']:.4f}\n")
                f.write(f"Best F1: {best_results['best_f1']:.4f}\n")
                f.write("-" * 40 + "\n\n")
            logger.info(f"Appended best results of run to {result_filepath}")
        except Exception as e:
            logger.error(f"Failed to write results to file: {e}")

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

if __name__ == '__main__':
    # 1. 加载配置文件
    with open(os.path.join(sys.path[0], 'config_files/DDNEL.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = to_namespace(configs)  # <--- 使用辅助函数转换为 SimpleNamespace

    # 2. 设置设备
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(f"Using device: {configs.device}")

    # 3. 实验运行循环 (与MLFE.py一致)
    scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    # scenarios_to_test = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    run_times_per_scenario = 10

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")
        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            seed = int(time.time()) + i
            set_random_seed(seed)
            main(0, seed, configs)