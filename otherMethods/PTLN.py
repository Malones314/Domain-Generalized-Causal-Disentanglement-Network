# reproduce_PTLN.py (Corrected Version)

#########################################################################
#
# 本脚本旨在复现论文《Progressive Transfer Learning: An Intelligent Fault
# Diagnosis Method for Unlabeled Rotating Machinery With Small Samples》(简称 PTLN)
#
# 核心复现点:
# 1. 严格遵循 'MAACCN.py' 的数据加载与实验框架。
# 2. 实现 PTLN 论文中描述的核心模型和算法。
# 3. 评估指标使用标准的单标签分类指标 (acc, auc, prec, recall, f1)。
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

# 假设这些工具脚本与 MAACCN.py 项目中的脚本位于相同的位置
from utils.DictObj import DictObj
from utils.CreateLogger import create_logger
from utils.CalIndex import cal_index # 使用相同的评估函数
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.DatasetClass import MultiInfiniteDataLoader

# --- PTLN 论文模型组件定义 ---

class GradReverse(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)
    在前向传播中，它是一个恒等变换。
    在反向传播中，它将梯度乘以一个负的常数(alpha)。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


def compute_mk_mmd_loss(x, y, sigma_list=[0.1, 1, 10], biased=True):
    """
    计算多核最大均值差异 (Multi-Kernel Maximum Mean Discrepancy, MK-MMD)
    """
    def _gaussian_kernel(x, y, sigma):
        # 计算高斯核矩阵
        beta = 1. / (2. * sigma)
        dist = torch.cdist(x, y)
        return torch.exp(-beta * dist.pow(2))

    # 计算核矩阵
    K_XX, K_YY, K_XY = 0, 0, 0
    for sigma in sigma_list:
        K_XX += _gaussian_kernel(x, x, sigma)
        K_YY += _gaussian_kernel(y, y, sigma)
        K_XY += _gaussian_kernel(x, y, sigma)

    # 计算MMD
    if biased:
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m, n = x.size(0), y.size(0)
        mmd = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1)) \
            + (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1)) \
            - 2 * K_XY.mean()
    return mmd


class PTLN(nn.Module):
    """ PTLN 核心模型 """
    def __init__(self, configs):
        super(PTLN, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")

        # --- 定义特征提取器 (F) ---
        # 根据论文表I，为源域和目标域创建独立的、结构相同但权重不同的渐进式提取器
        def create_feature_extractor_block(in_c, out_c, conv_params):
            # 辅助函数，创建单个卷积块
            k, s, p = conv_params
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )

        # F_1: 低维特征提取
        self.feature_extractor_S1 = create_feature_extractor_block(1, 16, (64, 3, 30)) # 假设输入通道为1，根据论文调整
        self.feature_extractor_T1 = create_feature_extractor_block(1, 16, (64, 3, 30))

        # F_2: 中维特征提取
        self.feature_extractor_S2 = create_feature_extractor_block(16, 32, (16, 1, 8))
        self.feature_extractor_T2 = create_feature_extractor_block(16, 32, (16, 1, 8))

        # F_3: 高维特征提取
        self.feature_extractor_S3 = create_feature_extractor_block(32, 64, (5, 1, 2))
        self.feature_extractor_T3 = create_feature_extractor_block(32, 64, (5, 1, 2))

        # ============================ FIX START ============================
        # --- 动态计算Flatten后的维度 (Corrected) ---
        # 假设输入信号长度为 self.configs.signal_len (e.g., 1024)
        with torch.no_grad(): # 确保此过程不计入梯度
            dummy_input = torch.randn(1, 1, self.configs.signal_len)

            # Pass the actual tensors through the network sequentially
            f1_tensor = self.feature_extractor_S1(dummy_input)
            f2_tensor = self.feature_extractor_S2(f1_tensor)
            f3_tensor = self.feature_extractor_S3(f2_tensor)

            # Now, get the shapes from the computed tensors
            f1_dim_shape = f1_tensor.shape
            f2_dim_shape = f2_tensor.shape
            f3_dim_shape = f3_tensor.shape

            # Calculate the flattened dimensions using the retrieved shapes
            self.flat_dim1 = f1_dim_shape[1] * f1_dim_shape[2]
            self.flat_dim2 = f2_dim_shape[1] * f2_dim_shape[2]
            self.flat_dim3 = f3_dim_shape[1] * f3_dim_shape[2]
        # ============================= FIX END =============================

        # --- 定义域判别器 (D) ---
        def create_discriminator(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, 2), # 2个类别: source vs target
            )

        self.domain_discriminator1 = create_discriminator(self.flat_dim1)
        self.domain_discriminator2 = create_discriminator(self.flat_dim2)
        self.domain_discriminator3 = create_discriminator(self.flat_dim3)

        # --- 定义状态分类器 (C) ---
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, configs.model.num_classes)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)

        # --- 性能跟踪 ---
        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_auc = -1.0
        self.best_recall = -1.0
        self.best_precision = -1.0
        self.early_stop_counter = 0

    def forward(self, x_s=None, x_t=None, alpha=0.0, is_test=False):
        if is_test:
            # --- 测试/推理模式 ---
            f_t1 = self.feature_extractor_T1(x_t)
            f_t2 = self.feature_extractor_T2(f_t1)
            f_t3 = self.feature_extractor_T3(f_t2)
            v_t = f_t3.view(f_t3.size(0), -1)
            class_output = self.classifier(v_t)
            return class_output

        # --- 训练模式 ---
        # 1. 源域前向传播
        f_s1 = self.feature_extractor_S1(x_s)
        f_s2 = self.feature_extractor_S2(f_s1)
        f_s3 = self.feature_extractor_S3(f_s2)
        v_s = f_s3.view(f_s3.size(0), -1)

        # 2. 目标域前向传播
        f_t1 = self.feature_extractor_T1(x_t)
        f_t2 = self.feature_extractor_T2(f_t1)
        f_t3 = self.feature_extractor_T3(f_t2)
        v_t = f_t3.view(f_t3.size(0), -1)

        # 3. 分类器输出
        class_output_s = self.classifier(v_s)

        # 4. 域判别器输出 (通过GRL)
        v_s1_flat = f_s1.view(f_s1.size(0), -1)
        v_t1_flat = f_t1.view(f_t1.size(0), -1)
        domain_output1 = self.domain_discriminator1(grad_reverse(torch.cat([v_s1_flat, v_t1_flat]), alpha))

        v_s2_flat = f_s2.view(f_s2.size(0), -1)
        v_t2_flat = f_t2.view(f_t2.size(0), -1)
        domain_output2 = self.domain_discriminator2(grad_reverse(torch.cat([v_s2_flat, v_t2_flat]), alpha))

        domain_output3 = self.domain_discriminator3(grad_reverse(torch.cat([v_s, v_t]), alpha))

        return class_output_s, domain_output1, domain_output2, domain_output3, v_s, v_t

    def train_model(self, train_minibatches_iterator_src, train_minibatches_iterator_tgt, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        # 损失函数
        loss_class = nn.CrossEntropyLoss()
        loss_domain = nn.CrossEntropyLoss()

        for step in range(1, self.configs.steps + 1):
            self.train()

            # --- 动态调整GRL的alpha参数 ---
            p = float(step) / self.configs.steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # --- 数据加载 ---
            # 源域数据
            source_minibatches = next(train_minibatches_iterator_src)
            xs_src_batch, ys_src_batch = source_minibatches[0]
            xs_src = xs_src_batch.to(self.device)
            ys_src = ys_src_batch.to(self.device)

            # 目标域数据
            target_minibatches = next(train_minibatches_iterator_tgt)
            xs_tgt_batch, _ = target_minibatches[0]
            xs_tgt = xs_tgt_batch.to(self.device)

            # --- 数据预处理 ---
            # 保证batch size一致
            batch_size = min(xs_src.size(0), xs_tgt.size(0))
            xs_src, ys_src = xs_src[:batch_size], ys_src[:batch_size]
            xs_tgt = xs_tgt[:batch_size]

            # 适配单通道输入
            if xs_src.shape[1] > 1: xs_src = xs_src[:, 0, :].unsqueeze(1)
            if xs_tgt.shape[1] > 1: xs_tgt = xs_tgt[:, 0, :].unsqueeze(1)

            # 裁剪信号长度
            if xs_src.shape[2] > self.configs.signal_len:
                xs_src = xs_src[:, :, :self.configs.signal_len]
            if xs_tgt.shape[2] > self.configs.signal_len:
                xs_tgt = xs_tgt[:, :, :self.configs.signal_len]

            # --- 训练步骤 ---
            self.optimizer.zero_grad()

            class_output_s, domain_out1, domain_out2, domain_out3, v_s, v_t = self.forward(xs_src, xs_tgt, alpha)

            # 1. 计算分类损失 (Lc)
            err_s_label = loss_class(class_output_s, ys_src)

            # 2. 计算域对抗损失 (Ld)
            domain_labels = torch.cat([
                torch.zeros(batch_size, device=self.device, dtype=torch.long),
                torch.ones(batch_size, device=self.device, dtype=torch.long)
            ])
            err_domain1 = loss_domain(domain_out1, domain_labels)
            err_domain2 = loss_domain(domain_out2, domain_labels)
            err_domain3 = loss_domain(domain_out3, domain_labels)
            err_domain_total = (err_domain1 + err_domain2 + err_domain3) / 3.0

            # 3. 计算 MMD 损失 (L_mmd)
            err_mmd = compute_mk_mmd_loss(v_s, v_t)

            # 4. 计算总损失
            total_loss = (err_s_label +
                          self.configs.model.lambda_d * err_domain_total +
                          self.configs.model.lambda_mmd * err_mmd)

            total_loss.backward()
            self.optimizer.step()

            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:
                logger.info(f"Step [{step}/{self.configs.steps}] Total Loss: {total_loss.item():.4f} | "
                            f"Cls_Loss: {err_s_label.item():.4f} | Dom_Loss: {err_domain_total.item():.4f} | "
                            f"MMD_Loss: {err_mmd.item():.4f}")

                acc, auc, prec, recall, f1 = self.test_model(test_loaders)
                avg_acc, avg_f1, avg_auc = np.mean(acc), np.mean(f1), np.mean(auc)
                avg_recall, avg_prec = np.mean(recall), np.mean(prec)

                logger.info(f"Validation -> Avg ACC: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}, Avg AUC: {avg_auc:.4f}")

                if avg_acc > self.best_acc:
                    logger.info(f"New best ACC found: {avg_acc:.4f} (previously {self.best_acc:.4f})")
                    self.best_acc = avg_acc
                    self.best_f1 = avg_f1
                    self.best_auc = avg_auc
                    self.best_recall = avg_recall
                    self.best_precision = avg_prec
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.configs.early_stopping_patience and self.configs.early_stop:
                        logger.info("Early stopping triggered!")
                        break
        return {
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision,
            'best_recall': self.best_recall,
            'best_f1': self.best_f1
        }

    def test_model(self, loaders):
        self.eval()
        all_acc, all_auc, all_prec, all_recall, all_f1 = [], [], [], [], []

        num_tgt_domains = len(self.configs.datasets_tgt)
        target_loaders = loaders[:num_tgt_domains]

        with torch.no_grad():
            for loader in target_loaders:
                if loader is None: continue
                y_pred_lst, y_prob_lst, y_true_lst = [], [], []

                for x, label_fault in loader:
                    x = x.to(self.device)

                    # 适配单通道输入
                    if x.shape[1] > 1: x = x[:, 0, :].unsqueeze(1)

                    if x.shape[2] > self.configs.signal_len:
                        x = x[:, :, :self.configs.signal_len]

                    class_output = self.forward(x_t=x, is_test=True)

                    probs = F.softmax(class_output, dim=1)
                    y_preds = torch.argmax(probs, dim=1)

                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_prob_lst.append(probs.cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                if not y_true_lst: continue

                y_true, y_pred, y_prob = np.array(y_true_lst), np.array(y_pred_lst), np.vstack(y_prob_lst)
                acc, auc, prec, recall, f1 = cal_index(y_true, y_pred, y_prob)

                all_acc.append(acc)
                all_auc.append(auc)
                all_prec.append(prec)
                all_recall.append(recall)
                all_f1.append(f1)

        self.train()
        return all_acc, all_auc, all_prec, all_recall, all_f1

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(idx, seed, configs):
    # (此函数大部分内容与 MAACCN.py 保持一致)
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'
    log_dir_name = f"{dir_prefix}_{configs.fan_section}"
    full_path_log = os.path.join('Output/PTLN_reproduction/log_files', log_dir_name,
                                 f"tgt_{idx if not is_scenario else 'all'}")
    os.makedirs(full_path_log, exist_ok=True)
    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, f'log_file_{currtime}'))

    # 数据集定义 (完全复用)
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
        datasets_src = datasets_map[configs.fan_section]['source']
        datasets_tgt = datasets_map[configs.fan_section]['target']
        datasets_object_src = [ReadMIMII(domain, seed, section, configs) for domain in datasets_src]
        datasets_object_tgt = [ReadMIMII(domain, seed, section, configs) for domain in datasets_tgt]

    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src
    logger.info(f"Source Domains: {datasets_src}")
    logger.info(f"Target Domains: {datasets_tgt}")

    # --- 数据加载器准备 ---
    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src if train is not None]
    test_loaders_src = [test for train, test in train_test_loaders_src if test is not None]

    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    train_loaders_tgt = [train for train, test in train_test_loaders_tgt if train is not None] # PTLN训练需要目标域数据
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

    if not train_loaders_src or not train_loaders_tgt:
        logger.error("源域或目标域训练数据加载器为空，无法继续训练。")
        return

    # 创建无限迭代器
    train_minibatches_iterator_src = MultiInfiniteDataLoader(train_loaders_src)
    train_minibatches_iterator_tgt = MultiInfiniteDataLoader(train_loaders_tgt)

    model = PTLN(configs)

    for k, v in sorted(vars(configs).items()):
        logger.info(f'\t{k}: {v}')

    best_results = model.train_model(
        train_minibatches_iterator_src, train_minibatches_iterator_tgt,
        test_loaders_tgt + test_loaders_src, logger
    )

    # --- 结果保存 ---
    if best_results and best_results.get('best_acc', -1) > -1:
        save_dir = f'checkpoints/PTLN/{dir_prefix}_{configs.fan_section}'
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"section{configs.fan_section}_best_result.txt"
        result_filepath = os.path.join(save_dir, result_filename)
        file_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(result_filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{file_timestamp}] (seed: {seed})\n")
                f.write(f"Best ACC:\n{best_results['best_acc']:.4f}\n")
                f.write(f"Best AUC:\n{best_results['best_auc']:.4f}\n")
                f.write(f"Best Precision:\n{best_results['best_precision']:.4f}\n")
                f.write(f"Best Recall:\n{best_results['best_recall']:.4f}\n")
                f.write(f"Best F1:\n{best_results['best_f1']:.4f}\n")
                f.write("-" * 40 + "\n\n")
            logger.info(f"Appended best results of run to {result_filepath}")
        except Exception as e:
            logger.error(f"Failed to write results to file: {e}")

if __name__ == '__main__':
    with open(os.path.join(sys.path[0], 'config_files/PTLN.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(f"Using device: {configs.device}")

    scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    run_times_per_scenario = 10 # 建议减少运行次数以便快速验证

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")
        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            seed = int(time.time()) + i
            set_random_seed(seed)
            main(0, seed, configs)