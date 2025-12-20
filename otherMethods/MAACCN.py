# reproduce_MAACCN.py (Updated for Single-Label Classification Metrics)

#########################################################################
#
# 本脚本旨在复现论文《MAACCN: An Intelligent Decoupling Diagnosis
# Method for Compound Faults in Electrohydrostatic Actuators》(简称 MAACCN)
#
# 核心复现点:
# 1. 严格遵循 'MLFE.py' 的数据加载与实验框架。
# 2. 实现 MAACCN 论文中描述的核心模型和算法。
# 3. [按要求修改] 评估指标已从多标签解耦指标切换为标准的单标签
#    分类指标 (acc, auc, prec, recall, f1)。
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
from math import log2

# 假设这些工具脚本与 MLFE.py 项目中的脚本位于相同的位置
from utils.CreateLogger import create_logger
# 按要求，使用 cal_index 函数
from utils.CalIndex import cal_index
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.DatasetClass import MultiInfiniteDataLoader


# --- MAACCN 论文模型组件定义 (与之前版本相同) ---

class ECA(nn.Module):
    """ 高效通道注意力 (ECA) 模块 """

    def __init__(self, channels, b=1, y=2):
        super(ECA, self).__init__()
        t = int(abs((log2(channels) + b) / y))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1)
        return x * y.expand_as(x)


class MaximizedAggregationRouting(nn.Module):
    """ 最大化聚合路由算法 """

    def __init__(self, in_caps, out_caps, in_dim, out_dim, num_iterations=3):
        super(MaximizedAggregationRouting, self).__init__()
        self.num_iterations = num_iterations
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_A = nn.Parameter(torch.randn(in_caps, in_dim))
        self.B_A = nn.Parameter(torch.randn(in_caps))
        self.W_F1 = nn.Parameter(torch.randn(in_caps, in_dim, in_dim))
        self.W_F2 = nn.Parameter(torch.randn(out_caps, in_dim, out_dim))
        self.B_F2 = nn.Parameter(torch.randn(out_caps, out_dim))
        self.W_G1 = nn.Parameter(torch.randn(out_dim, out_dim))
        self.W_G2 = nn.Parameter(torch.randn(out_caps, out_dim, in_dim))
        self.B_G2 = nn.Parameter(torch.randn(out_caps, in_dim))
        self.layer_norm = nn.LayerNorm(out_dim)
        self.W_S = nn.Parameter(torch.randn(in_caps, out_caps, in_dim))
        self.B_S = nn.Parameter(torch.randn(in_caps, out_caps))
        self.beta_use = nn.Parameter(torch.randn(in_caps, out_caps))
        self.beta_ign = nn.Parameter(torch.randn(in_caps, out_caps))

    def forward(self, x_inp):
        B = x_inp.shape[0]
        a_e = torch.einsum('bif,if->bi', x_inp, self.W_A) / np.sqrt(self.in_dim) + self.B_A
        R_em = torch.ones(B, self.in_caps, self.out_caps, device=x_inp.device) / self.out_caps
        for r in range(self.num_iterations):
            f_a_e = torch.sigmoid(a_e)
            D_use = f_a_e.unsqueeze(-1) * R_em
            D_ign = f_a_e.unsqueeze(-1) - D_use
            phi_em = self.beta_use * D_use - self.beta_ign * D_ign
            temp_F = torch.einsum('bif,ifh->bih', x_inp, self.W_F1) / np.sqrt(self.in_dim)
            temp_F_routed = torch.einsum('bi,bih->bih', phi_em.sum(dim=-1), temp_F)
            x_out = torch.einsum('bih,mhd->bmd', temp_F_routed, self.W_F2) + self.B_F2
            x_out_norm = self.layer_norm(x_out)
            temp_G = torch.einsum('bmd,dh->bmh', x_out_norm, self.W_G1)
            x_inp_hat = torch.einsum('bmh,mhi->bmi', temp_G, self.W_G2) + self.B_G2
            consistency = torch.einsum('bif,bmf,imf->bim', x_inp, x_inp_hat, self.W_S) + self.B_S
            S_em = F.log_softmax(consistency, dim=-1)
            R_em = torch.exp(S_em)
        return x_out


class MAACCN(nn.Module):
    """ 论文核心模型 MAACCN """

    def __init__(self, configs):
        super(MAACCN, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")

        # --- 1. One-Dimensional CNN Block ---
        self.cnn_block = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=128, stride=4, padding=63), nn.BatchNorm1d(16), nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15), nn.BatchNorm1d(32), nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7), nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=2, padding=3), nn.BatchNorm1d(64), nn.LeakyReLU(),
        )

        # --- 2. ECA Module ---
        self.eca = ECA(channels=64)

        # --- 3. Capsule Network Layers ---
        # The output of cnn_block is (B, 64, 16).
        # This primary_caps layer will transform it to (B, 160, 8).
        self.primary_caps = nn.Conv1d(64, 8 * 20, kernel_size=4, stride=2, padding=1)
        self.primary_caps_dim = 20

        # ======================= DEFINITIVE FIX START =======================
        # The forward pass produces capsules from a tensor of shape (B, 160, 8).
        # It's reshaped to (B, 8, 160), then viewed as (B, 8 * (160/20), 20) = (B, 64, 20).
        # Therefore, the correct number of primary capsules is 64.
        self.num_primary_caps = 8 * 8  # Corrected to 64
        # ======================== DEFINITIVE FIX END ========================

        # Digit Capsule Layer
        self.digit_caps = MaximizedAggregationRouting(
            in_caps=self.num_primary_caps,  # This will now correctly pass 64
            out_caps=configs.model.num_classes,
            in_dim=self.primary_caps_dim,
            out_dim=configs.model.digit_caps_dim,
            num_iterations=configs.model.routing_iterations
        )

        self.optimizer = optim.Adam(self.parameters(), lr=configs.lr, weight_decay=1e-4)

        # --- Performance tracking ---
        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_auc = -1.0
        self.best_recall = -1.0
        self.best_precision = -1.0
        self.early_stop_counter = 0

    def forward(self, x):
        # x shape: (B, 7, 1024)
        x = self.cnn_block(x)  # -> (B, 64, 16)
        x = self.eca(x)  # -> (B, 64, 16)

        # Primary Capsules
        x = self.primary_caps(x)  # -> (B, 160, 8)
        x = x.transpose(1, 2).contiguous()  # -> (B, 8, 160)
        x = x.view(x.size(0), -1, self.primary_caps_dim)  # -> (B, 64, 20)

        # Digit Capsules with Routing
        digit_caps_output = self.digit_caps(x)
        return digit_caps_output

    # The rest of the MAACCN class (margin_loss, train_model, test_model)
    # remains the same as the previous version. You only need to replace
    # the MAACCN class definition block.

    def margin_loss(self, y_pred_vectors, y_true):
        # Margin Loss as described in Eq. (14)
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5

        # L2 norm of vectors to get probabilities
        # v_k shape: (B, num_classes, 1)
        v_k = torch.sqrt((y_pred_vectors ** 2).sum(dim=2, keepdim=True))

        # ======================= FIX START =======================
        # One-hot encode true labels and add a dimension to match v_k's shape.
        # This changes shape from (B, num_classes) to (B, num_classes, 1).
        y_true_one_hot = F.one_hot(y_true, num_classes=self.configs.model.num_classes).float().unsqueeze(-1)
        # ======================== FIX END ========================

        # Calculate loss for each class
        loss_plus = y_true_one_hot * F.relu(m_plus - v_k).pow(2)
        loss_minus = lambda_val * (1 - y_true_one_hot) * F.relu(v_k - m_minus).pow(2)

        loss = (loss_plus + loss_minus).sum(dim=1)
        return loss.mean()

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        # (No changes here)
        self.logger = logger
        self.to(self.device)

        for step in range(1, self.configs.steps + 1):
            self.train()
            source_minibatches = next(train_minibatches_iterator)
            all_xs_src, all_ys_src = [], []
            for xs_src_batch, ys_src_batch in source_minibatches:
                all_xs_src.append(xs_src_batch.to(self.device))
                all_ys_src.append(ys_src_batch.to(self.device))

            xs_src, ys_src = torch.cat(all_xs_src), torch.cat(all_ys_src)

            if xs_src.shape[1] == 1:
                xs_src = xs_src.expand(-1, 7, -1)

            if xs_src.shape[2] > self.configs.signal_len:
                xs_src = xs_src[:, :, :self.configs.signal_len]

            self.optimizer.zero_grad()
            output_vectors = self.forward(xs_src)
            loss = self.margin_loss(output_vectors, ys_src)
            loss.backward()
            self.optimizer.step()

            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:
                logger.info(f"Step [{step}/{self.configs.steps}] Train Loss: {loss.item():.4f}")

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
        # (No changes here)
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

                    if x.shape[1] == 1:
                        x = x.expand(-1, 7, -1)

                    if x.shape[2] > self.configs.signal_len:
                        x = x[:, :, :self.configs.signal_len]

                    output_vectors = self.forward(x)

                    probs = torch.sqrt((output_vectors ** 2).sum(dim=2))
                    y_preds = torch.argmax(probs, dim=1)
                    y_probs_normalized = F.softmax(probs, dim=1)

                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_prob_lst.append(y_probs_normalized.cpu().numpy())
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
    # (函数内容不变)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(idx, seed, configs):
    # (函数大部分内容不变)
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'
    log_dir_name = f"{dir_prefix}_{configs.fan_section}"
    full_path_log = os.path.join('Output/MAACCN_reproduction/log_files', log_dir_name,
                                 f"tgt_{idx if not is_scenario else 'all'}")
    os.makedirs(full_path_log, exist_ok=True)
    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, f'log_file_{currtime}'))

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

    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)
    model = MAACCN(configs)

    for k, v in sorted(vars(configs).items()):
        logger.info(f'\t{k}: {v}')

    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger
    )

    # --- [修改] 文件写入逻辑, 保存新的指标 ---
    if best_results and best_results.get('best_acc', -1) > -1:
        save_dir = f'checkpoints/MAACCN/{dir_prefix}_{configs.fan_section}'
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
    # (主执行块内容不变, 仍从YAML加载配置)
    with open(os.path.join(sys.path[0], 'config_files/MAACCN.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = to_namespace(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(f"Using device: {configs.device}")

    scenarios_to_test = ['01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    # scenarios_to_test = ['00']
    run_times_per_scenario = 10

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")
        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            seed = int(time.time()) + i
            set_random_seed(seed)
            main(0, seed, configs)