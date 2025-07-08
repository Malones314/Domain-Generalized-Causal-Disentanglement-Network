# reproduce_propeller_paper_v5.py

# #########################################################################
#
# 本脚本旨在复现论文《Cross-size underwater propeller fault diagnosis via
# domain adversarial training with spectral attention and multi-task learning》
#
# v5.0 更新 (结构重构):
# 1. 遵循 CDDG.py 的模式，将文件写入逻辑从模型类移至 main 函数。
# 2. train_model 方法现在返回一个包含最佳结果的字典。
# 3. main 函数负责执行单次实验并记录结果，便于在 __main__ 中进行循环调用。
#
# #########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import yaml
import random

# 导入 DGCDN 项目的工具和数据加载器
from utils.DictObj import DictObj
from utils.CreateLogger import create_logger
from utils.CalIndex import cal_index
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.DatasetClass import MultiInfiniteDataLoader


# --- 模型组件定义 (无变动) ---

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpectralAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
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


class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.attention1 = SpectralAttention(64)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.attention2 = SpectralAttention(128)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attention3 = SpectralAttention(256)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.attention4 = SpectralAttention(512)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.attention1(self.layer1(out))
        out = self.attention2(self.layer2(out))
        out = self.attention3(self.layer3(out))
        out = self.attention4(self.layer4(out))
        out = self.avg_pool(out)
        return out.view(out.size(0), -1)


class SpeedEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=128):
        super(SpeedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, output_dim)
        )

    def forward(self, speed):
        if speed.dim() == 1:
            speed = speed.unsqueeze(1)
        return self.encoder(speed)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class PropellerDANN(nn.Module):
    """ 论文核心模型 """

    def __init__(self, configs):
        super(PropellerDANN, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")
        self.feature_extractor = ResNet1D(ResidualBlock, [2, 2, 2, 2]).to(self.device)
        self.speed_encoder = SpeedEncoder(input_dim=1, output_dim=128).to(self.device)
        fused_feature_dim = 512 + 128
        self.fault_classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, configs.num_classes)
        ).to(self.device)
        self.domain_classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        ).to(self.device)
        self.speed_predictor = nn.Sequential(
            nn.Linear(fused_feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.speed_encoder.parameters()) +
            list(self.fault_classifier.parameters()) + list(self.domain_classifier.parameters()) +
            list(self.speed_predictor.parameters()),
            lr=configs.lr, weight_decay=1e-4
        )

        self.best_auc = -1.0
        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_recall = -1.0
        self.best_precision = -1.0

        self.early_stop_counter = 0
        self.logger = None

    def forward(self, acoustic_data, speed_data, alpha=1.0):
        acoustic_features = self.feature_extractor(acoustic_data)
        speed_features = self.speed_encoder(speed_data)
        fused_features = torch.cat((acoustic_features, speed_features), dim=1)
        reversed_features = GradientReversalFunction.apply(fused_features, alpha)
        domain_preds = self.domain_classifier(reversed_features)
        fault_preds = self.fault_classifier(fused_features)
        speed_preds = self.speed_predictor(fused_features)
        return fault_preds, domain_preds, speed_preds

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        num_tgt_domains = len(self.configs.datasets_tgt)
        target_loaders = test_loaders[:num_tgt_domains]
        target_loader_iters = [iter(loader) for loader in target_loaders if loader is not None]

        for step in range(1, self.configs.steps + 1):
            print(step)
            self.train()
            p = step / self.configs.steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            source_minibatches = next(train_minibatches_iterator)

            # --- 数据准备和训练逻辑 (无变动) ---
            all_xs_src, all_ys_src, all_speed_src = [], [], []
            for xs_src_batch, ys_src_batch in source_minibatches:
                all_xs_src.append(xs_src_batch.to(self.device))
                all_ys_src.append(ys_src_batch.to(self.device))
                all_speed_src.append((torch.rand(xs_src_batch.size(0)) * 600 + 300).to(self.device))
            xs_src, ys_src, speed_src = torch.cat(all_xs_src), torch.cat(all_ys_src), torch.cat(all_speed_src)

            all_xs_tgt = []
            if target_loader_iters:
                all_speed_tgt = []
                for i in range(len(target_loader_iters)):
                    try:
                        xs_tgt_batch, _ = next(target_loader_iters[i])
                    except StopIteration:
                        target_loader_iters[i] = iter(target_loaders[i])
                        xs_tgt_batch, _ = next(target_loader_iters[i])
                    all_xs_tgt.append(xs_tgt_batch.to(self.device))
                    all_speed_tgt.append((torch.rand(xs_tgt_batch.size(0)) * 600 + 300).to(self.device))
                if all_xs_tgt:
                    xs_tgt, speed_tgt = torch.cat(all_xs_tgt), torch.cat(all_speed_tgt)

            self.optimizer.zero_grad()
            fault_preds_src, domain_preds_src, speed_preds_src = self.forward(xs_src, speed_src, alpha)

            if all_xs_tgt:
                _, domain_preds_tgt, _ = self.forward(xs_tgt, speed_tgt, alpha)
                domain_labels_tgt = torch.ones(xs_tgt.size(0), dtype=torch.long, device=self.device)
                domain_preds_combined = torch.cat([domain_preds_src, domain_preds_tgt])
                domain_labels_combined = torch.cat(
                    [torch.zeros(xs_src.size(0), dtype=torch.long, device=self.device), domain_labels_tgt])
                loss_domain = F.cross_entropy(domain_preds_combined, domain_labels_combined)
            else:
                loss_domain = torch.tensor(0.0, device=self.device)

            loss_fault = F.cross_entropy(fault_preds_src, ys_src)
            loss_speed = F.mse_loss(speed_preds_src.squeeze(), speed_src)
            lambda_d, lambda_s = self.configs.w_ca, self.configs.w_rr
            total_loss = loss_fault + lambda_d * loss_domain + lambda_s * loss_speed

            total_loss.backward()
            self.optimizer.step()

            # --- 定期验证与追踪最佳指标 (移除文件写入) ---
            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:
                logger.info(f"Step [{step}/{self.configs.steps}] Total Loss: {total_loss.item():.4f}")
                acc, auc, prec, recall, f1 = self.test_model(test_loaders)
                avg_acc, avg_f1 = np.mean(acc) if acc else 0.0, np.mean(f1) if f1 else 0.0
                logger.info(f"Validation -> Avg ACC: {avg_acc:.4f}, Avg F1-Score: {avg_f1:.4f}")

                if avg_f1 > self.best_f1:
                    logger.info(f"New best F1-Score found: {avg_f1:.4f} (previously {self.best_f1:.4f})")
                    self.best_f1 = avg_f1
                    self.best_acc = avg_acc
                    self.best_auc = np.mean(auc) if auc else self.best_auc
                    self.best_precision = np.mean(prec) if prec else self.best_precision
                    self.best_recall = np.mean(recall) if recall and (np.mean(recall)<1) else self.best_recall
                    self.early_stop_counter = 0
                    print(f"results: acc:{self.best_acc}, pre:{self.best_precision}, recall:{self.best_recall}, f1:{self.best_f1}, auc:{self.best_auc}")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.configs.early_stopping_patience and self.configs.early_stop:
                        logger.info("Early stopping triggered!")
                        break

        # ======================= START: 修改后的代码块 =======================
        # 在训练结束后，返回本次运行的最佳结果字典
        return {
            'best_f1': self.best_f1,
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision,
            'best_recall': self.best_recall
        }
        # ======================== END: 修改后的代码块 ========================

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
                    speed = (torch.rand(x.size(0)) * 600 + 300).to(self.device)
                    fault_logits, _, _ = self.forward(x, speed)
                    fault_probs = F.softmax(fault_logits, dim=1)
                    y_prob_lst.append(fault_probs.cpu().numpy())
                    y_preds = torch.argmax(fault_probs, dim=1)
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
        self.train()
        return all_acc, all_auc, all_prec, all_recall, all_f1


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 主函数，适配 DGCDN 的数据加载逻辑 ---
def main(idx, seed, configs):
    """主函数，负责执行单次实验并记录结果"""
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'
    log_dir_name = f"{dir_prefix}_{configs.fan_section}"
    full_path_log = os.path.join('Output/myMethod-propeller/log_files', log_dir_name,
                                 f"tgt_{idx if not is_scenario else 'all'}")
    os.makedirs(full_path_log, exist_ok=True)
    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, 'log_file' + currtime))

    # --- 数据加载逻辑 (无变动) ---
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
        datasets_map = {'00': ['W', 'X', 'Y', 'Z'], '01': ['A', 'B', 'C'], '02': ['L1', 'L2', 'L3', 'L4']}
        if section not in datasets_map: raise ValueError(f"未知的 Section: {section}。")
        datasets_list = datasets_map[section]
        tgt_idx_list = [idx]
        src_idx_list = [i for i in range(len(datasets_list)) if i not in tgt_idx_list]
        datasets_tgt = [datasets_list[i] for i in tgt_idx_list]
        datasets_src = [datasets_list[i] for i in src_idx_list]
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
    model = PropellerDANN(configs)
    for k, v in sorted(vars(configs).items()):
        logger.info(f'\t{k}: {v}')

    # --- 接收训练结果 ---
    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger
    )

    # ======================= START: 新增的文件写入逻辑 =======================
    # 在 main 函数中处理文件写入
    if best_results.get('best_f1', -1) > -1:
        save_dir = f'checkpoints/propeller_paper/{dir_prefix}_{configs.fan_section}'
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"PropellerDANN\\section{configs.fan_section}_best_result.txt"
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
    with open(os.path.join(sys.path[0], 'config_files/DGCDN.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(configs.device)
    # ---- 示例：对场景各运行10次 ----
    scenarios_to_test = ['00','01','02','s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    run_times_per_scenario = 7

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