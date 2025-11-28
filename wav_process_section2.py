import os
import glob
import librosa
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import sys
from collections import defaultdict
import time

# 全局配置参数 (与原代码保持一致)
CONFIG = {
    'n_mels': 128,  # 梅尔频带数
    'frames': 157,  # 时间帧数
    'n_fft': 1024,  # 帧长
    'hop_length': 210,  # 帧移
    'target_samples': 157 * 210 + 1024,  # 目标采样点数计算
    'dtype_feature': np.float32,
    'batch_size': 500
}


# ---------------------------------------------------------------------------
# 以下是您提供的原始代码中的核心函数，无需修改
# ---------------------------------------------------------------------------

def file_load(wav_name):
    try:
        y, sr = librosa.load(wav_name, sr=None, mono=True)
        return sr, y
    except Exception as e:
        print(f"文件加载失败: {wav_name} - {str(e)}")
        return None, None


def adjust_audio_length(y, target_length):
    """动态音频长度调整策略"""
    if len(y) < target_length:
        noise = np.random.normal(0, 0.001, target_length - len(y))
        return np.concatenate([y, noise])
    return y[:target_length]


def wav_to_mel1d(file_name):
    """
    修改后的特征提取函数：
      - 提取梅尔频谱并进行对数压缩；
      - 对时间帧进行填充或截取；
      - 返回形状为 (时间帧, 梅尔频带) 的二维矩阵。
    """
    sr, y = file_load(file_name)
    if sr is None:
        return None
    try:
        required_samples = (CONFIG['frames'] - 1) * CONFIG['hop_length'] + CONFIG['n_fft']
        y = adjust_audio_length(y, required_samples)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            power=2.0
        )
        log_mel = np.log1p(mel + sys.float_info.epsilon)
        log_mel = log_mel.astype(CONFIG['dtype_feature'])
        current_frames = log_mel.shape[1]
        if current_frames < CONFIG['frames']:
            log_mel = np.pad(log_mel, ((0, 0), (0, CONFIG['frames'] - current_frames)),
                             mode='constant', constant_values=-80)
        elif current_frames > CONFIG['frames']:
            start = (current_frames - CONFIG['frames']) // 2
            log_mel = log_mel[:, start:start + CONFIG['frames']]
        log_mel = log_mel.T
        if log_mel.shape != (CONFIG['frames'], CONFIG['n_mels']):
            print(f"特征维度调整错误: {log_mel.shape} != ({CONFIG['frames']}, {CONFIG['n_mels']})")
            return None
        return log_mel
    except Exception as e:
        print(f"特征提取失败: {file_name} - {str(e)}")
        return None


# ---------------------------------------------------------------------------
# 以下是根据您的新需求重写的主逻辑
# ---------------------------------------------------------------------------

def process_files_for_domain(file_paths, label, domain_id):
    """
    处理指定路径列表中的所有文件，并提取特征。
    """
    data_list, label_list, d1v_list = [], [], []

    for file_path in tqdm(file_paths, desc=f"处理域 {domain_id} (标签: {label})"):
        features = wav_to_mel1d(file_path)
        if features is not None:
            data_list.append(features)
            label_list.append(label)
            d1v_list.append(domain_id)  # 记录域名

    return data_list, label_list, d1v_list


def generate_mat_for_scenarios(data_root, output_dir):
    """
    根据新的场景划分为源域和目标域，并生成.mat文件。
    此版本支持多个源域和多个目标域。
    """
    # ========================== START OF CHANGES ==========================
    # 扩展 SCENARIOS 字典以包含 s1 到 s14
    SCENARIOS = {
        # --- 原始场景 (3源, 1目标) ---
        's1': {'source': ['00', '02', '04'], 'target': ['06']},
        's2': {'source': ['00', '02', '06'], 'target': ['04']},
        's3': {'source': ['00', '04', '06'], 'target': ['02']},
        's4': {'source': ['02', '04', '06'], 'target': ['00']},
        # --- 新增场景 (2源, 2目标) ---
        's5': {'source': ['00', '02'], 'target': ['04', '06']},
        's6': {'source': ['00', '04'], 'target': ['02', '06']},
        's7': {'source': ['00', '06'], 'target': ['02', '04']},
        's8': {'source': ['02', '04'], 'target': ['00', '06']},
        's9': {'source': ['02', '06'], 'target': ['00', '04']},
        's10': {'source': ['04', '06'], 'target': ['00', '02']},
        # --- 新增场景 (1源, 3目标) ---
        's11': {'source': ['00'], 'target': ['02', '04', '06']},
        's12': {'source': ['02'], 'target': ['00', '04', '06']},
        's13': {'source': ['04'], 'target': ['00', '02', '06']},
        's14': {'source': ['06'], 'target': ['00', '02', '04']},
    }
    # =========================== END OF CHANGES ===========================

    os.makedirs(output_dir, exist_ok=True)

    for name, scenario in SCENARIOS.items():
        print(f"\n{'=' * 20} 开始处理场景: {name} {'=' * 20}")

        # 1. 构建训练集 (源域的正常信号)
        train_save_dict = {}
        print("--- 正在构建训练集 ---")
        for domain_id in scenario['source']:
            # ... (这部分逻辑与之前完全相同，无需修改) ...
            domain_key = f"id_{domain_id}"
            normal_path_pattern = os.path.join(data_root, domain_key, 'normal', '*.wav')
            normal_files = glob.glob(normal_path_pattern)
            if not normal_files: continue
            data, labels, d1v = process_files_for_domain(normal_files, label=0, domain_id=domain_key)
            if data:
                train_save_dict[domain_key] = {
                    'data': np.stack(data, axis=0).astype(CONFIG['dtype_feature']),
                    'label': np.array(labels, dtype=np.uint8),
                    'd1p': np.array(d1v, dtype=object),
                    'd1v': np.array(d1v, dtype=object)
                }

        if train_save_dict:
            train_output_path = os.path.join(output_dir, f"{name}_train.mat")
            sio.savemat(train_output_path, train_save_dict, do_compression=True)
            print(f"成功生成训练集: {train_output_path}")

        # 2. 构建测试集 (目标域的正常+异常信号)
        test_save_dict = {}
        print("\n--- 正在构建测试集 ---")
        # ========================== START OF CHANGES ==========================
        # 修改为遍历目标域列表
        for target_domain_id in scenario['target']:
            target_domain_key = f"id_{target_domain_id}"

            target_normal_path = os.path.join(data_root, target_domain_key, 'normal', '*.wav')
            target_abnormal_path = os.path.join(data_root, target_domain_key, 'abnormal', '*.wav')

            target_normal_files = glob.glob(target_normal_path)
            target_abnormal_files = glob.glob(target_abnormal_path)

            if not target_normal_files and not target_abnormal_files:
                print(f"警告：在 {target_domain_key} 中未找到任何测试文件，跳过。")
                continue

            all_data, all_labels, all_d1v = [], [], []
            if target_normal_files:
                data, labels, d1v = process_files_for_domain(target_normal_files, label=0, domain_id=target_domain_key)
                all_data.extend(data)
                all_labels.extend(labels)
                all_d1v.extend(d1v)

            if target_abnormal_files:
                data, labels, d1v = process_files_for_domain(target_abnormal_files, label=1,
                                                             domain_id=target_domain_key)
                all_data.extend(data)
                all_labels.extend(labels)
                all_d1v.extend(d1v)

            # 将每个目标域的数据存入字典
            if all_data:
                test_save_dict[target_domain_key] = {
                    'data': np.stack(all_data, axis=0).astype(CONFIG['dtype_feature']),
                    'label': np.array(all_labels, dtype=np.uint8),
                    'd1p': np.array(all_d1v, dtype=object),
                    'd1v': np.array(all_d1v, dtype=object)
                }
        # =========================== END OF CHANGES ===========================

        if test_save_dict:
            test_output_path = os.path.join(output_dir, f"{name}_test.mat")
            sio.savemat(test_output_path, test_save_dict, do_compression=True)
            print(f"成功生成测试集: {test_output_path}")

    print(f"\n{'=' * 20} 所有场景处理完毕 {'=' * 20}")

if __name__ == "__main__":
    # 请将此路径修改为您包含 id_00, id_02 等文件夹的根目录
    # 根据您提供的示例，应该是 'E:\code\myMethod-20250415\Data\0_dB_fan\fan'
    data_root_path = r"E:\code\myMethod-20250415\Data\0_dB_fan\fan"

    # mat文件将保存在数据根目录下的 'mat_files' 文件夹中
    output_mat_path = os.path.join(data_root_path, "mat_files_scenarios")

    generate_mat_for_scenarios(data_root_path, output_mat_path)