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

# 全局配置参数 (修改部分参数)
CONFIG = {
    'n_mels': 128,      # 梅尔频带数
    'frames': 157,      # 时间帧数
    'n_fft': 1024,      # 帧长
    'hop_length': 210,  # 帧移
    'target_samples': 157 * 210 + 1024,  # 目标采样点数计算
    'dtype_feature': np.float32,
    'batch_size': 500
}

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
        # 后缘填充噪声而非零值
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
        # 根据配置计算目标采样点数并调整音频长度
        required_samples = (CONFIG['frames'] - 1) * CONFIG['hop_length'] + CONFIG['n_fft']
        y = adjust_audio_length(y, required_samples)

        # 生成梅尔频谱 (输出形状: (n_mels, 时间帧))
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            power=2.0
        )

        # 对数压缩（使用log1p减少数值溢出问题）
        log_mel = np.log1p(mel + sys.float_info.epsilon)
        log_mel = log_mel.astype(CONFIG['dtype_feature'])

        # 调整时间轴长度，确保时间帧数为CONFIG['frames']
        current_frames = log_mel.shape[1]
        if current_frames < CONFIG['frames']:
            # 后缘填充 -80dB（代表静音）
            log_mel = np.pad(log_mel, ((0, 0), (0, CONFIG['frames'] - current_frames)),
                             mode='constant', constant_values=-80)
        elif current_frames > CONFIG['frames']:
            # 中心截取
            start = (current_frames - CONFIG['frames']) // 2
            log_mel = log_mel[:, start:start+CONFIG['frames']]

        # 转置，得到形状 (时间帧, n_mels)
        log_mel = log_mel.T

        # 检查尺寸是否正确
        if log_mel.shape != (CONFIG['frames'], CONFIG['n_mels']):
            print(f"特征维度调整错误: {log_mel.shape} != ({CONFIG['frames']}, {CONFIG['n_mels']})")
            return None

        return log_mel

    except Exception as e:
        print(f"特征提取失败: {file_name} - {str(e)}")
        return None


def process_section(csv_path, data_root):
    """处理单个section（保持原有路径结构）"""
    df = pd.read_csv(csv_path)
    domain_data = {
        'train': defaultdict(lambda: {'data': [], 'label': [], 'd1p': [], 'd1v': []}),
        'test': defaultdict(lambda: {'data': [], 'label': [], 'd1p': [], 'd1v': []})
    }

    # 预定义所有可能域标签（根据MIMII数据集规范）
    expected_domains = {'L1', 'L2', 'L3', 'L4', 'W', 'X', 'Y', 'Z', 'A', 'B', 'C'}

    for _, row in tqdm(df.iterrows(), desc=f"处理 {os.path.basename(csv_path)}"):
        try:
            path_parts = row['file_name'].split('/')
            filename = path_parts[-1]
            raw_data_type = 'test' if 'test' in path_parts else 'train'
            full_path = os.path.join(data_root, raw_data_type, filename)

            if not os.path.exists(full_path):
                print(f"文件不存在: {full_path}")
                continue

            features = wav_to_mel1d(full_path)
            if features is None:
                continue

            label = 1 if 'anomaly' in filename.lower() else 0
            domain = row['d1v']
            d1p = str(row['d1p'])
            d1v = str(row['d1v'])

            # 决定最终加入哪个数据集（train/test）
            final_data_type = raw_data_type
            if label == 1 and raw_data_type == 'test':
                if np.random.rand() < 0.5:
                    final_data_type = 'train'

            # 统一添加数据和标签
            domain_data[final_data_type][domain]['data'].append(features)
            domain_data[final_data_type][domain]['label'].append(label)
            domain_data[final_data_type][domain]['d1p'].append(d1p)
            domain_data[final_data_type][domain]['d1v'].append(d1v)

        except Exception as e:
            print(f"处理行失败: {str(e)}")
            continue

    # 将列表转换为numpy数组，确保每个域的数据为三维 (样本, 时间帧, 梅尔频带)
    for data_type in ['train', 'test']:
        for domain in domain_data[data_type]:
            data = domain_data[data_type][domain]
            if len(data['data']) == 0:
                continue
            data['data'] = np.stack(data['data'])  # (样本数, frames, n_mels)
            data['label'] = np.array(data['label'], dtype=np.int8)
            data['d1p'] = np.array(data['d1p'], dtype=object)
            data['d1v'] = np.array(data['d1v'], dtype=object)

    return domain_data


def generate_mat_files(data_root):
    """
    生成MAT文件：
      - 按section生成.mat文件，每个.mat文件中包含 section_info 以及按域分开的数据。
      - 每个域对应的结构体包含 'data' (形状为 (样本, 时间帧, 梅尔频带))、'label'、'd1p' 和 'd1v'。
    """
    csv_files = glob.glob(os.path.join(data_root, 'attributes_*.csv'))

    # 定义各section结构
    SECTION_CONFIG = {
        '00': {'domains': ['W', 'X', 'Y', 'Z'], 'desc': '混合不同机械声音'},
        '01': {'domains': ['A', 'B', 'C'], 'desc': '混合不同工厂噪音'},
        '02': {'domains': ['L1', 'L2', 'L3', 'L4'], 'desc': '不同级别噪音'}
    }

    for csv_file in csv_files:
        try:
            section_num = os.path.basename(csv_file).split('_')[1].split('.')[0]
            current_cfg = SECTION_CONFIG.get(section_num, {})

            # 处理原始数据
            domain_data = process_section(csv_file, data_root)

            for data_type in ['train', 'test']:
                output_path = os.path.join(data_root, f"attributes_{section_num}_{data_type}.mat")
                # 构建保存字典，包含 section_info 及各域数据
                save_dict = {
                    'section_info': {'section': section_num, 'desc': current_cfg.get('desc', '')}
                }

                for domain in current_cfg.get('domains', []):
                    if domain in domain_data.get(data_type, {}):
                        data = domain_data[data_type][domain]
                        # 转换为numpy数组
                        data_np = np.array(data['data'], dtype=CONFIG['dtype_feature'])
                        label_np = np.array(data['label'], dtype=np.uint8)
                        d1p_np = np.array(data['d1p'], dtype=object)
                        d1v_np = np.array(data['d1v'], dtype=object)
                        # 确保数据为三维 (样本, 时间帧, 梅尔频带)
                        assert data_np.ndim == 3, f"数据维度错误 应为(样本,时间,频带) 实际{data_np.shape}"

                        save_dict[domain] = {
                            'data': data_np,
                            'label': label_np,
                            'd1p': d1p_np,
                            'd1v': d1v_np
                        }

                sio.savemat(output_path, save_dict, do_compression=True)
                print(f"成功生成: {output_path}")
                for domain in current_cfg.get('domains', []):
                    if domain in save_dict:
                        shape = save_dict[domain]['data'].shape
                        print(f"域 {domain} 数据结构: data={shape}")

        except Exception as e:
            print(f"处理失败 {csv_file}: {str(e)}")
            raise


if __name__ == "__main__":
    data_root = "E:/code/DGFDBenchmark-main/Data/fan/fan"
    generate_mat_files(data_root)
