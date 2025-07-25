import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import librosa
from collections import OrderedDict

# ==============================================================================
#  导入您的自定义模块
# ==============================================================================
try:
    from DGCDN import DGCDN, DictObj
    from models.Networks import Encoder_DGCDN
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保此脚本与您的项目在同一个Python环境中运行。")
    sys.exit(1)

# ==============================================================================
#  音频处理函数
# ==============================================================================
CONFIG = {
    'n_mels': 128, 'frames': 157, 'n_fft': 1024, 'hop_length': 210,
    'dtype_feature': np.float32,
}


def file_load(wav_name):
    try:
        y, sr = librosa.load(wav_name, sr=None, mono=True)
        return sr, y
    except Exception as e:
        print(f"文件加载失败: {wav_name} - {str(e)}");
        return None, None


def adjust_audio_length(y, target_length):
    if len(y) < target_length:
        noise = np.random.normal(0, 0.001, target_length - len(y))
        return np.concatenate([y, noise])
    return y[:target_length]


def wav_to_mel1d(file_name):
    sr, y = file_load(file_name)
    if sr is None: return None
    try:
        required_samples = (CONFIG['frames'] - 1) * CONFIG['hop_length'] + CONFIG['n_fft']
        y = adjust_audio_length(y, required_samples)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'], power=2.0
        )
        log_mel = np.log1p(mel + sys.float_info.epsilon).astype(CONFIG['dtype_feature'])
        current_frames = log_mel.shape[1]
        if current_frames < CONFIG['frames']:
            log_mel = np.pad(log_mel, ((0, 0), (0, CONFIG['frames'] - current_frames)),
                             mode='constant', constant_values=-80)
        elif current_frames > CONFIG['frames']:
            start = (current_frames - CONFIG['frames']) // 2
            log_mel = log_mel[:, start:start + CONFIG['frames']]
        log_mel = log_mel.T
        if log_mel.shape != (CONFIG['frames'], CONFIG['n_mels']): return None
        return log_mel
    except Exception as e:
        print(f"特征提取失败: {file_name} - {str(e)}");
        return None


# ==============================================================================
#  Grad-CAM 核心实现
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output): self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        _, health_features_vec = self.model.encoder_h(x)
        logits = self.model.classifer(health_features_vec)
        if class_idx is None: class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        return cam, class_idx


# ==============================================================================
#  主可视化函数
# ==============================================================================
def visualize_and_save(input_tensor, original_mel, model, file_name, output_dir):
    try:
        target_layer = model.encoder_h.conv5
        print(f"成功定位目标层: model.encoder_h.conv5")
    except AttributeError as e:
        print(f"❌ 错误: 无法在模型中找到 'conv5'。请重新检查模型结构。");
        raise e

    grad_cam = GradCAM(model, target_layer=target_layer)

    is_anomaly = 'anomaly' in file_name.lower() or 'abnormal' in file_name.lower()
    target_class = 1 if is_anomaly else 0
    cam, pred_class = grad_cam(input_tensor, class_idx=target_class)
    print(f"文件: {os.path.basename(file_name)}, CAM目标类别: {target_class}, 模型预测类别: {pred_class}")

    heatmap_1d = cam.cpu().squeeze().numpy()  # 将热力图移回CPU进行处理
    original_mel_t = original_mel.T
    heatmap_resized = cv2.resize(heatmap_1d[np.newaxis, :], (original_mel_t.shape[1], original_mel_t.shape[0]))
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (
                np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    mel_for_viz = (original_mel_t - np.min(original_mel_t)) / (np.max(original_mel_t) - np.min(original_mel_t) + 1e-8)
    mel_for_viz_rgb = np.stack([mel_for_viz] * 3, axis=-1)

    alpha = 0.5
    superimposed_img = np.uint8(mel_for_viz_rgb * 255 * (1 - alpha) + heatmap_colored * alpha)

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    predicted_label = "Anomaly" if pred_class == 1 else "Normal"
    fig.suptitle(f'Grad-CAM for {os.path.basename(file_name)} - Predicted as {predicted_label}', fontsize=16)

    axs[0].imshow(original_mel_t, aspect='auto', origin='lower', cmap='viridis');
    axs[0].set_title('Original Mel Spectrogram')
    axs[1].imshow(heatmap_normalized, aspect='auto', origin='lower', cmap='jet');
    axs[1].set_title('Grad-CAM Heatmap')
    axs[2].imshow(superimposed_img, aspect='auto', origin='lower');
    axs[2].set_title('Superimposed Image')
    for ax in axs: ax.set_xlabel('Time Frames'); ax.set_ylabel('Mel Bins')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    save_path = os.path.join(output_dir, f"grad_cam_{base_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 可视化结果已保存至: {save_path}")
    plt.close()


def prepare_input(wav_path):
    mel_spectrogram_2d = wav_to_mel1d(wav_path)
    if mel_spectrogram_2d is None: raise ValueError(f"无法处理文件: {wav_path}")
    input_tensor = torch.from_numpy(mel_spectrogram_2d).float().reshape(1, -1).unsqueeze(1)
    return input_tensor, mel_spectrogram_2d


# ==============================================================================
#  脚本执行入口
# ==============================================================================
if __name__ == '__main__':
    # --- ⚙️ 用户配置区 ⚙️ ---
    MODEL_PATH = r'checkpoints\section00\section00_acc-1.0000_auc-1.0000pre_-1rec_0.5686713286713286_f1-1.0000_20250707_113058.pth'
    HEALTHY_WAV_PATH = r'E:\code\myMethod-20250415\Data\fan\test\section_00_target_test_normal_0002_m-n_Z.wav'
    ANOMALY_WAV_PATH = r'E:\code\myMethod-20250415\Data\fan\test\section_00_target_test_anomaly_0002_m-n_Z.wav'
    OUTPUT_DIR = "case_study_visuals"

    if not all(os.path.exists(p) for p in [MODEL_PATH, HEALTHY_WAV_PATH, ANOMALY_WAV_PATH]):
        print("❌ 错误: 一个或多个文件路径不存在。请检查配置的路径是否正确。")
    else:
        print("🚀 开始 Grad-CAM 可视化流程...")
        # ✅ 第1步: 确定运行设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"将使用设备: {device}")

        # 加载模型时，先统一加载到CPU，避免显存问题
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        configs = DictObj(checkpoint['configs']) if isinstance(checkpoint['configs'], dict) else checkpoint['configs']

        model = DGCDN(configs, seed=checkpoint.get('seed', 42))
        model.load_state_dict(
            {k: v for k, v in checkpoint.items() if k in model.state_dict()},
            strict=False
        )
        # ✅ 第2步: 将整个模型移动到目标设备
        model.to(device)
        model.eval()
        print("✅ 模型加载并移动至设备完毕。")

        print("\n准备输入数据...")
        healthy_input, healthy_mel = prepare_input(HEALTHY_WAV_PATH)
        anomaly_input, anomaly_mel = prepare_input(ANOMALY_WAV_PATH)
        print("✅ 输入数据准备完毕。")

        # ✅ 第3步: 将输入数据也移动到和模型相同的设备
        healthy_input = healthy_input.to(device)
        anomaly_input = anomaly_input.to(device)

        print("\n生成可视化图像...")
        visualize_and_save(healthy_input, healthy_mel, model, HEALTHY_WAV_PATH, OUTPUT_DIR)
        visualize_and_save(anomaly_input, anomaly_mel, model, ANOMALY_WAV_PATH, OUTPUT_DIR)
        print("\n🎉 流程全部完成！")