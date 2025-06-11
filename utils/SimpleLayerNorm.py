import torch
import torch.nn as nn

# pytorch自带的nn.LayerNorm为https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html，但是需要提前指定normalized_shape
#为此，我使用Functional模块对其进行简单的封装，这样无需提前指定normalized_shape，而是在forward阶段从输入的x中进行推导。
# https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html

# class SimpleLayerNorm1d(nn.Module):
#     '''
#     test code
#     x = torch.randn((3,1,1024))
#     B,C,L = x.size()
#     print(B,C,L)

#     the_ln = SimpleLayerNorm1d()
#     y = the_ln(x)

#     print(y.shape)
#     print(list(the_s_ln.parameters()))#这样做的一个问题就是其参数为空，导致无法直接优化


#     '''
#     def __init__(self):
#         super(SimpleLayerNorm1d, self).__init__()

#     def forward(self, x):
#         B, C, L = x.size()
#         y = torch.nn.functional.layer_norm(x, [C,L], eps=1e-05)

#         return y



class LayerNorm(nn.Module):
    """
    自定义层归一化模块（适配多维输入）

    特性：
    - 自动推导输入形状进行归一化
    - 支持半精度浮点计算
    - 可选仿射变换（可学习参数）

    设计说明：
    1. 与PyTorch原生LayerNorm的区别：
       - 原生需要预先指定normalized_shape
       - 本实现自动根据输入维度计算
    2. 特别处理半精度浮点类型输入
    3. 仿射变换参数按特征维度学习

    初始化参数：
    :param num_features: 特征维度数（通常为通道数）
    :param eps: 数值稳定系数，防止除零（默认1e-5）
    :param affine: 是否启用仿射变换（默认True）
    """
    '''
    # Copied from DDG code
    # the_ln = LayerNorm(out_chan)
    '''
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features  # 特征维度数（通常为通道数）
        self.affine = affine              # 是否启用仿射变换
        self.eps = eps                    # 数值稳定系数
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        """
        前向传播流程：
        1. 计算每个样本的均值和标准差
        2. 执行归一化
        3. 应用仿射变换（如果启用）
        """
        # 形状调整参数：[-1, 1, 1,...]（保持批次维度，其他维度为1）
        shape = [-1] + [1] * (x.dim() - 1)  # 例如输入为(B,C,H,W)则生成[-1,1,1,1]
        # 半精度浮点特殊处理（float16）
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape).half()  # 计算均值后转回半精度
            std = x.view(-1).float().std().view(*shape).half()    # 计算标准差
        else:
            # 常规计算流程
            # 展平非批次维度：[B, C*H*W]（保持批次维度）
            x_flat = x.view(x.size(0), -1)
            # 计算各样本的均值（沿特征维度）
            mean = x_flat.mean(dim=1).view(*shape)  # [B, 1, 1,...]
            # 计算各样本的标准差（无偏估计：ddof=1）
            std = x_flat.std(dim=1, unbiased=True).view(*shape)  # [B, 1, 1,...]

        # 执行归一化：(x - mean) / (std + eps)
        x = (x - mean) / (std + self.eps)

        # 应用仿射变换（如果启用）
        if self.affine:
            # 调整参数形状：[1, C, 1, 1,...]（匹配输入维度）
            param_shape = [1, -1] + [1] * (x.dim() - 2)  # 例如输入为(B,C,H,W)则生成[1,C,1,1]
            # 缩放和平移
            x = x * self.gamma.view(*param_shape) + self.beta.view(*param_shape)
        return x
