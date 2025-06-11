import torch
import torch.nn as nn
# 导入自定义的层归一化模块
from utils.SimpleLayerNorm import LayerNorm


class Conv1dBlock(nn.Module):
    '''
    自定义1D卷积模块，集成填充、卷积、归一化和激活功能

    测试代码：

        conv = Conv1dBlock(1, 32, 7, 1, 'lrelu', norm='ln', pad_type='reflect', padding=3)

        x = torch.randn((5,1,1024))

        y = conv(x)

        print(y.size())  # 输出：torch.Size([5, 32, 1024])
    '''
    '''
    Description:
    Convolutional layer

    test code:
    conv = Conv1dBlock(1, 32, 7, 1,'lrelu', norm='ln', pad_type='reflect', padding=3)
    x = torch.randn((5,1,1024))
    y = conv(x)
    print(y.size())
    output:
    >> torch.Size([5, 32, 1024])
    '''
    def __init__(self, in_chan, out_chan, kernel_size, stride, activation = 'lrelu', norm='LN', pad_type='reflect', padding=0):
        super(Conv1dBlock,self).__init__()

        # 是否使用偏置（虽然定义为True，但实际由卷积层的bias参数控制）
        self.use_bias = True

        # -------------------- 填充层配置 --------------------
        if pad_type == 'reflect':
            # 反射填充：镜像边缘像素，适合保持信号边界信息
            self.pad = nn.ReflectionPad1d(padding)
        elif pad_type == 'replicate':
            # 复制填充：重复边缘像素
            self.pad = nn.ReplicationPad1d(padding)
        elif pad_type == 'zero':
            # 零填充：用0填充边界
            self.pad = nn.ConstantPad1d(padding, 0.0)
        else:
            raise ValueError(f"不支持的填充类型: {pad_type}")

        # -------------------- 归一化层配置 --------------------
        norm_dim = out_chan  # 归一化维度=输出通道数
        if norm.lower() == 'bn':
            # 批归一化：按批次统计均值和方差
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm.lower() == 'in':
            # 实例归一化：每个样本单独归一化
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm.lower() == 'ln':
            # 层归一化：沿通道维度归一化（使用自定义实现）
            self.norm = LayerNorm(norm_dim)
        elif norm.lower() == 'none' or norm is None:
            self.norm = None
        else:
            raise ValueError(f"不支持的归一化类型: {norm}")

        # -------------------- 激活函数配置 --------------------
        if activation == 'relu':
            # ReLU激活函数，inplace操作节省内存
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            # LeakyReLU，负斜率0.2
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            # 参数化ReLU，可学习斜率
            self.activation = nn.PReLU()
        elif activation == 'selu':
            # 自缩放指数线性单元
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            # 双曲正切激活函数
            self.activation = nn.Tanh()
        elif activation == 'none' or activation is None:
            self.activation = None
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # -------------------- 卷积层核心 --------------------
        # 创建1D卷积层（注意：padding设为'valid'表示不在卷积层内自动填充）
        self.conv = nn.Conv1d(in_chan, out_chan,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=self.use_bias,  # 是否使用偏置
                              padding='valid')  # 使用外部填充层处理填充

        # 使用Kaiming初始化卷积核权重（针对LeakyReLU优化）
        nn.init.kaiming_normal_(self.conv.weight.data,
                                nonlinearity='leaky_relu',
                                a=0.2)

    def forward(self, x):
        '''
        前向传播流程：
        输入x形状：(batch_size, in_channels, sequence_length)
        输出形状：(batch_size, out_channels, sequence_length)
        '''
        # 1. 填充处理
        x = self.pad(x)  # 输出形状：(batch, in_chan, padded_length)

        # 2. 卷积运算
        x = self.conv(x)  # 输出形状：(batch, out_chan, new_length)

        # 3. 归一化处理（如果启用）
        if self.norm:
            x = self.norm(x)  # 归一化不改变形状

        # 4. 激活函数（如果启用）
        if self.activation:
            x = self.activation(x)

        return x
