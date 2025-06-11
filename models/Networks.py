# 该代码库实现了多种先进的域自适应故障诊断网络，通过不同的网络架构和训练策略，能够有效处理来自不同设备的振动信号，并在跨域场景下保持高精度的故障诊断能力。
# 导入PyTorch核心库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
from models.Conv1dBlock import Conv1dBlock  # 自定义1D卷积块


#########################################
###################################
class Network_fan(nn.Module):
    """
    风扇故障诊断网络模型
    测试代码示例：
    x = torch.randn((5,1,20032))  # 5个样本，1通道，20032个数据点
    the_model = Network(configs)
    fv, logits = the_model(x)
    """
    '''
    test code1:
    x = torch.randn((5,1,20032))
    the_model = Network(configs)
    fv, logits = the_model(x)
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        # 第1层卷积：输入1通道，输出32通道，卷积核128，池化核4
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(4)

        # 第2层卷积：保持32通道，卷积核128，池化核4
        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        # 第3层卷积：保持32通道，卷积核128，池化核4
        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        # 第4层卷积：保持32通道，卷积核64，池化核2
        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        # 第5层卷积：保持32通道，卷积核64，池化核2
        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool5 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)  # 30%的dropout防止过拟合
        self.linear1 = nn.Linear(in_features=640, out_features=300)  # 全连接层1
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)  # 输出层

    def forward(self, x):
        """前向传播"""
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)
        x7 = self.linear1(x6)
        x8 = self.linear2(x7)

        return x6, x8 #特征向量和分类logits, x6 is the feature vector Z, x8 is the output logits for classification

    def forward_penul_fv(self, x):
        """获取倒数第二层特征"""
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)
        x7 = self.linear1(x6)

        return x7
###########
###########

# 以下为风扇数据对应的特征生成器、故障分类器和域分类器
# 结构与轴承版本类似，主要区别在于输入尺寸和网络参数
class FeatureGenerator_fan(nn.Module):
    """DGNIS风扇特征生成器

        DGNIS

        test code:
            the_model = FeatureGenerator()

            x = torch.randn((5,1,1024))

            y = the_model(x)

            print(y.shape)

            torch.Size([5, 3520])
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        # 5层卷积+池化结构，池化核更大
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool5 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        """生成特征向量"""
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)
        return x6

class FaultClassifier_fan(nn.Module):
    """DGNIS风扇故障分类器"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)

    def forward(self,x):
        """分类前向传播"""
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x2

class DomainClassifier_fan(nn.Module):
    """DGNIS风扇域分类器"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_domains = len(configs.datasets_src)

        self.linear1 = nn.Linear(in_features = 640, out_features = 300)
        self.linear2 = nn.Linear(in_features = 300, out_features =  self.num_domains)

    def forward(self, x):
        """域分类前向传播"""
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2
# DGCDN风扇数据相关组件（结构与轴承版本类似）
class Encoder_DGCDN(nn.Module):
    """DGCDN风扇编码器"""

    def __init__(self):
        super().__init__()
        # 5层卷积+池化（池化核更大）
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool5 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

    def forward(self, x):

        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        f_map = self.pool5(self.conv5(x4)) # (B, 32, 20)
        f_vec = self.flatten(f_map) # (B, 640)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, f_map.shape)

        return f_map, f_vec # f_map: feature maps (B,C,L); f_vec: feature vector (B, C*L)


class Decoder_DGCDN(nn.Module):
    """DGCDN风扇解码器"""
    def __init__(self):
        super().__init__()
        # 上采样+卷积结构（镜像编码器）
        self.up1 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True) #(B,64, 40)
        self.conv1 = Conv1dBlock(in_chan=32*2, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='zero', padding=63) #(B,32,103)
        self.up2 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True) #(B,32,206)
        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=64) #(B,32, 271)
        self.up3 = nn.Upsample(scale_factor=4, mode ='linear', align_corners=True) #(B,32,1084)
        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=127) #(B,32,1211)
        self.up4 = nn.Upsample(scale_factor=4, mode ='linear', align_corners=True) #(B,32,4844)
        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=129) #(B,32,4975)
        self.up5 = nn.Upsample(scale_factor=4, mode ='linear', align_corners=True) #(B,32,19900)
        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32,  kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=129)#(B,32,20031)
        #最后一层输出不加Activation
        self.conv6 = Conv1dBlock(32, 1, 128, 1, 'none', 'BN', 'reflect', 64)  # 输出层无激活

    def forward(self, x):
        '''
        'x' is the feature maps
        '''
        x1 = self.conv1(self.up1(x ))
        x2 = self.conv2(self.up2(x1))
        x3 = self.conv3(self.up3(x2))
        x4 = self.conv4(self.up4(x3))
        x5 = self.conv5(self.up5(x4))
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # >> torch.Size([5, 32, 128]) torch.Size([5, 32, 272]) torch.Size([5, 32, 576]) torch.Size([5, 32, 1216]) torch.Size([5, 32, 2560])

        x_rec = self.conv6(x5)  # 重构信号

        return x_rec

class Classifier_DGCDN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.linear2 = nn.Linear(in_features=300, out_features=2)
    def forward(self, x):
        '''
        'x' is the feature vector
        '''
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2  # 返回分类logits



### 关键组件总结
#
# 1. **基础网络结构**：
#    - 所有网络都采用5层卷积+池化作为特征提取器
#    - 轴承数据使用较小的池化核（2），风扇数据使用较大的池化核（4/2）
#    - 卷积后使用LeakyReLU激活和批归一化（BN）
#
# 2. **架构变体**：
#    - **DGNIS**：特征生成器+双分类器（故障分类器和域分类器）
#    - **IEDGNet**：更深的分类器/判别器（3层全连接），带Dropout
#    - **ADN**：对抗域适应网络，结构与DGNIS类似但用途不同
#    - **DGCDN**：编码器-解码器结构，支持特征重构
#
# 3. **核心设计特点**：
#    - 所有卷积层使用反射填充（reflect padding）保持信号边界信息
#    - 分类器采用浅层结构（2-3层全连接）
#    - 特征向量维度：
#      - 轴承：1984维（32通道×62长度）
#      - 风扇：640维（32通道×20长度）
#
# 4. **特殊方法**：
#    - `forward_penul_fv()`：获取倒数第二层特征（用于特征分析）
#    - `forward1()`：获取分类器中间特征（DGCDN专用）
#
# 5. **输入输出规格**：
#    - 轴承输入：1×2560（Network_bearing）或1×1024（其他）
#    - 风扇输入：1×20032（Network_fan）或1×长序列
#    - 输出：特征向量+分类logits（或重构信号）
#
# 该代码库实现了多种先进的域自适应故障诊断网络，通过不同的网络架构和训练策略，能够有效处理来自不同设备的振动信号，并在跨域场景下保持高精度的故障诊断能力。
