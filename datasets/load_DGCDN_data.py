# 导入PyTorch相关库
import torch  # PyTorch主库
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # 数据加载和数据集基类
import os
# 导入科学计算库
import scipy.io as sio  # 用于读取.mat格式的MATLAB数据文件
import  numpy as np
# 从本地utils模块导入自定义数据加载器
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset
from collections import Counter


class ReadMIMII():
    """
    读取MIMII数据集的自定义类
    MIMII: 工业机器异常声音数据集
    """
    def __init__(self, domain, seed, section, configs):
        """
        初始化函数
        :param domain: 数据域/机器类型
        :param section: 数据分区(00/01/02)
        :param configs: 配置对象
        """
        # print(seed)
        self.configs = configs  # 存储配置参数
        self.section = section  # 数据分区
        self.domain = domain  # 数据域/机器类型
        self.seed = seed
        # 根据分区设置对应的域列表
        if self.section=='00' or self.section == 'sec00':
            self.domains = ['W','X','Y','Z']  # 分区00的机器类型
        elif self.section=='01'or self.section == 'sec01':
            self.domains=['A','B','C']  # 分区01的机器类型
        elif self.section == '02'or self.section == 'sec02':
            self.domains=['L1','L2','L3','L4']  # 分区02的机器类型

        self.batch_size = configs.batch_size  # 批量大小

    def read_data_file(self):
        """
        读取.mat数据文件并转换为PyTorch张量，构造独立的训练集和测试集。
        :return: dict，包含 'train' 和 'test'
        """
        train_file = os.path.join(r"E:\code\myMethod-20250415\Data\fan",
                                  f"attributes_{self.section}_train.mat")
        test_file = os.path.join(r"E:\code\myMethod-20250415\Data\fan",
                                 f"attributes_{self.section}_test.mat")
        train_data_mat = sio.loadmat(train_file)
        test_data_mat = sio.loadmat(test_file)
        if self.domain not in train_data_mat:
            raise ValueError(f"Train文件 {train_file} 中不存在 {self.domain} 键，请检查生成过程")
        if self.domain not in test_data_mat:
            raise ValueError(f"Test文件 {test_file} 中不存在 {self.domain} 键，请检查生成过程")
        train_domain = train_data_mat[self.domain]
        test_domain = test_data_mat[self.domain]
        raw_train_data = train_domain['data'][0, 0]
        raw_test_data = test_domain['data'][0, 0]
        train_labels = train_domain['label'][0, 0].squeeze()
        test_labels = test_domain['label'][0, 0].squeeze()
        # 调试：打印原始标签分布
        # unique_labels_train, counts_train = np.unique(train_labels, return_counts=True)
        # unique_labels_test, counts_test = np.unique(test_labels, return_counts=True)
        # print("Train原始标签分布：", unique_labels_train, counts_train)
        # print("Test原始标签分布：", unique_labels_test, counts_test)
        # === 修复开始 ===
        min_len = min(raw_train_data.shape[0], train_labels.shape[0])
        raw_train_data = raw_train_data[:min_len]
        train_labels = train_labels[:min_len]
        # === 修复结束 ===

        valid_train = ~np.isnan(raw_train_data).any(axis=(1, 2))
        valid_test = ~np.isnan(raw_test_data).any(axis=(1, 2))
        raw_train_data = raw_train_data[valid_train]
        raw_test_data = raw_test_data[valid_test]
        train_labels = train_labels[valid_train]
        test_labels = test_labels[valid_test]
        # raw_train_data = (raw_train_data - np.mean(raw_train_data)) / (np.std(raw_train_data) + 1e-8)
        # raw_test_data = (raw_test_data - np.mean(raw_test_data)) / (np.std(raw_test_data) + 1e-8)

        # 使用训练集的统计量归一化测试集
        train_mean = np.mean(raw_train_data)
        train_std = np.std(raw_train_data)

        raw_train_data = (raw_train_data - train_mean) / (train_std + 1e-8)
        raw_test_data = (raw_test_data - train_mean) / (train_std + 1e-8)  # <-- 使用训练集统计量

        train_tensor = torch.from_numpy(raw_train_data).float()
        test_tensor = torch.from_numpy(raw_test_data).float()
        train_tensor = train_tensor.reshape(train_tensor.shape[0], -1).unsqueeze(1)
        test_tensor = test_tensor.reshape(test_tensor.shape[0], -1).unsqueeze(1)
        train_dict = {'data': train_tensor, 'label': torch.from_numpy(train_labels).long()}
        test_dict = {'data': test_tensor, 'label': torch.from_numpy(test_labels).long()}

        # === 自动计算类别权重（供外部使用） ===
        counter = Counter(train_dict['label'].tolist())
        total = sum(counter.values())
        num_classes = len(counter)
        # weights = [1.0 - (counter.get(i, 0) / total) for i in range(num_classes)]

        # 确保类别顺序固定
        sorted_classes = sorted(counter.keys())
        weights = [1.0 - (counter[cls] / total) for cls in sorted_classes]

        weights = torch.tensor(weights, dtype=torch.float32).to(self.configs.device)
        self.class_weights = weights  # ← 外部可通过 model.class_weights 获取

        return {'train': train_dict, 'test': test_dict}

    def load_dataloaders(self):
        """
        创建并返回数据加载器
        :return: 训练和测试数据加载器
        """

        # 在数据加载函数中设置worker随机种子
        g = torch.Generator()
        g.manual_seed(self.seed)
        the_data = self.read_data_file()
        train_dict = the_data['train']
        test_dict  = the_data['test']

        # —— 新：构造 WeightedRandomSampler —— #
        # labels = train_dict['label'].numpy()
        # class_count = np.bincount(labels, minlength=self.configs.num_classes)
        # class_weights = 1.0 / (class_count + 1e-8)
        # sample_weights = class_weights[labels]


        # 训练集 DataLoader（保证类别平衡）
        dataset_train = SimpleDataset(train_dict)
        # print(f"[Debug] Domain {self.domain} 创建训练加载器: batch_size={self.batch_size}, 总样本={len(dataset_train)}")
        # 动态计算安全批次大小
        safe_batch_size_train = min(self.batch_size, len(dataset_train))
        # if safe_batch_size_train != self.batch_size:
        #     print(f"[Warning] Domain {self.domain} 自动调整 batch_size: {self.batch_size} -> {safe_batch_size_train}")

#########################################################################################
        # def _worker_init_fn(worker_id):
        #     np.random.seed(self.configs.seed + worker_id)

        train_loader = DataLoader(
            dataset_train,
            batch_size=safe_batch_size_train,  # 使用动态调整后的批次大小
            shuffle=True,  # 让 DataLoader 自行乱序
            generator=g,
            num_workers=0,
#############################################################################################
            # worker_init_fn=_worker_init_fn,
            drop_last=True
            )

        # 测试集 DataLoader
        dataset_test = SimpleDataset(test_dict)
        # 动态计算安全批次大小
        safe_batch_size_test = min(self.batch_size, len(dataset_test))
        test_loader = DataLoader(
            dataset_test,
            batch_size=safe_batch_size_test,
            shuffle=True,
            generator=g,
            num_workers=0,
            # worker_init_fn=_worker_init_fn,
            drop_last=False
        )

        # +++ 关键检查 +++
        if len(dataset_train) < safe_batch_size_train:
            raise ValueError(
                f"Domain {self.domain} 训练集样本数 ({len(dataset_train)}) "
                f"小于 safe_batch_size ({safe_batch_size_train}) 且 drop_last=True，这将导致空加载器!"
            )
        return train_loader, test_loader
