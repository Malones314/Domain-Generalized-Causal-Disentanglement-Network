# 导入PyTorch相关库
import torch  # PyTorch主库
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # 数据加载和数据集基类
import os
# 导入科学计算库
import scipy.io as sio  # 用于读取.mat格式的MATLAB数据文件
import numpy as np
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
        self.class_weights = None # 初始化类别权重
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
        ## <<<<<< MODIFICATION START: Modified for Zero-Shot Learning >>>>>>>>
        读取.mat数据文件并转换为PyTorch张量。
        为支持Zero-Shot，此函数现在可以独立处理训练集和测试集。
        如果一个域（作为目标域）在训练文件中不存在，它将返回一个空的训练集。
        :return: dict，包含 'train' 和 'test'
        """
        train_file = os.path.join(r"E:\code\myMethod-20250415\Data\fan",
                                  f"attributes_{self.section}_train.mat")
        test_file = os.path.join(r"E:\code\myMethod-20250415\Data\fan",
                                 f"attributes_{self.section}_test.mat")
        train_data_mat = sio.loadmat(train_file)
        test_data_mat = sio.loadmat(test_file)

        # 初始化返回值，以防某个域只存在于训练集或测试集
        train_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}
        test_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}
        train_mean, train_std = None, None

        # --- 处理训练数据 (如果存在) ---
        if self.domain in train_data_mat:
            train_domain = train_data_mat[self.domain]
            raw_train_data = train_domain['data'][0, 0]
            train_labels = train_domain['label'][0, 0].squeeze()

            # 确保数据和标签长度一致
            min_len = min(raw_train_data.shape[0], train_labels.shape[0])
            raw_train_data = raw_train_data[:min_len]
            train_labels = train_labels[:min_len]

            valid_train = ~np.isnan(raw_train_data).any(axis=(1, 2))
            raw_train_data = raw_train_data[valid_train]
            train_labels = train_labels[valid_train]

            # 计算并存储训练集的统计量用于归一化
            train_mean = np.mean(raw_train_data)
            train_std = np.std(raw_train_data)
            raw_train_data = (raw_train_data - train_mean) / (train_std + 1e-8)

            train_tensor = torch.from_numpy(raw_train_data).float()
            train_tensor = train_tensor.reshape(train_tensor.shape[0], -1).unsqueeze(1)
            train_dict = {'data': train_tensor, 'label': torch.from_numpy(train_labels).long()}

            # 计算类别权重 (仅当有训练数据时)
            counter = Counter(train_dict['label'].tolist())
            total = sum(counter.values())
            sorted_classes = sorted(counter.keys())
            weights = [1.0 - (counter[cls] / total) for cls in sorted_classes]
            self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.configs.device)

        # --- 处理测试数据 (如果存在) ---
        if self.domain in test_data_mat:
            test_domain = test_data_mat[self.domain]
            raw_test_data = test_domain['data'][0, 0]
            test_labels = test_domain['label'][0, 0].squeeze()

            valid_test = ~np.isnan(raw_test_data).any(axis=(1, 2))
            raw_test_data = raw_test_data[valid_test]
            test_labels = test_labels[valid_test]

            # 归一化: 优先使用训练集统计量，如果不存在（zero-shot目标域），则使用其自身统计量
            if train_mean is not None and train_std is not None:
                raw_test_data = (raw_test_data - train_mean) / (train_std + 1e-8)
            else:
                # 这是目标域的情况，没有对应的训练数据
                print(f"[Info] Domain {self.domain} is a target domain. Normalizing test data with its own stats.")
                test_mean = np.mean(raw_test_data)
                test_std = np.std(raw_test_data)
                raw_test_data = (raw_test_data - test_mean) / (test_std + 1e-8)

            test_tensor = torch.from_numpy(raw_test_data).float()
            test_tensor = test_tensor.reshape(test_tensor.shape[0], -1).unsqueeze(1)
            test_dict = {'data': test_tensor, 'label': torch.from_numpy(test_labels).long()}

        if not (self.domain in train_data_mat or self.domain in test_data_mat):
             print(f"[Warning] Domain {self.domain} not found in train or test .mat files.")

        return {'train': train_dict, 'test': test_dict}
        ## <<<<<< MODIFICATION END >>>>>>>>

    def load_dataloaders(self):
        """
        ## <<<<<< MODIFICATION START: Made robust for empty datasets >>>>>>>>
        创建并返回数据加载器。会优雅地处理空数据集的情况。
        """
        g = torch.Generator()
        g.manual_seed(self.seed)
        the_data = self.read_data_file()
        train_dict = the_data['train']
        test_dict  = the_data['test']

        train_loader, test_loader = None, None

        # --- 创建训练数据加载器 (如果存在训练数据) ---
        if len(train_dict['data']) > 0:
            dataset_train = SimpleDataset(train_dict)
            safe_batch_size_train = min(self.batch_size, len(dataset_train))

            # 仅当有足够样本时才创建加载器（如果drop_last=True）
            if len(dataset_train) >= safe_batch_size_train or not getattr(self.configs, 'drop_last_train', True):
                train_loader = DataLoader(
                    dataset_train,
                    batch_size=safe_batch_size_train,
                    shuffle=True,
                    generator=g,
                    num_workers=0,
                    drop_last=getattr(self.configs, 'drop_last_train', True)
                )
            else:
                 print(f"[Info] Domain {self.domain} has {len(dataset_train)} train samples, which is less than batch size {safe_batch_size_train} with drop_last=True. Returning empty train loader.")


        # --- 创建测试数据加载器 (如果存在测试数据) ---
        if len(test_dict['data']) > 0:
            dataset_test = SimpleDataset(test_dict)
            safe_batch_size_test = min(self.batch_size, len(dataset_test))
            test_loader = DataLoader(
                dataset_test,
                batch_size=safe_batch_size_test,
                shuffle=True,
                generator=g,
                num_workers=0,
                drop_last=False
            )

        return train_loader, test_loader
    ## <<<<<< MODIFICATION END >>>>>>>>


class ReadScenarioData(): # MIMIdatabase
    """
    读取按场景(scenario)划分的数据集的自定义类
    """

    def __init__(self, scenario, domain_id, seed, configs):
        """
        初始化函数
        :param scenario: 数据场景 (例如 's1', 's2')
        :param domain_id: 数据域ID (例如 'id_00', 'id_02')
        :param seed: 随机种子
        :param configs: 配置对象
        """
        self.configs = configs
        self.scenario = scenario
        self.domain_id = domain_id
        self.seed = seed
        self.batch_size = configs.batch_size
        self.class_weights = None  # 初始化类别权重

    def read_data_file(self):
        """
        读取.mat数据文件并转换为PyTorch张量。
        根据域是源域还是目标域，它可能只包含训练数据或测试数据。
        :return: dict，包含 'train' 和 'test'
        """
        # !!重要!!: 请确保此路径指向您新生成的 .mat 文件所在的文件夹
        mat_file_root = r"E:\code\myMethod-20250415\Data\0_dB_fan\fan\mat_files_scenarios"

        train_file = os.path.join(mat_file_root, f"{self.scenario}_train.mat")
        test_file = os.path.join(mat_file_root, f"{self.scenario}_test.mat")

        train_data_mat = sio.loadmat(train_file)
        test_data_mat = sio.loadmat(test_file)

        is_source_domain = self.domain_id in train_data_mat
        is_target_domain = self.domain_id in test_data_mat

        if not is_source_domain and not is_target_domain:
            raise ValueError(f"域 {self.domain_id} 在场景 {self.scenario} 的训练集和测试集中都未找到。")

        # 初始化返回值
        train_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}
        test_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}

        # 使用一个通用的处理函数来避免代码重复
        def process_data(raw_data, labels):
            if raw_data.size == 0:
                return torch.empty(0), torch.empty(0, dtype=torch.long)

            valid_indices = ~np.isnan(raw_data).any(axis=(1, 2))
            raw_data = raw_data[valid_indices]
            labels = labels[valid_indices]

            # 归一化（这里仅对自身归一化，更优做法是使用所有源域的统计量）
            mean = np.mean(raw_data)
            std = np.std(raw_data)
            processed_data = (raw_data - mean) / (std + 1e-8)

            tensor_data = torch.from_numpy(processed_data).float()
            tensor_data = tensor_data.reshape(tensor_data.shape[0], -1).unsqueeze(1)
            tensor_labels = torch.from_numpy(labels).long()
            return tensor_data, tensor_labels

        if is_source_domain:
            # 作为源域加载，只有训练数据
            domain_data_train = train_data_mat[self.domain_id]
            raw_train_data = domain_data_train['data'][0, 0]
            train_labels = domain_data_train['label'][0, 0].squeeze()

            train_data_tensor, train_labels_tensor = process_data(raw_train_data, train_labels)
            train_dict['data'] = train_data_tensor
            train_dict['label'] = train_labels_tensor

            # 计算类别权重
            if len(train_labels_tensor) > 0:
                counter = Counter(train_labels_tensor.tolist())
                total = sum(counter.values())
                sorted_classes = sorted(counter.keys())
                weights = [1.0 - (counter[cls] / total) for cls in sorted_classes]
                self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.configs.device)

        if is_target_domain:
            # 作为目标域加载，只有测试数据
            domain_data_test = test_data_mat[self.domain_id]
            raw_test_data = domain_data_test['data'][0, 0]
            test_labels = domain_data_test['label'][0, 0].squeeze()

            test_data_tensor, test_labels_tensor = process_data(raw_test_data, test_labels)
            test_dict['data'] = test_data_tensor
            test_dict['label'] = test_labels_tensor

        return {'train': train_dict, 'test': test_dict}

    def load_dataloaders(self):
        """
        创建并返回数据加载器。会优雅地处理空数据集的情况。
        """
        g = torch.Generator()
        g.manual_seed(self.seed)
        the_data = self.read_data_file()
        train_dict = the_data['train']
        test_dict = the_data['test']

        train_loader, test_loader = None, None

        # 创建训练数据加载器（如果存在训练数据）
        if len(train_dict['data']) > 0:
            dataset_train = SimpleDataset(train_dict)
            safe_batch_size_train = min(self.batch_size, len(dataset_train))
            if len(dataset_train) < safe_batch_size_train and getattr(self.configs, 'drop_last_train', True):
                print(
                    f"[Warning] Domain {self.domain_id} 训练集样本数({len(dataset_train)})不足一个batch，将返回空加载器。")
            else:
                train_loader = DataLoader(
                    dataset_train,
                    batch_size=safe_batch_size_train,
                    shuffle=True,
                    generator=g,
                    num_workers=0,
                    drop_last=getattr(self.configs, 'drop_last_train', True)  # 从配置读取drop_last
                )

        # 创建测试数据加载器（如果存在测试数据）
        if len(test_dict['data']) > 0:
            dataset_test = SimpleDataset(test_dict)
            safe_batch_size_test = min(self.batch_size, len(dataset_test))
            test_loader = DataLoader(
                dataset_test,
                batch_size=safe_batch_size_test,
                shuffle=True,
                generator=g,
                num_workers=0,
                drop_last=False
            )

        return train_loader, test_loader
