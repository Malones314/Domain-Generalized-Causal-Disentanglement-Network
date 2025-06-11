import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class _InfiniteSampler(torch.utils.data.Sampler):
    """
    无限采样器：将普通采样器包装为无限循环的采样器
    """

    def __init__(self, sampler):
        super().__init__(sampler)
        self.sampler = sampler
        self.epoch_counter = 0

    def __iter__(self):
        """
        无限生成采样索引，每个 epoch 遍历一次原始采样器
        """
        while True:
            self.epoch_counter += 1
            yield from iter(self.sampler)


class InfiniteDataLoader:
    """
    无限数据加载器：不断产生数据批次
    """

    def __init__(self, dataset, batch_size=128, weights=None, num_workers=0):
        """
        参数说明：
          - dataset: 必须是 torch.utils.data.Dataset 实例
          - batch_size: 每个批次的样本数
          - weights: 样本权重（用于加权采样），默认为 None
          - num_workers: 数据加载线程数
        """
        super().__init__()
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a torch.utils.data.Dataset instance")

        # 初始化采样器（当样本数较少时自动使用 replacement=True）
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=int(1e10),
                replacement=True
            )
        else:
            replacement = len(dataset) < 1000
            sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement)

        # 创建批量采样器时设 drop_last=False，确保即使样本数量不足一个 batch 也输出
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,  # 基础采样器
            batch_size=batch_size,
            drop_last=False  # 修改：允许输出不完整的批次
        )

        # 构造 DataLoader，并保存到 self.dataloader 便于后续重新生成迭代器
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True,
            worker_init_fn=lambda _: np.random.seed()  # 防止多进程重复
        )
        self._infinite_iterator = iter(self.dataloader)

    def __iter__(self):
        """
        无限迭代，产生数据批次
        """
        while True:
            try:
                yield next(self._infinite_iterator)
            except StopIteration:
                # 重新初始化迭代器
                self._infinite_iterator = iter(self.dataloader)
                yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError('This is an infinite dataloader!')

    def __del__(self):
        del self._infinite_iterator


class SimpleDataset(Dataset):
    def __init__(self, data_content):
        # data_content['data'] 已经是 (N, 1, 20096)
        self.data = torch.as_tensor(data_content['data'], dtype=torch.float32)
        self.class_label = torch.as_tensor(data_content['label'], dtype=torch.long)
        # +++ 新增数据校验 +++
        if len(self.data) == 0:
            raise ValueError("数据集不能为空!")
        if torch.isnan(self.data).any():
            raise ValueError("数据包含 NaN 值!")
        if torch.isinf(self.data).any():
            raise ValueError("数据包含 Inf 值!")

        # 如果数据已经是三维(例如 (N, 1, 20096))，无需再进行 unsqueeze
        if self.data.ndim != 3:
            raise ValueError("数据维度不正确，期望3D张量")

        if len(self.data) != len(self.class_label):
            raise ValueError(
                f"数据/标签数量不匹配: {len(self.data)} vs {len(self.class_label)}"
            )

    def __getitem__(self, index):
        return self.data[index], self.class_label[index]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        # 数据有效性验证
        if torch.isnan(data).any() or torch.isinf(data).any():
            raise ValueError(f"Invalid data at index {index}")
        return data, self.class_label[index]
# class MultiInfiniteDataLoader:
#     def __init__(self, loaders):
#         self.loaders = loaders
#         self.iters = [iter(dl) for dl in loaders]
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         batches = []
#         for i in range(len(self.loaders)):
#             try:
#                 batch = next(self.iters[i])
#             except StopIteration:
#                 self.iters[i] = iter(self.loaders[i])
#                 batch = next(self.iters[i])
#             batches.append(batch)
#         return batches
class MultiInfiniteDataLoader:
    def __init__(self, loaders, max_retries=3):
        self.loaders = loaders
        self.iters = [iter(dl) for dl in loaders]
        self.max_retries = max_retries  # 最大重试次数

        # 前置检查：确保每个 DataLoader 有效
        for i, loader in enumerate(loaders):
            try:
                sample = next(iter(loader))  # 尝试获取一个样本
            except StopIteration:
                raise ValueError(f"DataLoader {i} 初始化失败：无法获取任何数据")
            except Exception as e:
                raise RuntimeError(f"DataLoader {i} 初始化异常：{str(e)}")

    def __iter__(self):
        return self

    def __next__(self):
        batches = []
        for i in range(len(self.loaders)):
            retry_count = 0
            while True:
                try:
                    batch = next(self.iters[i])
                    batches.append(batch)
                    break
                except StopIteration:
                    # 重置迭代器
                    self.iters[i] = iter(self.loaders[i])
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        raise RuntimeError(
                            f"DataLoader {i} 在 {self.max_retries} 次重试后仍无法生成数据\n"
                            f"可能原因：1.数据集为空 2.batch_size设置错误 3.数据预处理异常"
                        )
        return batches
