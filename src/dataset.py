# -*- coding: utf-8 -*-
"""
PyTorch 数据集与数据加载器
==========================
- SentimentDataset:包装 preprocess 产出的 .pkl(token id 序列 + 标签)为 torch Dataset;
- build_loaders:构造 训练/验证/测试 三个 DataLoader。

工程约束(Windows):
  - DataLoader 一律 num_workers=0,避免 Windows 多进程 spawn 在 Git Bash 下的各种隐患;
  - 序列在 preprocess 阶段已做定长 padding(128),无需动态 collate。
"""

import os
import pickle

import numpy as np
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


class SentimentDataset(Dataset):
    """情感分类数据集:ids [N, maxlen] int64 / labels [N] int64。

    suffix 参数用于加载其他任务的同格式产物(如五级分类的 *_5c.pkl);
    默认 "" 时加载二分类的 {split}.pkl,既有行为完全不变。
    """

    def __init__(self, split: str = "train", processed_dir: str = None, suffix: str = ""):
        path = os.path.join(processed_dir or PROCESSED_DIR, f"{split}{suffix}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)
        self.texts = data.get("texts")  # 原文,仅供可视化/展示用
        assert len(self.ids) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def build_loaders(batch_size: int = 64, processed_dir: str = None, suffix: str = ""):
    """构造三个 DataLoader。shuffle 仅训练集开启,保证验证/测试评估结果稳定。

    suffix 参数用于加载其他同格式产物(如混域二分类的 *_mix.pkl);
    默认 "" 时加载二分类的 {split}.pkl,既有行为完全不变(向后兼容)。
    """
    train_ds = SentimentDataset("train", processed_dir, suffix=suffix)
    val_ds = SentimentDataset("val", processed_dir, suffix=suffix)
    test_ds = SentimentDataset("test", processed_dir, suffix=suffix)
    common = dict(num_workers=0, pin_memory=False)  # Windows 约束:num_workers=0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 自检:打印各 split 规模与一个 batch 形状
    tr, va, te = build_loaders(batch_size=64)
    print(f"train/val/test 批次数: {len(tr)}/{len(va)}/{len(te)}")
    x, y = next(iter(tr))
    print(f"batch ids: {tuple(x.shape)} {x.dtype}, labels: {tuple(y.shape)} {y.dtype}")
