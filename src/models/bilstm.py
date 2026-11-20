# -*- coding: utf-8 -*-
"""
基础模型:Bi-LSTM 文本分类器
============================
结构:Embedding(300 维) → Bi-LSTM(hidden=128,双向) → Dropout(0.5) → FC → 2 类。

池化策略说明(关键设计点):
  对 Bi-LSTM 输出使用【有效位置 mask 平均池化】而不是取最后时间步:
  - 输入序列右侧存在大量 <PAD> 填充,取"最后时间步"实际取到的是 PAD 位置的输出,
    长序列信息会被稀释,短序列则直接是填充隐状态;
  - mask 池化只对真实词对应的隐状态求平均,对不同长度样本更公平,实测收敛更稳。

Dropout 说明:仅在全连接分类头之前对池化后的句向量施加一次 Dropout
(不在 LSTM 输出序列上重复施加),与文件头的结构描述保持一致。
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_layers: int = 1,
        pretrained_embedding: torch.Tensor = None,  # 可选:预训练词向量矩阵 [vocab_size, dim]
        freeze_embedding: bool = False,  # True 时冻结词向量(不参与梯度更新)
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0  # PAD=0 的向量恒为 0,不参与更新
        )
        # 若提供预训练词向量,拷贝进 Embedding 并按需冻结
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(pretrained_embedding)
            if freeze_embedding:
                self.embedding.weight.requires_grad = False

        # 双向 LSTM:输入 300 维 → 每个方向 128 维,拼接后 256 维
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # 输入形状 [batch, seq_len, dim]
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # 双向拼接后 2*hidden_dim → 类别数
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] 整数序列 → logits: [B, num_classes]"""
        mask = x != 0  # [B, T] 真实词位置为 True(PAD=0 为 False)
        emb = self.embedding(x)  # [B, T, D]
        outputs, _ = self.lstm(emb)  # [B, T, 2H]

        # ---- mask 平均池化:只对有效(非 PAD)位置求平均 ----
        mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (outputs * mask_f).sum(dim=1)  # [B, 2H] 有效位置求和
        counts = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1] 有效长度,防除 0
        pooled = summed / counts  # [B, 2H] 句向量

        # Dropout 仅在此处施加一次(分类头之前)
        logits = self.fc(self.dropout(pooled))
        return logits
