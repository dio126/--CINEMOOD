# -*- coding: utf-8 -*-
"""
进阶模型 A:Bi-LSTM + Bahdanau Attention(加性注意力)
======================================================
在 Bi-LSTM 输出序列上叠加可训练的对齐打分网络(Bahdanau 加性注意力):
  score(t) = v^T · tanh(W · h_t + b)     —— 每个位置的打分
  alpha    = softmax(score)              —— 归一化注意力权重
  context  = Σ alpha_t · h_t             —— 加权求和得到句向量
  logits   = FC(context)

关键实现细节:
  - 【PAD 位置必须 mask】:softmax 之前把 pad 位置的打分置为 -inf,
    否则注意力权重会被大量填充位稀释,模型学不到有意义的对齐;
  - forward 同时返回 logits 与注意力权重 alpha,供 attention_viz.py 画热力图
    和 Web 应用做关键词高亮。
"""

import torch
import torch.nn as nn


class AttnBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_layers: int = 1,
        pretrained_embedding: torch.Tensor = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(pretrained_embedding)
            if freeze_embedding:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        feat_dim = 2 * hidden_dim  # 双向拼接维度

        # ---- Bahdanau 加性注意力打分网络:score = v^T tanh(W h + b) ----
        self.attn_W = nn.Linear(feat_dim, feat_dim, bias=True)  # 对齐网络 W、b
        self.attn_v = nn.Linear(feat_dim, 1, bias=False)  # 打分向量 v(映射到标量)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """x: [B, T] → (logits [B, num_classes], attn_weights [B, T])"""
        mask = x != 0  # [B, T] 有效词位置
        lengths = mask.sum(dim=1)  # 每条样本的有效长度

        emb = self.embedding(x)  # [B, T, D]
        outputs, _ = self.lstm(emb)  # [B, T, 2H]

        # ---- 计算每个位置的注意力打分 ----
        scores = self.attn_v(torch.tanh(self.attn_W(outputs))).squeeze(-1)  # [B, T]

        # ---- PAD 位置 mask:softmax 前置 -inf,使其权重恒为 0(防填充稀释)----
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]
        # 数值安全:若整条全 PAD(理论不会出现),softmax 会得到 NaN,做保护
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # ---- 加权求和得到句向量 ----
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # [B, 2H]

        logits = self.fc(self.dropout(context))
        return logits, attn_weights
