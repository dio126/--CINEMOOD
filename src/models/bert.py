# -*- coding: utf-8 -*-
"""
BERT 分类模型封装(方案A:hfl/chinese-roberta-wwm-ext 微调)
=============================================================
设计目标:让 BERT 与现有 Bi-LSTM 共用【同一套评估口径】:
  - 前向接口与 BiLSTMClassifier 对齐:forward(ids: [B, L]) → logits [B, num_classes];
  - input_ids 以 0([PAD])补齐时,attention_mask 由 ids != 0 自动推导,
    因此 dataset/evaluate 的"id 序列 + 标签"DataLoader 循环无需任何改动即可评估 BERT;
  - 不返回注意力权重(与普通 BiLSTMClassifier 一致,predict 里 attn_weights=None)。

本地权重:models/bert-base/(HF 镜像下载,git 忽略),加载用 local_files_only=True,
杜绝运行时静默联网。ckpt 的 config 里 model_type='bert' + model_path 记录来源目录,
推理时按 config 自动重建(见 evaluate.load_checkpoint_for_eval 的 bert 分支)。
"""

import os

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BERT_DIR = os.path.join(PROJECT_ROOT, "models", "bert-base")


def resolve_bert_dir(model_path: str = None) -> str:
    """解析 BERT 本地目录:绝对路径直接用;相对路径按项目根拼接。"""
    p = model_path or DEFAULT_BERT_DIR
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"BERT 本地权重目录不存在:{p}(应从 HF 镜像下载 hfl/chinese-roberta-wwm-ext,"
            f"见 README 方案A章节;禁止使用残缺权重)")
    return p


class BertClassifier(nn.Module):
    """Chinese-RoBERTa-wwm-ext + 分类头,前向接口对齐 Bi-LSTM(输入仅 ids)。"""

    def __init__(self, num_classes: int = 2, model_path: str = None):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        # 展示层微调(仅日志级别,不改任何加载/推理逻辑):在 transformers 导入之后设置,
        # 才不会被其默认配置覆盖;压掉 LOAD REPORT MISSING/UNEXPECTED 等吓人的 WARNING 噪音
        import logging

        logging.getLogger("transformers").setLevel(logging.ERROR)

        path = resolve_bert_dir(model_path)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_classes, local_files_only=True)
        self.num_classes = num_classes
        self.model_path = model_path or "models/bert-base"

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, L] 的 BERT input_ids([PAD]=0);attention_mask 按非零位推导
        attn_mask = (ids != 0).long()
        out = self.bert(input_ids=ids, attention_mask=attn_mask)
        return out.logits  # [B, num_classes],与 BiLSTMClassifier 返回形状一致
