# -*- coding: utf-8 -*-
"""
推理封装(供 Streamlit Web 应用调用)
=====================================
SentimentPredictor:
  - 加载指定 run 的最佳检查点与词汇表(自动识别二分类/五级分类 ckpt);
  - predict(text) 返回:情感极性、置信度、逐词注意力权重(注意力模型);
  - 二分类:极性为 正面/负面,probs 键为 {负面, 正面}(行为不变);
  - 五级分类:极性为 非常负面/负面/中性/正面/非常正面,probs 键为五个类名。

一致性保证(重点):
  推理预处理【直接调用 src/preprocess.py 的 tokenize/encode_tokens】,
  与训练用同一份清洗/分词/停用词/否定词合并/词汇表代码路径,
  绝不复制第二份逻辑,这是保证线上预测与离线训练指标一致的关键。
"""

import json
import os

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


class SentimentPredictor:
    def __init__(self, run_name: str = None, device=None):
        from evaluate import load_checkpoint_for_eval

        # 未指定 run 时:优先 best_model.pt 对应记录,否则取指标最好的 attn/bilstm 检查点
        if run_name is None:
            best_log = os.path.join(OUT_DIR, "best_model_info.json")
            if os.path.exists(best_log):
                with open(best_log, encoding="utf-8") as f:
                    run_name = json.load(f)["run_name"]
            else:
                cands = [p[:-len("_best.pt")] for p in os.listdir(OUT_DIR) if p.endswith("_best.pt")]
                if not cands:
                    raise FileNotFoundError("outputs/ 下没有任何模型检查点,请先训练")
                run_name = sorted(cands)[0]
        self.run_name = run_name

        # 复用 evaluate 模块的检查点加载(同一份重建逻辑;按 ckpt 自动识别任务)
        self.model, self.ckpt, self.vocab, self.device = load_checkpoint_for_eval(run_name)
        self.num_classes = self.ckpt["config"].get("num_classes", 2)  # 旧 ckpt 默认二分类
        self.is_attn = self.ckpt["config"].get("model") == "attn_bilstm"
        # 方案A:BERT ckpt(config.model_type='bert');tokenizer 目录随 ckpt config 记录
        self.is_bert = self.ckpt["config"].get("model_type") == "bert"
        self.maxlen = self.ckpt["config"].get("maxlen", 128)
        self._tokenizer = None

    @property
    def tokenizer(self):
        """BERT 分词器(懒加载,仅 BERT ckpt 使用;目录来自 ckpt config,本地加载)。"""
        if self._tokenizer is None:
            from models.bert import resolve_bert_dir
            from transformers import AutoTokenizer

            d = self.ckpt["config"].get("tokenizer_dir") or self.ckpt["config"].get("model_path")
            self._tokenizer = AutoTokenizer.from_pretrained(resolve_bert_dir(d), local_files_only=True)
        return self._tokenizer

    def _probs_to_result(self, probs: "np.ndarray", words=None) -> dict:
        """概率数组 → 统一返回结构(tokens 传入时与词一一对应)。"""
        label = int(np.argmax(probs))
        if self.num_classes == 5:
            from preprocess_5class import CLASS_NAMES

            probs_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
            polarity = CLASS_NAMES[label]
        else:
            probs_dict = {"负面": float(probs[0]), "正面": float(probs[1])}
            polarity = "正面" if label == 1 else "负面"
        return {
            "label": label,
            "polarity": polarity,
            "num_classes": self.num_classes,
            "confidence": float(probs[label]),
            "probs": probs_dict,
            "tokens": words,
            "attn_weights": None,  # BERT/普通 Bi-LSTM 无逐词注意力
        }

    def predict(self, text: str) -> dict:
        """单条文本推理。返回 label/prob/confidence/tokens/attn_weights。"""
        if self.is_bert:
            enc = self.tokenizer(str(text), truncation=True, max_length=self.maxlen,
                                 padding="max_length", return_tensors="pt")
            x = enc["input_ids"].to(self.device)
            with torch.no_grad():
                logits = self.model(x)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            n_valid = int(enc["attention_mask"][0].sum())
            words = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0][:n_valid].tolist())
            return self._probs_to_result(probs, words)

        from preprocess import MAXLEN, encode_tokens, tokenize

        # 关键:与训练完全一致的预处理路径
        words = tokenize(text)[:MAXLEN]  # 超过 maxlen 的词截断,保证 tokens 与注意力权重等长
        ids = encode_tokens(words, self.vocab)  # 定长 128,pad=0
        x = torch.tensor(ids[None, :]).to(self.device)

        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, tuple):  # 注意力模型:同时拿到权重
                logits, attn = out
                weights = attn[0].cpu().numpy()
            else:
                logits, weights = out, None

            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        label = int(np.argmax(probs))
        # 极性与概率字典按任务类型映射(二分类键名保持 负面/正面 不变,零回归)
        if self.num_classes == 5:
            from preprocess_5class import CLASS_NAMES

            probs_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
            polarity = CLASS_NAMES[label]
        else:
            probs_dict = {"负面": float(probs[0]), "正面": float(probs[1])}
            polarity = "正面" if label == 1 else "负面"
        return {
            "label": label,
            "polarity": polarity,
            "num_classes": self.num_classes,
            "confidence": float(probs[label]),  # 预测类别的置信度
            "probs": probs_dict,
            "tokens": words,
            # 截取有效长度内的注意力权重(与词一一对应)
            "attn_weights": weights[: len(words)].tolist() if weights is not None else None,
        }

    def predict_batch(self, texts: list, batch_size: int = 256):
        """批量推理(Bi-LSTM 与 BERT 统一入口),返回 (labels, confs)。

        预处理与 predict() 单条完全同路径(分词/编码或 tokenizer),
        供外部基准批量评估使用,绝不另写第二份逻辑。
        """
        if self.is_bert:
            labels, confs = [], []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    enc = self.tokenizer([str(t) for t in texts[i:i + batch_size]],
                                         truncation=True, max_length=self.maxlen,
                                         padding="max_length", return_tensors="pt")
                    x = enc["input_ids"].to(self.device)
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    labels.extend(probs.argmax(axis=1).tolist())
                    confs.extend(probs.max(axis=1).tolist())
            return labels, confs

        from preprocess import MAXLEN, encode_tokens, tokenize

        token_lists = [tokenize(t)[:MAXLEN] for t in texts]  # 与 predict() 相同截断
        labels, confs = [], []
        with torch.no_grad():
            for i in range(0, len(token_lists), batch_size):
                ids = np.stack([encode_tokens(ts, self.vocab)
                                for ts in token_lists[i:i + batch_size]])
                x = torch.from_numpy(ids).to(self.device)
                out = self.model(x)
                logits = out[0] if isinstance(out, tuple) else out  # 注意力模型返回 (logits, alpha)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                labels.extend(probs.argmax(axis=1).tolist())
                confs.extend(probs.max(axis=1).tolist())
        return labels, confs


if __name__ == "__main__":
    # 简单自测:python predict.py [run_name]
    import sys

    p = SentimentPredictor(sys.argv[1] if len(sys.argv) > 1 else None)
    for t in ("这家酒店位置很好,服务态度也不错,下次还会来。",
              "房间太脏了,空调也是坏的,非常失望,不推荐。",
              "屏幕细腻,运行流畅,就是电池续航一般。"):
        r = p.predict(t)
        print(f"{r['polarity']}({r['confidence']:.2%}) ← {t}")
