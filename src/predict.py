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
        self.is_attn = self.ckpt["config"]["model"] == "attn_bilstm"

    def predict(self, text: str) -> dict:
        """单条文本推理。返回 label/prob/confidence/tokens/attn_weights。"""
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


if __name__ == "__main__":
    # 简单自测:python predict.py [run_name]
    import sys

    p = SentimentPredictor(sys.argv[1] if len(sys.argv) > 1 else None)
    for t in ("这家酒店位置很好,服务态度也不错,下次还会来。",
              "房间太脏了,空调也是坏的,非常失望,不推荐。",
              "屏幕细腻,运行流畅,就是电池续航一般。"):
        r = p.predict(t)
        print(f"{r['polarity']}({r['confidence']:.2%}) ← {t}")
