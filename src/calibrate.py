# -*- coding: utf-8 -*-
"""
置信度校准(Temperature Scaling)
================================
动机:外部 30000 条测评发现模型"说大话"——中性/模糊文本的平均置信度 0.75,
与清晰文本几乎相同,置信度数字不可信。
方法:在【验证集】上为每个模型拟合单一温度参数 T(Guo et al. 2017):
  softmax(logits / T),T>1 使概率分布变"软"(置信度诚实),T<1 变"硬"。
  预测(argmax)在 T>0 下数学上不变 —— 只改概率,不改判断。
产物:outputs/<run>_temperature.json(含 T、校准前后 NLL、规模);
src/predict.py 推理时自动读取应用(文件缺失则 T=1,行为与旧版完全一致)。

纪律:只用验证集拟合(测试集/外部基准绝不参与);每个模型独立拟合;
校准后验证集 argmax 必须与校准前 100% 一致(脚本内断言)。

用法:.venv/Scripts/python.exe calibrate.py [run_name ...](缺省=全部模型)
"""

import json
import os
import pickle
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def _discover_runs():
    """与 Web 相同的发现规则:二分类 *_best.pt(排除 _5c_)+ 五级 *_5c_best.pt。"""
    runs = []
    for p in sorted(os.listdir(OUT_DIR)):
        if p.endswith("_5c_best.pt"):
            runs.append(p[:-len("_5c_best.pt")])
        elif p.endswith("_best.pt"):
            runs.append(p[:-len("_best.pt")])
    return runs


def _val_file_for(cfg: dict) -> str:
    """按 ckpt config 推断验证集文件(与训练时同一份数据划分)。"""
    suffix = cfg.get("data_suffix")
    if cfg.get("num_classes") == 5:
        suffix = suffix or "_5c"
    else:
        suffix = suffix or ""
    return os.path.join(PROC_DIR, f"val{suffix}.pkl")


def _collect_logits(predictor, cfg) -> tuple:
    """在验证集上收集 logits 与真值标签(Bi-LSTM 直接用 pkl 中已编码的 ids;
    BERT 用原始文本走 tokenizer —— 均与训练/推理同路径)。"""
    with open(_val_file_for(cfg), "rb") as f:
        data = pickle.load(f)
    labels = torch.tensor(np.asarray(data["labels"]), dtype=torch.long)

    logits_list = []
    B = 256
    if predictor.is_bert:
        texts = [str(t) for t in data["texts"]]
        with torch.no_grad():
            for i in range(0, len(texts), B):
                enc = predictor.tokenizer(texts[i:i + B], truncation=True,
                                          max_length=predictor.maxlen,
                                          padding="max_length", return_tensors="pt")
                out = predictor.model(enc["input_ids"].to(predictor.device))
                logits_list.append(out.cpu())
    else:
        ids = np.asarray(data["ids"])  # 已是训练时的定长编码,直接复用
        with torch.no_grad():
            for i in range(0, len(ids), B):
                x = torch.from_numpy(ids[i:i + B]).to(predictor.device)
                out = predictor.model(x)
                logits_list.append((out[0] if isinstance(out, tuple) else out).cpu())
    return torch.cat(logits_list), labels


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """一维寻优最小化 NLL(scipy 黄金分割;单参数凸问题)。"""
    from scipy.optimize import minimize_scalar

    logit_np = logits.numpy()
    label_np = labels.numpy()

    def nll(t: float) -> float:
        z = logit_np / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(label_np)), label_np].mean())

    raw_pred = logit_np.argmax(axis=1)
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    t = float(res.x)

    cal = (logit_np / t).argmax(axis=1)
    assert (cal == raw_pred).all(), "校准改变了预测(数学上不应发生,中止)"

    # 期望校准误差(ECE,10 桶):校准后应显著下降
    def ece(preds, probs_max):
        bins = np.linspace(0, 1, 11)
        total = len(preds)
        val = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (probs_max > lo) & (probs_max <= hi)
            if m.sum() == 0:
                continue
            acc = (preds[m] == label_np[m]).mean()
            conf = probs_max[m].mean()
            val += m.sum() / total * abs(acc - conf)
        return float(val)

    def max_prob(t):
        z = logit_np / t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    return {
        "temperature": t,
        "val_nll_raw": nll(1.0),
        "val_nll_calibrated": nll(t),
        "val_ece_raw": ece(raw_pred, max_prob(1.0).max(axis=1)),
        "val_ece_calibrated": ece(raw_pred, max_prob(t).max(axis=1)),
        "val_acc": float((raw_pred == label_np).mean()),
        "n_val": int(len(label_np)),
    }


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predict import SentimentPredictor

    targets = sys.argv[1:] or _discover_runs()
    print(f"[calibrate] 共 {len(targets)} 个模型")
    summary = {}
    for i, run in enumerate(targets, 1):
        try:
            predictor = SentimentPredictor(run)
            logits, labels = _collect_logits(predictor, predictor.ckpt["config"])
            report = _fit_temperature(logits, labels)
            out = os.path.join(OUT_DIR, f"{run}_temperature.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            summary[run] = report
            print(f"[{i}/{len(targets)}] {run}: T={report['temperature']:.3f} "
                  f"NLL {report['val_nll_raw']:.4f}→{report['val_nll_calibrated']:.4f} "
                  f"ECE {report['val_ece_raw']:.3f}→{report['val_ece_calibrated']:.3f} "
                  f"(n={report['n_val']}, acc={report['val_acc']:.4f} 不变)")
        except Exception as e:  # noqa: BLE001 — 单模型失败不影响其余
            print(f"[{i}/{len(targets)}] {run}: 失败 {type(e).__name__}: {e}")
    with open(os.path.join(OUT_DIR, "calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[calibrate] 完成,汇总 → outputs/calibration_summary.json")


if __name__ == "__main__":
    main()
