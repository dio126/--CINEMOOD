# 中文电影评论情感分析系统(PyTorch + Bi-LSTM/Attention + Streamlit)

基于深度学习的中文文本情感二分类系统。数据集为 **ChnSentiCorp**(酒店/书籍/电脑评论,
正面/负面二分类标注,本工程合并官方 train+dev 共 10800 条)。两项进阶均已实现:
**进阶 A:Bi-LSTM + Bahdanau Attention(含词级注意力热力图)**、
**进阶 B:腾讯 AI Lab 预训练词向量(冻结/微调三组对比)**。

- 训练设备:本机 NVIDIA GeForce RTX 4060 Laptop GPU(CUDA 自动检测,无 GPU 自动回退 CPU)
- **最佳模型:test_acc 0.9019 / test_F1 0.9022**(`outputs/best_model.pt`)

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11(本机 Windows 11 + Git Bash 实测) |
| Python | ≥ 3.9(本机 3.13.2) |
| GPU(可选) | NVIDIA 显卡 + 驱动支持 CUDA 12.x(自动检测,无则 CPU 训练) |

## 2. 安装步骤

以下命令均在项目根目录 `movie-sentiment/` 下执行(Git Bash)。

```bash
# 1) 创建虚拟环境
python -m venv .venv

# 2) 安装 torch(二选一)
#    有 NVIDIA 显卡(本机实测命令;注意:torch 2.11.0+cu128 在本机无法加载,
#    c10.dll 初始化失败 WinError 1114,固定使用 2.7.1+cu126):
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
#    无显卡(CPU 版):
.venv/Scripts/python.exe -m pip install torch==2.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 安装其余依赖(清华镜像)
.venv/Scripts/python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 目录结构

```
movie-sentiment/
├── data/raw/                # ChnSentiCorp 原始 CSV + 停用词表(.gitignore 忽略)
├── data/processed/          # train/val/test.pkl、vocab.json、embedding_matrix.npy
├── src/
│   ├── download_data.py     # 数据集获取(HF→镜像→GitHub 三级降级)
│   ├── download_wordvec.py  # 预训练词向量获取与精简(腾讯/HF 镜像→fastText→本地 w2v)
│   ├── eda.py               # 数据统计与长度分布图
│   ├── preprocess.py        # 清洗/分词/停用词/词汇表/序列化(训练推理共用,防不一致)
│   ├── dataset.py           # PyTorch Dataset 与 DataLoader(num_workers=0)
│   ├── models/bilstm.py     # 基础 Bi-LSTM(mask 平均池化)
│   ├── models/attn_bilstm.py# Bi-LSTM + Bahdanau Attention(PAD mask)
│   ├── train.py             # 训练循环+早停+曲线记录+自动评估+追加总表
│   ├── evaluate.py          # 指标计算 + 混淆矩阵(单一评估代码路径)
│   ├── attention_viz.py     # 词级注意力热力图
│   └── predict.py           # 推理封装(供 Web 调用,含注意力权重)
├── app/app.py               # Streamlit 三页面 Web 应用
├── outputs/                 # 曲线/混淆矩阵/热力图/对比表/过拟合分析/模型 ckpt
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. 完整复现命令(按序执行)

```bash
PY=.venv/Scripts/python.exe

# 阶段1:数据获取(HuggingFace 直连失败自动切 hf-mirror 镜像)+ 停用词 + EDA + 预处理
cd src
$PY download_data.py      # → data/raw/chnsenticorp.csv(10800 条)+ hit_stopwords.txt
$PY eda.py                # → outputs/eda_length_dist.png
$PY preprocess.py         # → 8640/1080/1080 划分 + vocab.json(33037 词)

# 阶段4前置:预训练词向量(腾讯 AI Lab 200 维精简版,HF 镜像流式下载)
$PY download_wordvec.py   # → data/processed/embedding_matrix.npy(词表命中 67.2%)

# 阶段2:Bi-LSTM 超参网格(lr ∈ {1e-3,5e-4,1e-4} × dropout ∈ {0.3,0.5})
$PY train.py --model bilstm --embedding random --lr 1e-3 --dropout 0.5 --run_name bilstm_lr1e-3_do0.5
#   …… 其余 5 组同理(lr/dropout 换值),每组自动产出 curves/metrics/confusion 并进总表

# 阶段3:注意力模型对比(与最优基线同超参)
$PY train.py --model attn_bilstm --embedding random --lr 1e-4 --dropout 0.3 --run_name attn_lr1e-4_do0.3
$PY train.py --model attn_bilstm --embedding random --lr 1e-4 --dropout 0.5 --run_name attn_lr1e-4_do0.5
$PY attention_viz.py attn_lr1e-4_do0.3     # → outputs/attention_1~5.png 词级热力图

# 阶段4:预训练词向量三组对比(random 基线已在阶段2)
$PY train.py --model bilstm --embedding pretrained        --lr 1e-4 --dropout 0.3 --run_name bilstm_lr1e-4_do0.3_pretrained
$PY train.py --model bilstm --embedding pretrained_finetune --lr 1e-4 --dropout 0.3 --run_name bilstm_lr1e-4_do0.3_pretune
# 进阶A+B 组合
$PY train.py --model attn_bilstm --embedding pretrained_finetune --lr 1e-4 --dropout 0.3 --run_name attn_lr1e-4_do0.3_pretune

# 单独评估某个 run(训练结束已自动评估,此命令可独立复查)
$PY evaluate.py <run_name>

# 阶段5:启动 Web 应用
cd ..
$PY -m streamlit run app/app.py --server.headless true
# 浏览器访问 http://localhost:8501(情感分析/实验仪表盘/批量分析 三页面)
```

## 5. 实验结果摘要(全部为真实训练产出,详见 outputs/experiments_summary.csv)

测试集 1080 条;Precision/Recall/F1 以"正面"为正类。

| run_name | 模型 | 词向量 | lr | dropout | val_acc | test_acc | test_F1 |
|---|---|---|---|---|---|---|---|
| bilstm_lr1e-3_do0.3 | Bi-LSTM | random | 1e-3 | 0.3 | 0.8787 | 0.8611 | 0.8590 |
| bilstm_lr1e-3_do0.5 | Bi-LSTM | random | 1e-3 | 0.5 | 0.8898 | 0.8648 | 0.8623 |
| bilstm_lr5e-4_do0.3 | Bi-LSTM | random | 5e-4 | 0.3 | 0.8926 | 0.8565 | 0.8534 |
| bilstm_lr5e-4_do0.5 | Bi-LSTM | random | 5e-4 | 0.5 | 0.8861 | 0.8639 | 0.8625 |
| bilstm_lr1e-4_do0.3 | Bi-LSTM | random | 1e-4 | 0.3 | 0.8981 | 0.8639 | 0.8648 |
| bilstm_lr1e-4_do0.5 | Bi-LSTM | random | 1e-4 | 0.5 | 0.8889 | 0.8648 | 0.8648 |
| attn_lr1e-4_do0.3 | +Attention | random | 1e-4 | 0.3 | 0.8963 | 0.8556 | 0.8553 |
| attn_lr1e-4_do0.5 | +Attention | random | 1e-4 | 0.5 | 0.8815 | 0.8500 | 0.8494 |
| bilstm_lr1e-4_do0.3_pretrained | Bi-LSTM | 腾讯冻结 | 1e-4 | 0.3 | 0.8685 | 0.8685 | 0.8697 |
| **bilstm_lr1e-4_do0.3_pretune** | **Bi-LSTM** | **腾讯微调** | **1e-4** | **0.3** | **0.9093** | **0.9019** | **0.9022** |
| attn_lr1e-4_do0.3_pretune | +Attention | 腾讯微调 | 1e-4 | 0.3 | 0.8954 | 0.8907 | 0.8878 |

**最佳模型 = `bilstm_lr1e-4_do0.3_pretune`**(test: acc 0.9019 / P 0.8891 / R 0.9157 / F1 0.9022),
已复制为 `outputs/best_model.pt`(对应信息见 `outputs/best_model_info.json`),
Web 应用默认加载该模型。

主要结论:
1. lr 影响最大:1e-4 明显优于 1e-3/5e-4(收敛更稳,过拟合更晚);
2. 预训练词向量**微调**提升显著(test_F1 0.8648 → 0.9022,+3.7pt),冻结版也略优于随机初始化;
3. 随机初始化下注意力模型与基础模型互有胜负,未带来稳定增益(注意力价值更多体现在可解释性,
   见 outputs/attention_*.png 与 Web 端关键词高亮);
4. 过拟合分析(真实曲线背离数据)见 `outputs/overfitting_analysis.md`。

## 6. 注意事项与已知限制

1. **torch 版本坑**:本机实测 `torch 2.11.0+cu128` 无法加载(c10.dll 初始化失败,
   WinError 1114),已固定 `torch==2.7.1+cu126`(requirements.txt 已注明)。
2. **网络降级链(实际发生的情况)**:
   - 数据集:HuggingFace 直连超时 → 自动走 `HF_ENDPOINT=https://hf-mirror.com` 镜像成功
     (lansinuote/ChnSentiCorp,合并官方 train+dev 共 10800 条);
   - 停用词:GitHub raw 与 jsdelivr CDN 均被连接重置(本机网络无法访问 GitHub),
     已按计划降级为代码内置精简中文停用词表(174 词,data/raw/hit_stopwords.txt 文件头注明);
   - 预训练词向量:腾讯官网(23KB/s)与 fastText 直连(185KB/s)在 10 分钟预算内无法完成
     GB 级下载,最终使用 **HF 镜像托管的腾讯 AI Lab 200 维精简版**
     (shibing624/text2vec-word2vec-tencent-chinese 的 light_Tencent_AILab_ChineseEmbedding.bin,
     143613 词 × 200 维,word2vec 二进制格式流式解析,来源记录于
     data/processed/wordvec_source.json),词表命中率 67.2%,未命中词随机初始化 N(0,0.01);
   - gensim 本地 word2vec 兜底未触发(Python 3.13 无对应 wheel,且在线源已成功)。
3. **维度说明**:预训练对比组 embedding 为 200 维(腾讯词向量固有维度),random 基线为 300 维;
   两组 hidden/dropout/lr/batch 完全一致。
4. **推理一致性**:推理与训练共用 `src/preprocess.py` 同一份清洗/分词/停用词/编码代码路径,
   Web 端不做任何复制粘贴的第二套逻辑。
5. Windows 约束:DataLoader 一律 `num_workers=0`;全部 matplotlib 图使用微软雅黑中文字体;
   全局随机种子 42(random/numpy/torch),划分与初始化可复现。
6. 数据为酒店/书籍/电脑评论(ChnSentiCorp),并非严格"电影评论";课程题目以中文评论情感
   分析为准,数据处理与建模流程不受影响。
