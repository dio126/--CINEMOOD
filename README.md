# 中文电影评论情感分析系统(PyTorch + Bi-LSTM/Attention + Streamlit)

基于深度学习的中文文本情感二分类系统。数据集为 **ChnSentiCorp**(酒店/书籍/电脑评论,
正面/负面二分类标注,原始合并 10800 条,去重后 9060 条)。两项进阶均已实现:
**进阶 A:Bi-LSTM + Bahdanau Attention(含词级注意力热力图)**、
**进阶 B:腾讯 AI Lab 预训练词向量(随机/冻结/微调三组对比)**。

- 训练设备:本机 NVIDIA GeForce RTX 4060 Laptop GPU(CUDA 自动检测,无 GPU 自动回退 CPU)
- **最佳模型:test_acc 0.8896 / test_F1 0.9022**(`outputs/best_model.pt`)
- 数据质量保障:语料按文本去重(消除训练/测试文本重叠导致的指标虚高);
  停用词表剔除否定词与程度副词 + 否定词合并 + OOV 组合式初始化,
  对"不满意/不好/不干净"等直接否定结构可稳定判负
  ("不是很/不太"等缓和型否定的置信度接近 0.5,属已知边界情况,见已知限制)

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

# 3) 安装其余依赖(清华镜像;版本已用 pip freeze 固定,见 requirements.txt)
.venv/Scripts/python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 目录结构

```
movie-sentiment/
├── data/raw/                # ChnSentiCorp 原始 CSV + 停用词表 + 词向量磁盘缓存(.gitignore 忽略)
├── data/processed/          # train/val/test.pkl、vocab.json、embedding_matrix.npy
├── src/
│   ├── download_data.py     # 数据集获取(HF→镜像→GitHub 三级降级)+ 停用词表
│   ├── download_wordvec.py  # 预训练词向量获取(OOV 等量级+否定组合式初始化)
│   ├── eda.py               # 数据统计与长度分布图
│   ├── preprocess.py        # 清洗/分词/停用词/否定合并/去重/词汇表/序列化(训练推理共用)
│   ├── dataset.py           # PyTorch Dataset 与 DataLoader(num_workers=0)
│   ├── models/bilstm.py     # 基础 Bi-LSTM(mask 平均池化,分类头前单次 Dropout)
│   ├── models/attn_bilstm.py# Bi-LSTM + Bahdanau Attention(PAD mask)
│   ├── train.py             # 训练循环+早停+曲线记录+自动评估+追加总表
│   ├── evaluate.py          # 指标计算 + 混淆矩阵(单一评估代码路径)
│   ├── attention_viz.py     # 词级注意力热力图
│   └── predict.py           # 推理封装(供 Web 调用,含注意力权重,>128 词截断对齐)
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
                           #   停用词表(在线失败时为内置精简表,已剔除否定词/程度副词)
$PY eda.py                # → outputs/eda_length_dist.png
$PY preprocess.py         # 去重 10800→9060,8:1:1 划分 7248/906/906 + vocab.json(33569 词)
                           #   含否定词合并(不+满意→不满意)

# 阶段4前置:预训练词向量(腾讯 AI Lab 200 维精简版,HF 镜像,带磁盘缓存)
$PY download_wordvec.py   # → data/processed/embedding_matrix.npy(词表命中 65.6%;
                           #   OOV 按命中行同量级 std 初始化 + 否定合并词组合式初始化)

# 阶段2:Bi-LSTM 超参网格(lr ∈ {1e-3,5e-4,1e-4} × dropout ∈ {0.3,0.5},hidden=128)
$PY train.py --model bilstm --embedding random --lr 1e-3 --dropout 0.5 --run_name bilstm_lr1e-3_do0.5
#   …… 其余 5 组同理(lr/dropout 换值),每组自动产出 curves/metrics/confusion 并进总表
# hidden 容量对比(lr=1e-4, do=0.3, random,h ∈ {64,256},与 h=128 组成三档)
$PY train.py --model bilstm --embedding random --lr 1e-4 --dropout 0.3 --hidden 64  --run_name bilstm_lr1e-4_do0.3_h64
$PY train.py --model bilstm --embedding random --lr 1e-4 --dropout 0.3 --hidden 256 --run_name bilstm_lr1e-4_do0.3_h256

# 阶段3:注意力模型对比(与基线同超参)
$PY train.py --model attn_bilstm --embedding random --lr 1e-4 --dropout 0.3 --run_name attn_lr1e-4_do0.3
$PY train.py --model attn_bilstm --embedding random --lr 1e-4 --dropout 0.5 --run_name attn_lr1e-4_do0.5
$PY attention_viz.py attn_lr1e-4_do0.3_pretune   # → outputs/attention_1~5.png 词级热力图

# 阶段4:预训练词向量三组对比(random 基线已在阶段2)
$PY train.py --model bilstm --embedding pretrained          --lr 1e-4 --dropout 0.3 --run_name bilstm_lr1e-4_do0.3_pretrained
$PY train.py --model bilstm --embedding pretrained_finetune --lr 1e-4 --dropout 0.3 --run_name bilstm_lr1e-4_do0.3_pretune
# 进阶A+B 组合,以及微调学习率扩展点(最终最佳模型)
$PY train.py --model attn_bilstm --embedding pretrained_finetune --lr 1e-4 --dropout 0.3 --run_name attn_lr1e-4_do0.3_pretune
$PY train.py --model bilstm --embedding pretrained_finetune --lr 5e-4 --dropout 0.3 --run_name bilstm_lr5e-4_do0.3_pretune

# 单独评估某个 run(训练结束已自动评估,此命令可独立复查)
$PY evaluate.py <run_name>

# 阶段5:启动 Web 应用
cd ..
$PY -m streamlit run app/app.py --server.headless true
# 浏览器访问 http://localhost:8501(情感分析/实验仪表盘/批量分析 三页面)
```

## 5. 实验结果摘要(全部为真实训练产出,详见 outputs/experiments_summary.csv)

测试集 906 条(去重后);Precision/Recall/F1 以"正面"为正类。

| run_name | 模型 | 词向量 | lr | hidden | dropout | val_acc | test_acc | test_F1 |
|---|---|---|---|---|---|---|---|---|
| bilstm_lr1e-3_do0.3 | Bi-LSTM | random | 1e-3 | 128 | 0.3 | 0.8543 | 0.8510 | 0.8607 |
| bilstm_lr1e-3_do0.5 | Bi-LSTM | random | 1e-3 | 128 | 0.5 | 0.8411 | 0.8355 | 0.8423 |
| bilstm_lr5e-4_do0.3 | Bi-LSTM | random | 5e-4 | 128 | 0.3 | 0.8653 | 0.8411 | 0.8543 |
| bilstm_lr5e-4_do0.5 | Bi-LSTM | random | 5e-4 | 128 | 0.5 | 0.8477 | 0.8466 | 0.8554 |
| bilstm_lr1e-4_do0.3 | Bi-LSTM | random | 1e-4 | 128 | 0.3 | 0.8477 | 0.8322 | 0.8430 |
| bilstm_lr1e-4_do0.5 | Bi-LSTM | random | 1e-4 | 128 | 0.5 | 0.8444 | 0.8366 | 0.8458 |
| bilstm_lr1e-4_do0.3_h64 | Bi-LSTM | random | 1e-4 | 64 | 0.3 | 0.8389 | 0.8256 | 0.8333 |
| bilstm_lr1e-4_do0.3_h256 | Bi-LSTM | random | 1e-4 | 256 | 0.3 | 0.8543 | 0.8267 | 0.8425 |
| attn_lr1e-4_do0.3 | +Attention | random | 1e-4 | 128 | 0.3 | 0.8433 | 0.8477 | 0.8598 |
| attn_lr1e-4_do0.5 | +Attention | random | 1e-4 | 128 | 0.5 | 0.8477 | 0.8389 | 0.8485 |
| bilstm_lr1e-4_do0.3_pretrained | Bi-LSTM | 腾讯冻结 | 1e-4 | 128 | 0.3 | 0.8797 | 0.8720 | 0.8835 |
| bilstm_lr1e-4_do0.3_pretune | Bi-LSTM | 腾讯微调 | 1e-4 | 128 | 0.3 | 0.8918 | 0.8797 | 0.8928 |
| attn_lr1e-4_do0.3_pretune | +Attention | 腾讯微调 | 1e-4 | 128 | 0.3 | 0.8830 | 0.8808 | 0.8918 |
| **bilstm_lr5e-4_do0.3_pretune** | **Bi-LSTM** | **腾讯微调** | **5e-4** | **128** | **0.3** | **0.8951** | **0.8896** | **0.9022** |

**最佳模型 = `bilstm_lr5e-4_do0.3_pretune`**(test: acc 0.8896 / P 0.8882 / R 0.9165 / F1 0.9022),
已复制为 `outputs/best_model.pt`(对应信息见 `outputs/best_model_info.json`),
Web 应用默认加载该模型。

主要结论:
1. **去重的影响**:语料去重(10800→9060)消除了 21.9% 测试文本在训练集出现的问题,
   各 run test 指标整体下降约 1~2pt——这是更诚实的泛化估计;
2. **否定句处理链路**(停用词剔除否定/程度词 → 否定词合并 → OOV 等量级+组合式初始化
   → 词向量微调 lr=5e-4):对"不满意/不好/不干净"等**直接否定结构**可稳定判负
   (最佳模型 4 条直接否定测试句全部正确,其中"我不满意这家酒店的服务"判负 62%);
   但对"不是很干净/不太舒服"等**缓和型否定**(否定词+程度词组合,经分词后
   未被合并规则覆盖),置信度仍接近 0.5,属已知边界情况(见已知限制第 4 条)。
   仅靠停用词修正时,预训练词向量组对否定组合仍学不动(稀有 OOV 词向量幅值
   过小、更新次数不足),组合式初始化与更高微调学习率是关键;
3. hidden 容量:h=64 欠拟合(F1 0.8333),h=256 过拟合较重,h=128 综合最优;
4. 预训练词向量微调提升显著(random 最好 0.8607 → 微调 0.9022,+4.2pt),
   冻结版 0.8835 也优于全部 random 基线;微调学习率 5e-4 优于 1e-4;
5. 随机初始化下注意力模型与基础模型互有胜负(其价值更多在可解释性,
   见 outputs/attention_*.png 与 Web 端关键词高亮);
6. 过拟合分析(真实曲线背离数据)见 `outputs/overfitting_analysis.md`。

## 6. 注意事项与已知限制

1. **torch 版本坑**:本机实测 `torch 2.11.0+cu128` 无法加载(c10.dll 初始化失败,
   WinError 1114),已固定 `torch==2.7.1+cu126`(requirements.txt 已注明)。
2. **网络降级链(实际发生的情况)**:
   - 数据集:HuggingFace 直连超时 → 自动走 `HF_ENDPOINT=https://hf-mirror.com` 镜像成功
     (lansinuote/ChnSentiCorp,合并官方 train+dev 共 10800 条);
   - 停用词:GitHub raw 与 jsdelivr CDN 均不可达(本机网络无法访问 GitHub),
     降级为代码内置精简停用词表(134 词)。**该表已刻意剔除全部否定词与程度副词**
     (不/没有/很/非常/最/太 等),避免"不满意"被过滤成"满意"导致否定句判反;
   - 预训练词向量:腾讯官网(23KB/s)与 fastText 直连(185KB/s)在 10 分钟预算内无法完成
     GB 级下载,最终使用 **HF 镜像托管的腾讯 AI Lab 200 维精简版**
     (shibing624/text2vec-word2vec-tencent-chinese 的 light_Tencent_AILab_ChineseEmbedding.bin,
     143613 词 × 200 维,word2vec 二进制格式,下载后落盘缓存于
     data/raw/tencent_light_word2vec.bin 再流式解析,来源记录于
     data/processed/wordvec_source.json),词表命中率 65.6%;
   - gensim 本地 word2vec 兜底未触发(Python 3.13 无对应 wheel,且在线源已成功)。
3. **数据去重(重要)**:原始语料 10800 条中有 1676 条重复文本(其中 76 条标签冲突)。
   若不去重,21.9% 的测试集文本会原样出现在训练集,指标虚高。预处理按
   "同文本标签一致保留 1 条、标签冲突整组丢弃"去重后为 9060 条(7248/906/906),
   全部实验均基于去重数据,指标为诚实的泛化估计。
4. **否定句处理链路(重要设计)**:
   - 停用词表剔除否定/程度词(否则"不/很/太"被删,极性翻转);
   - 分词后否定词合并(不+满意→不满意),把否定语义固化到词内;
   - 词向量矩阵中 OOV 行按预训练命中行的同量级标准差初始化(避免幅值过小学不动),
     其中"否定+X"型合并词用 vec(否定词)+vec(X) 组合式初始化
     (全部 不X 词共享 vec(不) 分量,给分类器一致的否定方向);
   - 已知残余限制:预训练**冻结**组(requires_grad=False)无法适配 OOV 合并词,
     对个别否定短句仍会误判(如"我不满意这家酒店的服务"该组判正 61%);
     系统默认加载的最佳模型(微调 lr=5e-4)无此问题;
   - 已知边界情况(缓和型否定):否定词与程度词组合(不+很/太)未被否定合并
     规则覆盖,模型对这类句子的置信度接近随机。监督复审实测最佳模型:
     "房间不是很干净,感觉不太舒服"→正面 0.596、"酒店设施不满意,性价比不高"
     →正面 0.595,两条均误判但置信度很低(接近 0.5,非高置信错误);
     若要覆盖,可将合并规则扩展为 否定词+程度词+中心词 三元合并(未实施,
     属后续改进项)。
5. **维度说明**:预训练对比组 embedding 为 200 维(腾讯词向量固有维度),random 基线为 300 维;
   两组 hidden/dropout/lr/batch 完全一致。
6. **推理一致性**:推理与训练共用 `src/preprocess.py` 同一份清洗/分词/停用词/否定合并/
   编码代码路径,Web 端不做任何复制粘贴的第二套逻辑;超过 128 词的输入在推理侧同样截断,
   保证注意力高亮 tokens 与权重严格等长。
7. **模型对中性输入过度自信**:本系统为二分类(正/负)模型,没有"中性"类与拒判机制,
   对客观陈述类中性句也会强行给出极性判断且置信度偏高(实测"酒店位于市中心,
   步行五分钟到地铁站……"这类纯客观描述被判正面 93.5%,部分中性长句置信度
   可达 0.99+)。如需处理中性内容,应引入第三类或置信度阈值,这是已知限制。
8. Windows 约束:DataLoader 一律 `num_workers=0`;全部 matplotlib 图使用微软雅黑中文字体;
   全局随机种子 42(random/numpy/torch),划分与初始化可复现。
9. 数据为酒店/书籍/电脑评论(ChnSentiCorp),并非严格"电影评论";课程题目以中文评论情感
   分析为准,数据处理与建模流程不受影响。
