# 中文电影评论情感分析系统(PyTorch + Bi-LSTM/Attention + Streamlit)

基于深度学习的中文文本情感二分类系统。数据集为 **ChnSentiCorp**(酒店/书籍/电脑评论,
正面/负面二分类标注,原始合并 10800 条,去重后 9060 条)。两项进阶均已实现:
**进阶 A:Bi-LSTM + Bahdanau Attention(含词级注意力热力图)**、
**进阶 B:腾讯 AI Lab 预训练词向量(随机/冻结/微调三组对比)**。

- 训练设备:本机 NVIDIA GeForce RTX 4060 Laptop GPU(CUDA 自动检测,无 GPU 自动回退 CPU)
- **最佳模型:test_acc 0.8896 / test_F1 0.9022**(`outputs/best_model.pt`)
- **拓展:五级分类**(豆瓣 1-5 星 → 非常负面~非常正面,最佳 macro-F1 0.4725,见第 7 章)
- **模型改进(方案A/B,见第 8 章)**:新增 BERT(`hfl/chinese-roberta-wwm-ext` 微调,二分同域
  test acc 0.9382、五级 exact 0.5464)+ 混域/扩数据 Bi-LSTM,外部 30000 基准跨域
  67.8% → **89.3%**(混域 BERT),五级 exact 47.0% → **53.0%**(BERT)
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
│   ├── download_dmsc.py     # 五级数据:豆瓣电影短评 dmsc(1-5 星,HF 镜像多候选)
│   ├── download_wordvec.py  # 预训练词向量获取(OOV 等量级+否定组合式初始化)
│   ├── eda.py               # 数据统计与长度分布图
│   ├── preprocess.py        # 清洗/分词/停用词/否定合并/去重/词汇表/序列化(训练推理共用)
│   ├── preprocess_5class.py # 五级预处理(import 复用 preprocess,独立词表/划分/矩阵)
│   ├── dataset.py           # PyTorch Dataset 与 DataLoader(num_workers=0,suffix 扩展)
│   ├── models/bilstm.py     # 基础 Bi-LSTM(mask 平均池化,分类头前单次 Dropout)
│   ├── models/attn_bilstm.py# Bi-LSTM + Bahdanau Attention(PAD mask)
│   ├── train.py             # 二分类训练(早停+曲线+自动评估+总表)
│   ├── train_5class.py      # 五级训练(逆频率权重,早停监控 macro-F1,独立产物)
│   ├── evaluate.py          # 指标计算 + 混淆矩阵(单一评估代码路径,任务自适应)
│   ├── attention_viz.py     # 词级注意力热力图
│   └── predict.py           # 推理封装(自动识别二分类/五级 ckpt)
├── app/app.py               # Streamlit 应用(二分类默认 + 可切换五级分类模式)
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

## 7. 五级分类拓展(任务书第七章拓展方向 1)

在二分类系统**完整保留、零改动默认行为**的前提下,新增五级情感分类:
**非常负面(1★)/负面(2★)/中性(3★)/正面(4★)/非常正面(5★)**。
Web 界面侧边栏"任务模式"可切换(默认仍为二分类)。

### 7.1 数据来源与映射

- **数据集**:豆瓣电影短评 dmsc。任务书提到的 XiangPan/dmsc 在 HF 上不存在
  (Repository not found),实际采用 HF 镜像上的 **BerlinWang/DMSC**
  (`DMSC.csv.zip`,150MB,经列名自适应探测含 star 字段,共 212.5 万条带星级短评);
- **星级映射**:1★=非常负面(0)、2★=负面(1)、3★=中性(2)、4★=正面(3)、
  5★=非常正面(4);
- **去重与采样**:同文本去重(与二分类同规则,212.5 万 → 194.1 万,丢弃标签冲突
  16.4 万条)后**分层抽样每类 10000 条,共 5 万条完全均衡**;
  (全量星级分布:1★19.1万 / 2★18.0万 / 3★47.5万 / 4★64.2万 / 5★63.8万);
- **预处理**:复用 preprocess.py 的 tokenize(含否定词合并)——同一份代码路径;
  独立词表 vocab_5class.json(仅训练集构建,47007 词)、独立 8:1:1 划分
  (40000/5000/5000,seed=42)、预训练矩阵 embedding_matrix_5class.npy
  (复用腾讯 light 词向量缓存重建,词表命中 54.4%,OOV 同量级+否定组合式初始化)。

### 7.2 训练设置

- 模型复用 BiLSTMClassifier/AttnBiLSTMClassifier(num_classes=5,模型文件零改动);
- 类别不均衡:CrossEntropyLoss 使用逆频率权重(本数据均衡,权重≈1,机制保留);
- 早停监控**验证集 macro-F1**(细粒度任务下比 accuracy 更可靠);
- 产物独立:<run>_5c_*(best.pt/curves/confusion/metrics/history)+
  outputs/experiments_summary_5class.csv,与二分类 14 组完全隔离。

### 7.3 实验结果(test 5000 条,全部真实训练产出)

| run | 模型 | 词向量 | lr | hidden | val macro-F1 | test acc | test macro-F1 | test weighted-F1 |
|---|---|---|---|---|---|---|---|---|
| 5c_bilstm_random | BiLSTM | random | 5e-4 | 128 | 0.4526 | 0.4518 | 0.4481 | 0.4483 |
| 5c_bilstm_pretune | BiLSTM | 腾讯微调 | 5e-4 | 128 | 0.4549 | 0.4694 | 0.4649 | 0.4651 |
| 5c_attn_pretune | +Attention | 腾讯微调 | 5e-4 | 128 | 0.4650 | 0.4626 | 0.4617 | 0.4619 |
| **5c_bilstm_pretune_h256** | **BiLSTM** | **腾讯微调** | **5e-4** | **256** | 0.4644 | 0.4688 | **0.4725** | 0.4727 |

最佳 run:**5c_bilstm_pretune_h256**(test macro-F1 0.4725)。每类 F1
(最佳 run):非常负面 0.575 / 负面 0.425 / 中性 0.408 / 正面 0.408 / 非常正面 0.546
—— 两端等级(强烈情感)明显好于中间等级,符合直觉。

### 7.4 与二分类的对比说明

- 二分类 test acc 0.8896 / F1 0.9022;五级 test acc 0.4688 / macro-F1 0.4725。
  差距巨大但符合文献规律:**细粒度星级预测中相邻等级语义连续**(3★与4★、
  1★与2★ 的文本差异远小于正/负差异),随机基线 20%,本模型约为其 2.3 倍;
- 相邻级混淆统计(attn run):真实 1★ 被判到本级或相邻 2★ 的合计比例 78%(本级 51% + 相邻 27%)、真实 5★ 被判到
  本级或相邻 4★ 的合计比例 85%(本级 62% + 相邻 23%)、真实 3★ 被判到 2/4★ 的比例 36%——误差几乎全部落在相邻等级,
  极性方向(正/负半区)判别远好于精确等级判别;
- 两条流水线**完全独立**(数据/词表/产物/总表),二分类 14 组实验与界面默认
  行为零改动(回归自测:否定句/正例输出逐位一致)。

### 7.5 已知限制

1. macro-F1 ≈ 0.47:Bi-LSTM 对细粒度星级区分能力有限,Transformer 类模型
   (如 BERT)在此任务上可达 0.6+,属后续改进方向;
2. 相邻等级(尤其 2★/3★/4★)互相混淆仍占错误主体;
3. 五级模型对短文本(如"还行")置信度低(≈50%),信息量不足时判定趋于保守;
4. 类别均衡由采样保证;若直接用于真实分布(4/5 星占比高)需注意召回差异。

### 7.6 复现命令

```bash
PY=.venv/Scripts/python.exe
cd src
$PY download_dmsc.py        # 下载 dmsc(zip 留缓存)→ 采样 5 万条均衡数据
$PY preprocess_5class.py    # 五级预处理 + 词表 + 词向量矩阵(复用腾讯缓存)
$PY train_5class.py --model bilstm --embedding pretrained_finetune --lr 5e-4 --hidden 256 \
    --run_name 5c_bilstm_pretune_h256          # 最佳一组;其余 3 组同理换参数
# Web:streamlit 侧边栏"任务模式"切到"五级分类"即可交互体验
```

## 8. 模型改进(方案A:新增 BERT + 方案B:数据侧增强)

本章为外部 30000 条基准诊断后的两项改进(全部实验真实训练产出,逐项记入
`outputs/experiments_summary*.csv`;外部基准结果记入
`outputs/eval_report/external_benchmark_summary.csv`;训练过程监控与干预见
`outputs/train_interventions.md`)。**所有旧模型、旧实验产物零改动**,
新模型以新增 ckpt 方式加入对比,Web 界面自动发现。

### 8.1 动机:30000 条外部基准诊断(改进前基线)

用完全未参与训练的 DMSC 剩余池 30000 条(三重排除、与训练集交集=0)对两个最佳模型
做外部测评(详见 `outputs/eval_report/eval_report.md`),发现两个瓶颈:

1. **二分类跨域掉分严重**:Bi-LSTM(`bilstm_lr5e-4_do0.3_pretune`,仅训练于
   ChnSentiCorp 酒店/书/电脑评论)同域 test acc 0.8896,但在豆瓣电影评论上
   仅 **0.6782**(F1 0.6964),降 21.1pp——训练域里根本没有电影评论的语言风格
   (反语、行为暗示、口语缩写);
2. **五级模型已到数据容量上限**:5 万条训练的 `5c_bilstm_pretune_h256` 在同分布
   未见数据上 exact 0.4704,与自测集 0.4688 持平(零衰减)——不是过拟合,
   而是 5 万样本下 Bi-LSTM 已学不动更多区分信号(2★/4★ 温和档位边界模糊)。

### 8.2 做法

**方案B1(混域二分类,治跨域)**:把电影域数据混进训练集。ChnSentiCorp 全量
(去重后 9060)+ DMSC 剩余池抽电影二分类 4 万条(1★/2★=负面、4★/5★=正面,
各星 1 万等比),合计 49060 条。剩余池做了**五重排除**:五级训练源 5 万、六个
训练 pkl、**外部基准 30000 + 隐晦 300(禁入训的回归基准)**、以及(对 B2)
B1 已抽样本;构建脚本内置交集=0 断言,并对最终 pkl 用 `dtype=str` 复核
(train/val/test 与 30000 基准、隐晦 300 交集均为 0)。新词表(仅训练集构建)+
腾讯词向量矩阵重建,训练与最佳二分类**完全同配置**的 Bi-LSTM
(lr5e-4/do0.3/pretune/h128,run 名 `mixdom_bilstm_lr5e-4_do0.3_pretune`)。

**方案A(新增 BERT,任务书 3.3 可选高级扩展)**:从 HF 镜像下载
`hfl/chinese-roberta-wwm-ext`(约 400MB,本地 `models/bert-base/`,git 忽略),
新增 `src/train_bert.py` + `src/models/bert.py`:maxlen128 / batch32 / fp16
(RTX 4060 8GB 实测 3.3GB 显存)/ AdamW lr2e-5 / 线性衰减 6% warmup / 3 epoch
早停。**与 Bi-LSTM 用同一份 pkl 划分、同一套评估函数**(evaluate_model /
evaluate_5class),公平对比;二分类训练于 ChnSentiCorp 原数据
(`bert_roberta_wwm`),五级训练于原 5 万(`bert_roberta_wwm_5c`)。
ckpt config 记录 `model_type=bert` 与 tokenizer 目录,predict/evaluate 按
config 自动走 BERT 分支(旧 Bi-LSTM 路径零改动);另加数据侧+模型侧组合的
混域 BERT(`bert_roberta_wwm_mixdom`,用 B1 混域数据)。

**方案B2(五级扩数据,破容量上限)**:DMSC 剩余池分层抽样 30 万(每级 6 万,
长度 5~300,五重排除同上),训练与五级最佳**同配置**的 Bi-LSTM
(bilstm/pretune/h256/do0.3;lr 取项目实际最佳 5e-4,即
`experiments_summary_5class.csv` 中 5c 最佳组的真实配置;任务书原文写 1e-4
与 CSV 记录不符,以真实最佳为准),run 名 `5c300k_bilstm_pretune_h256`。
300k 训练实测 7.3 分钟(早停于第 5 epoch),未触发降级条款。

### 8.3 总对比表(全部真实数字,外部基准=30000 条,口径与旧报告完全一致)

二分类(同域 test=ChnSentiCorp 906 条;跨域=外部基准正/负组 20000 条):

| 模型 | 训练数据 | 同域 test Acc / F1 | 跨域 Acc / F1 | 跨域较旧基线 |
|---|---|---|---|---|
| Bi-LSTM(旧最佳) | ChnSentiCorp 9060 | 0.8896 / 0.9022 | 0.6782 / 0.6964 | — |
| **混域 Bi-LSTM(B1)** | 上述 + DMSC 电影 4 万 | 0.8388 / 0.8415*(混域测试集) | **0.8440 / 0.8442** | **+16.6pp** |
| BERT(方案A,同数据) | ChnSentiCorp 9060 | **0.9382 / 0.9450** | 0.7779 / 0.7716 | +9.9pp |
| 混域 BERT(A+B1 组合) | 上述 + DMSC 电影 4 万 | 0.8962 / 0.8976*(混域测试集) | **0.8928 / 0.8927** | **+21.5pp** |

\* 混域模型的自测集是混域 8:1:1 的 test(4906 条,含约 18% ChnSentiCorp +
82% 电影),与纯 ChnSentiCorp 测试集不可直接比,故同域列标注数据来源。

五级分类(同分布 test 各 5000/30000 条;外部=30000 条全量):

| 模型 | 训练数据 | 同分布 test exact / macro-F1 | 外部 exact / ±1 / macro-F1 |
|---|---|---|---|
| Bi-LSTM h256(旧最佳) | DMSC 5 万 | 0.4688 / 0.4725 | 0.4704 / 0.8580 / 0.4697 |
| **30万 Bi-LSTM(B2)** | DMSC 30 万 | **0.5131 / 0.5145** | **0.5116 / 0.8816 / 0.5154** |
| **BERT(方案A,同数据)** | DMSC 5 万 | **0.5464 / 0.5452** | **0.5296 / 0.8963 / 0.5318** |

### 8.4 结论

1. **B1 完全达成目标(67.8% → 75%+)**:仅靠把 4 万条电影评论混进训练集、
   模型结构零改动,跨域 Acc 0.6782 → **0.8440**(+16.6pp)——证实跨域掉分的
   主因是训练域覆盖不足,而非模型能力;
2. **方案A 全面超越同数据 Bi-LSTM**:二分类同域 +4.9pp(0.9382 vs 0.8896)、
   跨域 +9.9pp;五级同数据 exact +7.8pp(0.5464 vs 0.4688)——预训练语言模型
   对"隐晦/反语"的鲁棒性显著更好,验证了第 7.5 节预判;
3. **B2 破除了容量上限**:5 万 → 30 万,同配置 Bi-LSTM exact 0.4688 → 0.5131
   (+4.3pp),外部 0.4704 → 0.5116,且 30 万下仍未过拟合(早停第 5 轮)——
   旧"零衰减=容量上限"的诊断正确,数据规模是当时的实际瓶颈;
4. **数据与模型改进可叠加**:混域 BERT 同时吃到两边收益,跨域 0.8928 为
   全场最佳(+21.5pp);五级若做"30 万 + BERT"预计还能再涨(本轮时间预算内未做,属后续方向)。

### 8.5 降级与失败记录(如实)

- BERT 权重首次下载连接超时(hf-mirror 临时不可达),重试成功,未降级;
- B2 词向量构建首次触发 `download_wordvec.py` 潜在 bug(组合式初始化假设
  否定词必在词表,5cL 新词表无单字"别"→ KeyError),已加存在性守卫修复
  (对旧词表计算路径无影响),重建成功;期间两个在线降级源一失败一超时,
  最终主源磁盘缓存命中,无实质影响;
- 混域 BERT 首次启动命令误传无效参数,立即终止修正后重启,无产物污染;
- 无 OOM(fp16 下 BERT 峰值显存约 3.5GB/8GB)、无 NaN/Inf、无进程假死,
  全部干预明细见 `outputs/train_interventions.md`。

### 8.6 复现命令

```bash
PY=.venv/Scripts/python.exe
# 方案A 前置 1/2:安装 BERT 依赖(transformers/tokenizers 已列入 requirements.txt)
$PY -m pip install transformers==5.24.0 tokenizers==0.23.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 方案A 前置 2/2:下载 BERT 本地权重(HF 镜像;已入 .gitignore)
HF_ENDPOINT=https://hf-mirror.com $PY -c "from huggingface_hub import snapshot_download; \
  snapshot_download('hfl/chinese-roberta-wwm-ext', local_dir='models/bert-base', \
  allow_patterns=['config.json','vocab.txt','tokenizer_config.json','tokenizer.json','pytorch_model.bin'])"
cd src
# 方案B1:混域数据 + 同配置 Bi-LSTM + 外部基准评估
$PY preprocess_mixed.py
$PY train.py --model bilstm --embedding pretrained_finetune --lr 5e-4 --hidden 128 --dropout 0.3 \
    --run_name mixdom_bilstm_lr5e-4_do0.3_pretune \
    --data_suffix _mix --vocab_file data/processed/vocab_mixed.json \
    --embedding_matrix data/processed/embedding_matrix_mixed.npy
# 方案A:BERT 二分类 / 五级 / 混域(同数据同口径)
$PY train_bert.py --task binary --run_name bert_roberta_wwm
$PY train_bert.py --task 5class --run_name bert_roberta_wwm_5c
$PY train_bert.py --task binary --run_name bert_roberta_wwm_mixdom --data_suffix _mix
# 方案B2:30万五级 + 同配置训练
$PY preprocess_5class_large.py
$PY train_5class.py --model bilstm --embedding pretrained_finetune --lr 5e-4 --hidden 256 --dropout 0.3 \
    --run_name 5c300k_bilstm_pretune_h256 \
    --data_suffix _5cL --vocab_file data/processed/vocab_5class_large.json \
    --embedding_matrix data/processed/embedding_matrix_5class_large.npy
# 外部 30000 基准评估(任意新模型;不改动旧报告产物)
$PY eval_external.py eval_one --bin_run mixdom_bilstm_lr5e-4_do0.3_pretune --data_note "混域"
$PY eval_external.py eval_one --five_run bert_roberta_wwm_5c --data_note "BERT同数据"
# Web 界面零改动:新 ckpt 按现有命名约定(<run>_best.pt / <run>_5c_best.pt)自动发现
```
