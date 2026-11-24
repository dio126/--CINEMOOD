# 训练过程干预记录(方案A/B 改进实验)

> 本文件如实记录方案A(新增 BERT)与方案B(数据侧增强)全部训练的监控与人工/自动干预:
> 每条记录含 时间 / 现象 / 处置 / 结果。失败与降级同样入册,不做选择性报告。

| # | 时间 | 实验 | 现象 | 处置 | 结果 |
|---|------|------|------|------|------|
| 1 | 11-24 11:05 | (环境)transformers 安装 | 未安装 | pip 清华镜像后台安装(与 B1 开发并行) | 成功,transformers 5.24.0 + tokenizers 0.23.3(Python 3.13 兼容) |
| 2 | 11-24 11:10 | (环境)BERT 权重下载 | 首次 snapshot_download 连接超时(WinError 10060,hf-mirror 暂不可达);后台管道 `| tail` 吞掉退出码,显示 exit 0 具有误导性 | curl 探活后原命令重试 | 重试成功:models/bert-base/ 完整落盘(pytorch_model.bin 401MB + vocab.txt + tokenizer,共 5 文件),tokenizer 自检通过;后续后台任务一律落盘 EXIT= 真实退出码 |
| 3 | 11-24 11:15 | B1 混域 Bi-LSTM 训练 | 正常:epoch1 val_acc 0.8465 → epoch2 峰值 0.8561 → epoch3 起过拟合(val_loss 0.34→0.50),第 5 轮早停触发;无 NaN/Inf | 无需干预(早停机制自愈) | 最佳 epoch2,test acc 0.8388 / F1 0.8415;外部 30000 基准 Acc 0.8440(旧模型 0.6780,+16.6pp) |
| 4 | 11-24 11:25 | BERT 二分类训练 | 正常:epoch1 val_acc 0.9161 → epoch2 0.9084(过拟合迹象),早停触发;无 NaN/Inf,GPU 97%/3.3GB(fp16 生效无 OOM) | 无需干预(早停自愈) | 最佳 epoch1,test acc 0.9382 / F1 0.9450;外部 30000 基准 Acc 0.7779 |
| 5 | 11-24 11:33 | B2 30万数据构建 | 词向量步骤崩溃:download_wordvec 组合式初始化 `vocab[neg]` 未判存在,新词表(5cL)无单字"别"→KeyError;后续两个在线降级源一失败一超时(预期内,主源已命中缓存) | 修复 bug(加 `vocab.get(neg)` 守卫;对含否定词的旧词表计算路径完全不变,已生成的旧矩阵文件不受影响),单独重跑 build_embedding_matrix | 数据 pkl 已建好不受影响(300000 条,五重排除自检通过);修复后重建矩阵成功(见 retry 日志) |
| 6 | 11-24 11:36 | BERT 五级训练 | 正常:3 epoch 全跑完(未触发早停),val macro-F1 0.5340→0.5471 持续提升,无 NaN/Inf,GPU 100%/3.5GB | 无需干预 | 最佳 epoch3,test exact 0.5464 / macro-F1 0.5452;外部 30000 exact 0.5296 / ±1 0.8963 / macro-F1 0.5318 |
| 7 | 11-24 11:56 | B2 30万 Bi-LSTM 训练 | 正常:epoch2 峰值 val_mf1 0.5233 → epoch3 起过拟合,第 5 轮早停;无 NaN/Inf;全程 7.3 分钟(远低于 90 分钟降级线,未降规模) | 无需干预 | test acc 0.5131 / macro-F1 0.5145;外部 exact 0.5116 / ±1 0.8816 / macro-F1 0.5154 |
| 8 | 11-24 12:05 | 混域 BERT(可选加练) | 首次启动误传无效参数 --data_note(argparse 直接报错退出,未产生任何产物);另发现 train_bert.py 不支持 _mix 数据后缀 | 立即终止、补 --data_suffix 参数与 ckpt config 记录、顺手修复 evaluate.py run_evaluate 对 BERT 需按 tokenizer 重编码的分支,重启训练 | 修正后正常:val acc 峰值 0.9022(epoch2),test acc 0.8962 / F1 0.8976;外部 30000 Acc 0.8928 / F1 0.8927(+21.5pp,全场最佳) |
| 9 | 11-24 12:40 | 文档写作自查 | 发现曾把混域 BERT 的预估数字误写入 README 对比表(训练尚未结束) | 立即改为显式占位符,训练完成后以 external_benchmark_summary.csv 真实数字回填 | 已用真实值(0.8962/0.8976;0.8928/0.8927)回填,文档中无任何非真实产出数字 |

补充记录(数据侧,非训练但影响结果可信度):
- B1 构建时"排除 30000 基准"步骤只移除 29999 条(差 1 条)。排查:pandas 读回 CSV 的 dtype 推断会使个别纯数字/布尔型文本变形(CSV 往返失配),该变形文本属于排除集合匹配失败侧。已对**最终 pkl** 用 dtype=str 重新做直接交集复核:train/val/test_mix 与 30000 基准、隐晦 300 交集均为 0,无泄漏;该未匹配文本亦未被抽入训练集。
- B1 训练集含 ChnSentiCorp 全量 9060 条(设计内),与旧二分类 pkl 存在文本重叠属预期(新模型新实验,不构成对旧模型测试集的污染);与五级训练数据、外部基准交集为 0。

