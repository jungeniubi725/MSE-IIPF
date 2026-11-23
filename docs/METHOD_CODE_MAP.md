# 论文方法与源码对应

本文档以稿件中的模块和数学表达式定位代码。原源码注释中的 `E1`–`E52`、`L31` 等是早期稿件的定位编号，各模块的编号可能重复，不能直接视为当前稿件的连续公式编号。

## 主流程与数学对象

| 论文对象 / 运算 | 实际入口 | 对应配置 |
| --- | --- | --- |
| `(V,A) → SG-DSFE → TMA → IIPF → (p,y,u,r)` | [`MSEIIPF.forward`](../src/models/mse_iipf.py) | `configs/base.yaml`、`configs/ravdess.yaml` |
| `H_v = E_v(V)` | [`VideoMAEExtractor`](../src/extractors/visual_videomae.py) | `--visual-model`、`--visual-pool`、`--num-frames` |
| `H_a = E_a(A)` | [`Emotion2VecExtractor`](../src/extractors/audio_emotion2vec.py) | `--audio-model`、`--max-frames` |
| `D_v = G_v(V)`、`D_a = G_a(A)` | [`LlavaCaptioner`](../src/extractors/caption_llava.py)、[`WhisperTranscriber`](../src/extractors/asr_whisper.py) | `--llava-dir`、`--whisper-dir` |
| `H_sv = E_t(D_v)`、`H_sa = E_t(D_a)` | [`RobertaEncoder`](../src/extractors/text_roberta.py) | `--text-model` |
| 双向 InfoNCE，`L_DSA = L_v,sv + L_a,sa` | [`SGDSFE.forward`](../src/models/sg_dsfe.py)、[`bidirectional_infonce`](../src/losses/contrastive.py) | `tau_dsa`、`w_dsa` |
| 跨注意力语义注入、一致性门控、文本注意力池化 | [`SGDSFE`](../src/models/sg_dsfe.py)、[`ConsistencyGate` 等基础层](../src/models/layers.py) | `use_semantic_stream`、`use_gate` |
| `r = sigmoid(Wz+b)`、`M = 1[r>gamma]` | [`ModalityAwareMasking.forward`](../src/models/tma.py) | `gamma`、`use_mam` |
| 三对模态的加权单向对比损失 `L_ITAC` | [`TriModalAlignment.forward`](../src/models/tma.py)、[`single_infonce`](../src/losses/contrastive.py) | `tau_itac`、`itac_weights` |
| 联合三模态正负匹配 `L_ITAM` | [`TriModalAlignment._itam`](../src/models/tma.py) | `use_itam`、`w_tma` |
| 模态可靠性 `omega`、共享证据中心 `c`、初始代理 `z0` | [`ProxyInit.forward`](../src/models/iipf.py) | `d_p`、`tau_r` |
| `L_CMC = L_rep + lambda_d L_dist` | [`ProxyInit`](../src/models/iipf.py)、[`dist_consistency_loss`](../src/losses/divergence.py) | `lambda_d`、`use_cmc` |
| GRU 时序代理、证据变化门控、对比预测 `L_TPC` | [`TemporalEvolution.forward`](../src/models/iipf.py) | `tau_z`、`use_tpc` |
| `L_PRR = L_rec + lambda_cyc L_cycle` | [`ProxyReconstruction.forward`](../src/models/iipf.py) | `mu`、`tau_q`、`lambda_cyc`、`use_prr` |
| 稳定先验、模态加权、`f = LN(e_ext + g * e_z)` | [`ProxyGuidedFusion`](../src/models/iipf.py) | `agg`、`use_proxy_weight`、`d_f` |
| `e = softplus(MLP(f))`、`alpha = e+1`、`p = alpha/S`、`u = K/S` | [`EvidentialHead._forward_evidential`](../src/models/euc_head.py) | `num_classes`、`use_evidential` |
| `L_EDL = mean(sum_k y_k [digamma(S)-digamma(alpha_k)]))` | [`EvidentialHead._edl_loss`](../src/models/euc_head.py) | — |
| 修正参数 `alpha_tilde` 与 Dirichlet KL 正则 | [`EvidentialHead._kl_loss`](../src/models/euc_head.py)、[`kl_dirichlet_uniform`](../src/losses/divergence.py) | `anneal_epochs` |
| 模态冲突 `kappa = mean(sqrt(JS_2(p_m,p_n)))` | [`conflict_kappa`](../src/losses/divergence.py) | 推理时 `use_sqrt=True` |
| 总损失及辅助项 warm-up | [`MSEIIPF.forward`](../src/models/mse_iipf.py)、[`Trainer.train_one_epoch`](../src/engine/trainer.py) | `w_dsa`、`w_tma`、`w_iipf`、`aux_warmup_epochs` |

## 必须按实际代码解释的细节

### 校准目标与辅助监督

现有实现的校准目标为：

$$L_{cal}=\frac{1}{N}\sum_t\left(u_t-\mathbf{1}[\hat y_t\ne y_t]\right)^2.$$

稿件描述的是回归至 stop-gradient 模态冲突的目标。源码计算 `kappa` 用于诊断统计，但默认 `L_cal` 不使用它。发布时保留这一现状，没有在包装过程中替换训练目标。对应函数是 `EvidentialHead._calibration_loss`。

实际头部目标还包含辅助分类、三模态有监督头和可选序列监督：

$$L_{head}=L_{EDL}+\lambda_{KL}L_{KL}+\lambda_{cal}L_{cal}+\lambda_{CE}L_{CE}+\lambda_{uni}L_{uni}+\lambda_{seq}L_{seq}.$$

默认 `lambda_ce=4.0`、`lambda_uni=0.5`、`lambda_seq=0.0`，具体值以保存的 `config.json` 为准。它们不能在公式说明中被省略为“完全等同于只有 EDL、KL 和冲突校准的目标”。

### MAM 与时间门控

MAM 使用 straight-through 梯度估计处理硬阈值，并在整行被遮蔽时保留一个 token；`L_MAM` 由保留率正则和遮蔽前后分布一致性构成。这些是原实验代码对算法的具体实现选择。

`TemporalEvolution` 的门控输入为 `[delta_c; ||delta_c||_2]`，门控层维度为 `d_p+1`；当前稿件部分文字使用 `[delta_c; |delta_c|]`。两者输入维度不同，不能认为是同一逐元素表达式。

### 时间单位、序列预测和填充

RAVDESS 默认将每段视频分成 4 个时间单元；视觉与音频按片段提特征，文本按整个 sequence 生成并广播。证据聚合使用一次先验：

$$\alpha_{seq}=1+\sum_{t\in valid}(\alpha_t-1),\quad p_{seq}=\alpha_{seq}/\sum_k\alpha_{seq,k}.$$

对应 [`aggregate_dirichlet_evidence`](../src/utils/evidential.py)。不能直接累加 `alpha_t`，否则重复计算先验。Softmax 消融则平均有效片段的概率。

原 IIPF 的部分对比项在 batch 内使用全部样本作负样本，未对每一个对比项完整排除变长序列的 padding；本包保留其研究实现。RAVDESS 默认等长 4 片段，CMU 为变长视频内 utterance。若修改这一处理，应作为新的方法版本记录并重新训练，不能用新实现的结果冒充历史实验。

### CMU 的实现边界

[`experiments/cmu/`](../experiments/cmu/) 使用 [`tri_albef_align.py`](../src/models/tri_albef_align.py) 的动量编码器、队列及成对 ITC/ITM，然后训练共享 IIPF 和证据头。它没有从原始音视频执行 SG-DSFE，也不是 RAVDESS 联合三元组 ITAM 的同名替代。默认三分类阈值为 ±0.5。

### LLaVA 权重来源

下载脚本提供上游 `lmms-lab/LLaVA-Video-7B-Qwen2`。检查到的本地下载元数据指向该模型配置的提交 `013210b3aff822f1558b166d39c1046dd109520f`。这证明有上游下载记录，不证明进行了本论文专用微调，也不等同于逐分片权重校验。

本次未找到本论文专用 LLaVA 微调的数据配方、训练入口和可核实的独立发布地址，因而不编造这些文件。若另有作者微调模型，可通过 `download_models.py --llava-repo` 或 `s1_gen_text.py --llava-dir` 使用，并记录其版本。

## 结果边界

默认配置来自实际实验源码，未依据论文中最高分数字反向设置超参数或修改预测。论文后期表格中出现的其他切片数、代理维度等设置，需要对应配置、特征重新提取和正式运行记录才能验证。项目测试通过只说明代码流程通过测试，不构成这些论文数字的复现证据。

