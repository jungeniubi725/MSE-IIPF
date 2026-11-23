# 发布验证说明

本次整理于 **2026-11-23** 验证了源码的加载、训练、独立测试、预测导出和绘图流程。验证使用小规模人工生成的测试输入，其指标不属于论文实验结果。

## 已完成的检查

| 检查 | 实际范围与结果 |
| --- | --- |
| 自动化测试 | `pytest` 完整测试集：**28 项通过**，包括数值性质、数据划分、预测导出和两条 CMU 流程 |
| RAVDESS 模型流程 | CPU 上使用 12 个合成 sequence、每个 2 个 unit，完成训练、验证集选模、重新加载 checkpoint、独立测试和绘图 |
| CMU-MOSI / CMU-MOSEI | 分别使用临时 CSV 和明确的训练/验证/测试名单，完成数据准备、ALBEF 对齐训练、IIPF 分类训练和独立测试；训练阶段未生成测试集预测 |
| 预测与特征导出 | 检查样本 ID、概率形状及归一化；检查 TMA/IIPF 特征导出和证据聚合；保留不足一个 batch 的最后一批样本 |
| 绘图 | 从保存的预测生成混淆矩阵、校准曲线、风险–覆盖率曲线、不确定性分布和训练曲线，导出 PNG/SVG/PDF 与 CSV |
| 下载入口 | 检查数据和模型下载计划；测试官方划分列表的解析、CSD 转换及 RAVDESS 归档提取规则 |
| 命令行入口 | 25 个脚本的 `--help` 均成功返回 |
| 静态检查 | 79 个 Python 文件通过语法解析，项目配置通过 TOML 解析，Markdown 本地链接检查通过 |
| 来源核对 | 64 个继承源码文件的原始副本校验值保持不变；发布副本与原始副本的哈希记录见 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) |
| 发布包规则 | 验证权重、数据、历史结果和缓存不会进入 ZIP，同时保留 `src/data/` 中的数据加载源码 |

完整测试使用 Python 3.12.12、PyTorch 2.8.0、NumPy 2.4.1、pandas 2.3.3、SciPy 1.17.0、scikit-learn 1.8.0、Matplotlib 3.10.8、h5py 3.14.0 和 pytest 8.3.4。训练与流程检查均使用 CPU；环境中可用的 CUDA 不代表完整 GPU 实验已执行。

## 如何重新检查

按照 [README](../README.md#installation) 安装核心依赖，然后在项目根目录执行：

```bash
python -m pytest -q
python scripts/smoke_test.py --out-dir runs/software-check --device cpu
python scripts/package_release.py --output dist/MERC-GitHub-source.zip
```

测试产生的临时数据、checkpoint、指标和图形不会随源码发布。`smoke_test.py` 的通过记录标明 `research_results: false`。重复运行时请为已有实验或压缩包选择新的输出名称。

GitHub Actions 配置位于 [tests.yml](../.github/workflows/tests.yml)，将运行 CPU 测试、流程检查与打包。远程运行状态以仓库 Actions 页面的实际记录为准，本地验证记录不替代远程 CI 状态。

## 本次没有验证的内容

- 没有重新下载完整数据集及全部大模型权重，也没有逐个验证上游文件服务器当前可用性。
- 没有在全新环境中重新安装完整 LLaVA/FunASR 工具链，或实际加载全部预训练编码器；这部分依赖及已知环境版本见 [DATA_AND_MODELS.md](DATA_AND_MODELS.md)。
- 没有重新运行完整数据上的长时间 GPU 训练、多种子/五折实验和全部消融，也没有核实论文表格中的每个数值。
- 没有将历史结果、模型权重或训练数据加入发布包。下载与训练后才会在本地产生这些文件。

论文描述和原实验实现之间的已知差异见 [METHOD_CODE_MAP.md](METHOD_CODE_MAP.md)。软件测试通过说明上述流程可执行，不能替代正式的实验复现。
