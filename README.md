# 硬件木马触发寄存器检测 (Hardware Trojan Trigger Register Detection)

## 项目简介
本项目旨在构建基于 **Transformer** 和 **混合专家模型 (MoE)** 架构的深度学习模型，用于检测芯片设计中的硬件木马触发寄存器。
任务目标是给定一个 design 的扫描链数据序列，识别出其中极少量的木马触发寄存器。

## 核心特性
*   **Transformer 架构**: 利用自注意力机制捕捉寄存器序列的长程依赖关系。
*   **MoE (Mixture of Experts)**: 引入稀疏混合专家模型，通过 Top-K 门控路由机制增强模型容量，同时保持计算效率。
*   **数据增强**: 针对正负样本极度不平衡的问题（木马寄存器极少），采用了多种数据增强策略（高斯噪声、特征缩放、特征掩码）和加权采样。
*   **实时监控**: 训练过程中实时监控 F1-score、Precision、Recall 和 Loss 变化。
*   **动态阈值**: 推理和验证阶段采用动态阈值搜索，以最大化 F1 分数。

## 目录结构
```
ht_detector/
│
├── data/
│   ├── seqs_data/              # 存放 .npz 数据
│   ├── labels.json             # 标签文件
│
├── src/
│   ├── transformer/            # Transformer & MoE 核心代码
│   │   ├── train.py            # 训练脚本 (包含训练循环、验证、日志)
│   │   ├── model.py            # 模型定义 (Transformer, MoE, Gate, Block)
│   │   ├── augmentation.py     # 数据增强实现
│   │
│   ├── config.py               # 全局配置 (路径、超参数)
│   ├── utils.py                # 工具函数
│   ├── dataset_builder.py      # 数据集构建
│   ├── infer.py                # 推理脚本
│   ├── evaluate.py             # 评估脚本
│
├── models/
│   ├── transformer/            # 训练好的 Transformer 模型及日志
│   │   ├── best_model.pth      # 最佳模型权重
│   │   ├── train.log           # 训练日志
│
├── requirements.txt
└── README.md
```

## 环境依赖
请确保安装以下依赖：
```bash
pip install -r requirements.txt
```
主要依赖库包括：`torch`, `numpy`, `scikit-learn`, `tqdm` 等。

## 使用方法

### 1. 数据准备
确保数据位于 `data/seqs_data` 目录下。如果需要重新构建数据集索引：
```bash
python src/dataset_builder.py
```

### 2. 训练模型
运行 Transformer 模型的训练脚本。该脚本会自动进行 K-Fold 交叉验证，并应用数据增强和 MoE 架构。
```bash
python src/transformer/train.py
```
训练过程中，终端会实时显示 Loss、Learning Rate 以及验证集的 Precision, Recall, F1 Score。

### 3. 模型推理
使用训练好的模型对指定数据文件进行推理：
```bash
python src/infer.py --input data/seqs_data/aes-t1200-aes_trojan_comb.npz --model transformer --output results.npy
```
*   `--input`: 输入的 .npz 数据文件路径。
*   `--model`: 指定模型类型，这里使用 `transformer`。
*   `--output`: 输出预测结果的文件路径 (.npy 格式，包含所有寄存器的概率值)。

### 4. 模型评估
评估模型在整个数据集上的性能：
```bash
python src/evaluate.py
```

## 评估指标
由于木马样本极其稀缺，我们主要关注以下指标：
*   **F1 Score**: 精确率和召回率的调和平均。
*   **Precision (查准率)**: 预测为木马的样本中实际为木马的比例。
*   **Recall (查全率)**: 实际为木马的样本中被正确预测的比例。
*   **Average Precision (AP)**: PR 曲线下的面积，综合反映模型性能。

## 注意事项
*   `src/lgbm/` 目录下的 LightGBM 相关代码已不再作为本项目的主要维护方向，请优先使用 Transformer 模型。
*   训练脚本会自动处理 `GradScaler` 的兼容性问题，支持不同版本的 PyTorch。
