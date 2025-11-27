# 硬件木马触发寄存器检测 (Hardware Trojan Trigger Register Detection)

## 目标
构建寄存器级木马触发检测模型，并提供推理接口。
任务目标：给定一个 design 的扫描链数据 seqs（T×N），输出长度 N 的寄存器概率向量 p[i]。

评估指标（要求 > 90%）：
*   Precision, Recall, F1
*   Precision@k, Recall@k
*   AUROC
*   IoU (Jaccard Index)

## 目录结构
```
ht_detector/
│
├── data/
│   ├── seqs_data/              # 存放 .npz 数据
│   ├── labels.json             # 标签文件
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── feature_extractor.py
│   ├── dataset_builder.py
│   ├── train_lgbm.py
│   ├── evaluate.py
│   ├── infer.py
│
├── models/
│   ├── lgbm_model.txt          # 训练好的模型
│   ├── scaler.pkl              # 归一化参数
│
├── requirements.txt
└── README.md
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 构建训练集
```bash
python src/dataset_builder.py
```
该步骤会读取 `data/seqs_data` 下的 npz 文件，抽取特征，并生成训练数据。

### 2. 训练模型
```bash
python src/train_lgbm.py
```
训练 LightGBM 模型，并保存到 `models/lgbm_model.txt`。

### 3. 评估模型 (LODO)
```bash
python src/evaluate.py
```
使用 Leave-One-Design-Out 方式验证模型泛化性能，输出 Recall, F1 等指标。

### 4. 推理
```bash
python src/infer.py --file data/seqs_data/aes-t1200-aes_trojan_comb.npz --top_k 20
```
对指定设计文件进行推理，输出最可疑的前 20 个寄存器。
