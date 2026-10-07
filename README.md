# 自然语言处理 作业一：文本分类

使用不同文本表示方法完成 NYT 新闻三分类（business / politics / sports），
统一在 NYT Test Set 上报告 Accuracy 与 Macro-F1。

## 目录结构

```
HW-1/
├── datasets/                 # 课程提供的数据（只读，不修改）
│   ├── nyt.csv               # NYT 新闻：text + label，11519 条，3 类
│   └── ag.csv                # AG News：仅 text，90000 条（Task 2 训练 Word2Vec 用）
├── common/
│   ├── data.py               # 数据加载 + 固定 80/10/10 划分（落盘缓存）
│   ├── text.py               # nltk 分词 + 磁盘缓存
│   ├── embeddings.py         # GloVe 解析 / Word2Vec 训练 / 文档向量平均
│   └── modeling.py           # Logistic Regression 训练与评价（Task 1/2 共用）
├── scripts/
│   ├── check_env.py          # 依赖自检
│   ├── inspect_data.py       # 数据集结构勘察
│   ├── analyze_lengths.py    # 文档长度分布 / max_len 截断程度分析
│   └── compare_results.py    # 汇总 Task 1/2/3 生成对比表
├── task1_bow.py              # Task 1: Bag of Words + Logistic Regression
├── task2_word2vec.py         # Task 2: GloVe / Word2Vec + Logistic Regression
├── task3_bert.py             # Task 3: BERT-base-uncased 微调
├── splits/                   # 固定划分索引（首次运行自动生成）
├── cache/                    # 分词缓存（首次运行自动生成）
├── embeddings/               # 从 glove.6B.zip 解压的词向量（自动生成，已忽略）
├── results/                  # 指标 json（自动生成）
└── requirements.txt
```

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Task 3 需要 CUDA 版 torch（约 2.5 GB），单独从 PyTorch 官方源安装：

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

自检依赖是否就绪：

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

### 数据准备

课程提供的原始数据未入库，clone 后需要自行放到 `datasets/` 目录下：

```
datasets/
├── nyt.csv    # NYT 新闻，两列 text / label，11519 行，3 类
└── ag.csv     # AG News，一列 text，90000 行（仅 Task 2 训练 Word2Vec 用）
```

放置后用勘察脚本确认读取正常：

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py
```

预期输出：`nyt.csv` 为 `(11519, 2)`、3 类（business 1429 / politics 1451 /
sports 8639）；`ag.csv` 为 `(90000, 1)` 且只有 `text` 列。
若行数或列名不符，说明数据版本不同，需要相应调整 `common/data.py`。

## 运行方式

### Task 1：Bag of Words

```powershell
.\.venv\Scripts\python.exe task1_bow.py
```

首次运行会调用 `nltk.word_tokenize` 对全语料分词（约 1 分钟），结果缓存到
`cache/nyt_word_tokenize_lower_n11519.pkl`，后续实验直接复用（约 1 秒）。
指标同时打印到终端并写入 `results/task1_results.json`。

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | 42 | 数据划分随机种子 |
| `--C` | 1.0 | Logistic Regression 正则强度倒数 |
| `--max-iter` | 1000 | 最大迭代次数 |
| `--limit` | 0（全部） | 只用前 N 条样本，用于冒烟测试 |
| `--no-token-cache` | 关闭 | 忽略缓存，强制重新分词 |

冒烟测试：

```powershell
.\.venv\Scripts\python.exe task1_bow.py --limit 2000
```

缓存与结果文件的命名都带上了样本数，因此冒烟测试**不会**污染正式实验产物：
划分存为 `splits/nyt_split_seed42_n2000.pkl`、结果存为
`results/task1_results_limit2000.json`，与全量实验的 `_n11519` 文件互不影响。

### Task 2：GloVe / Word2Vec

```powershell
.\.venv\Scripts\python.exe task2_word2vec.py
```

首次运行会依次完成：从 `glove.6B.zip` 解压 `glove.6B.100d.txt`（347 MB）并解析成
400000 × 100 的词向量矩阵（缓存为 `.npy`，之后加载约 1 秒）；对 AG News 的 90000 条
文本分词并缓存；分别训练两个 Word2Vec 模型。全流程约 4 分钟。

只跑其中一组：

```powershell
.\.venv\Scripts\python.exe task2_word2vec.py --only glove
```

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--only` | glove ag nyt | 只运行指定的实验 |
| `--nyt-scope` | both | NYT 自训 Word2Vec 的语料范围：`train`（仅训练集，无泄漏）/ `all` / `both` |
| `--dim` | 100 | 词向量维度 |
| `--w2v-window` | 5 | 上下文窗口大小 |
| `--w2v-min-count` | 5 | 最低词频 |
| `--w2v-sg` | 0 | 0 = CBOW，1 = Skip-gram |
| `--w2v-epochs` | 5 | 训练轮数 |
| `--workers` | 1 | 线程数（1 保证结果可复现） |
| `--seed` / `--C` / `--max-iter` / `--limit` | 同 Task 1 | |

### Task 3：BERT 微调

```powershell
.\.venv\Scripts\python.exe task3_bert.py
```

首次运行会从 HF 镜像下载 `bert-base-uncased`（约 440 MB），随后在 GPU 上训练。
本机实测（RTX 4060 Laptop 8 GB，bf16）：3 epochs 共 **98.9 秒**。

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | 3 | 训练轮数（作业规定 3） |
| `--max-len` | 64 | 最大序列长度（作业规定 64） |
| `--batch-size` | 32 | 训练 batch size |
| `--lr` | 2e-5 | 学习率 |
| `--weight-decay` | 0.01 | 权重衰减 |
| `--warmup-ratio` | 0.1 | 线性 warmup 比例 |
| `--model` | google-bert/bert-base-uncased | 预训练模型名 |
| `--limit` | 0（全部） | 只用前 N 条样本，用于冒烟测试 |

冒烟测试（约 10 秒，可确认 GPU 与流程正常）：

```powershell
.\.venv\Scripts\python.exe task3_bert.py --limit 500 --epochs 1
```

## 统一实验约定（Task 1/2/3 共用）
为保证不同表示方法之间可比，以下设定在所有 Task 中保持一致：

- **固定划分**：`common/data.py` 用 `numpy.random.default_rng(42)` 对 11519 条样本
  随机打乱，按 80% / 10% / 10% 切分，索引缓存于 `splits/nyt_split_seed42_n11519.pkl`。
  划分基于**行索引**而非行序，即使 csv 行序变化也完全一致。
  实际规模：train / val / test = **9215 / 1151 / 1153**。
- **分词**：Task 1/2 用 `nltk.word_tokenize`，统一转小写（与 GloVe-6B、Word2Vec 的小写词表对齐），
  **不做**去停用词、词干化或标点过滤。Task 3 的 BERT 使用其自带的 WordPiece 分词器，
  与 nltk 分词无关。
- **词表**：**仅由训练集**构建（避免验证/测试集信息泄漏），按字典序编号使特征下标
  与语料顺序无关；向量化时词表外的词（OOV）直接忽略。词表大小 |V| = 127352。
- **分类器**：`sklearn.linear_model.LogisticRegression`，C=1.0，max_iter=1000。

## Task 1：Bag of Words + Logistic Regression（30 分）

两种词袋表示，均为 `|V|` 维稀疏向量，分类器为 Logistic Regression：

1. **Binary BoW**：`x_i = 1` 若词 i 出现在文档中，否则 0。
2. **Word Frequency**：`x_i` = 词 i 在文档中的出现次数。

### 结果（NYT Test Set，n=1153）

| 表示方法 | Accuracy | Macro-F1 | 训练用时 | 迭代次数 |
| --- | --- | --- | --- | --- |
| Binary BoW | 0.9870 | 0.9717 | 2.45 s | 18 |
| **Word Frequency** | **0.9896** | **0.9751** | 19.72 s | 141 |

验证集（n=1151）结果：

| 表示方法 | Accuracy | Macro-F1 |
| --- | --- | --- |
| Binary BoW | 0.9800 | 0.9579 |
| Word Frequency | 0.9852 | 0.9676 |

### 分类别指标（Test Set）

Binary BoW：

| 类别 | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| business | 0.9769 | 0.9270 | 0.9513 | 137 |
| politics | 0.9494 | 0.9868 | 0.9677 | 152 |
| sports | 0.9954 | 0.9965 | 0.9960 | 864 |

Word Frequency：

| 类别 | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| business | 0.9847 | 0.9416 | 0.9627 | 137 |
| politics | 0.9490 | 0.9803 | 0.9644 | 152 |
| sports | 0.9977 | 0.9988 | 0.9983 | 864 |

### 混淆矩阵（行 = 真实，列 = 预测）

Binary BoW：

| | business | politics | sports |
| --- | --- | --- | --- |
| **business** | 127 | 7 | 3 |
| **politics** | 1 | 150 | 1 |
| **sports** | 2 | 1 | 861 |

Word Frequency：

| | business | politics | sports |
| --- | --- | --- | --- |
| **business** | 129 | 7 | 1 |
| **politics** | 2 | 149 | 1 |
| **sports** | 0 | 1 | 863 |

### 分析

1. **词频略优于二值。** Word Frequency 在测试集上 Accuracy 提升 0.26 个百分点、
   Macro-F1 提升 0.34 个百分点。词频保留了词的强度信息，对判别性关键词（在 sports
   新闻中反复出现的队名、赛事词）有额外区分度，但整体提升有限——说明在这个数据集上
   "词是否出现"已提供了绝大部分信息。代价是训练迭代次数从 18 次增至 141 次
   （特征取值尺度更大、条件数更差），用时约为二值的 8 倍。

2. **Accuracy 明显高于 Macro-F1（约 1.5 个百分点）。** 原因是类别严重不均衡：
   sports 占 75.0%（support 864/1153），且其 F1 高达 0.996；而样本稀少的
   business / politics 只有 0.95 ~ 0.97。Accuracy 被占多数类的 sports 主导，
   Macro-F1 对每个类别等权，因此更能反映模型在少数类上的真实能力。这也是作业要求
   同时报告两个指标的意义所在。

3. **错误集中在 business ↔ politics。** 两种表示的混淆矩阵都显示：15 / 12 个错误中
   有 8 ~ 9 个发生在 business 与 politics 之间（business 被误判为 politics 7 次，
   反向 1 ~ 2 次），而 sports 几乎不会与另两类混淆。business 的召回率最低
   （0.927 / 0.942），politics 的精确率最低（0.949 / 0.949）——这与两类新闻主题天然
   重叠（政治、经济新闻常互相涉及）相符。若要进一步提升，需要能区分主题边界的建模方式
   （如 Task 2 的分布式词向量、Task 3 的上下文预训练模型），而非单纯扩充词表。

## Task 2：Word Embedding / Word2Vec（50 分）

三组实验，均为 100 维词向量；文档向量取文档内所有**有效单词**词向量的平均
（`e(d) = (1/n) Σ e(w_i)`，OOV 跳过）；分类器仍为 Logistic Regression。

Word2Vec 超参数（AG News 与 NYT 两组完全一致）：`vector_size=100, window=5,
min_count=5, sg=0(CBOW), negative=5, epochs=5, workers=1`。

### 结果（NYT Test Set，n=1153）

| 表示方法 | 词表 | token 覆盖率 | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| GloVe-6B-100d (pre-trained) | 400000 | 98.07% | 0.9818 | 0.9585 |
| Word2Vec-100d (AG News) | 25376 | 94.25% | 0.9809 | 0.9553 |
| **Word2Vec-100d (NYT train split)** | 31574 | 97.87% | **0.9853** | **0.9633** |
| Word2Vec-100d (NYT full)〔对照〕 | 35447 | 98.15% | 0.9861 | 0.9676 |

> `NYT full` 的 Word2Vec 在训练时见过测试集文本（无监督、无标签），属于 transductive
> 设置，**不作为主结果**，仅作对照；主实验是 `NYT train split`（只用训练集，无泄漏）。

验证集（n=1151）结果：

| 表示方法 | Accuracy | Macro-F1 |
| --- | --- | --- |
| GloVe-6B-100d (pre-trained) | 0.9748 | 0.9465 |
| Word2Vec-100d (AG News) | 0.9713 | 0.9418 |
| Word2Vec-100d (NYT train split) | 0.9731 | 0.9435 |
| Word2Vec-100d (NYT full) | 0.9791 | 0.9583 |

### 分类别 F1（Test Set）

| 表示方法 | business | politics | sports |
| --- | --- | --- | --- |
| GloVe-6B-100d | 0.9254 | 0.9548 | 0.9954 |
| Word2Vec (AG News) | 0.9219 | 0.9481 | 0.9960 |
| Word2Vec (NYT train split) | 0.9398 | 0.9518 | 0.9983 |
| Word2Vec (NYT full) | 0.9478 | 0.9579 | 0.9971 |

### 分析

1. **词向量平均整体不如词袋。** 四组词向量实验的 Macro-F1 落在 0.955 ~ 0.968，
   而 Task 1 的 Word Frequency 为 **0.9751**、Binary BoW 为 0.9717。原因有两层：
   (a) **平均会稀释判别性强的词**——NYT 文档平均 731 个 token，一篇 sports 新闻里
   几十个队名、赛事词的信息，被大量中性词平均掉；(b) **词表被 min_count=5 截断**——
   词袋保留了 127352 个词（含大量低频但高判别性的词），而 Word2Vec 词表只有 2.5 万 ~ 3.5 万。
   这也说明：在训练样本充足的主题分类任务上，稀疏的高维词袋仍是很难超越的强基线。

2. **同域训练优于跨域。** NYT 自训（0.9633）> 预训练 GloVe（0.9585）> AG News 自训
   （0.9553）。NYT 语料只有 8.4M token，远少于 GloVe 的 6B token，但同域语料学到的
   词向量更贴合新闻主题词，因此反超预训练词向量。

3. **AG News 模型与 GloVe 几乎打平，但覆盖率最低。** AG News 词表仅 25376 词，
   对 NYT 的 token 覆盖率 94.25%，明显低于 GloVe 的 98.07%——这是跨域的直接体现
   （AG News 是短标题式新闻，NYT 是长篇正文，词汇更丰富）。覆盖率低 3.8 个百分点，
   Accuracy 却只差 0.09 个百分点：被丢掉的 OOV 多是低频词，对文档平均向量的影响本就很小。

4. **business 依旧是最难的类别。** 四组词向量的 business F1 在 0.92 ~ 0.95，
   均低于词袋的 0.9627，与 Task 1 的结论一致：business 与 politics 主题重叠，
   而词向量平均这种"一袋词求均值"的做法无法刻画主题边界。

## Task 3：Pre-trained Neural Models（20 分）

微调 `google-bert/bert-base-uncased`（109.5M 参数）做三分类。
按作业规定：**max sequence length = 64**、**3 epochs**。

超参数（作业未规定，采用 BERT 微调的常规设置并在下方说明）：
AdamW，lr=2e-5，weight_decay=0.01，linear warmup 10%（86/864 步），
batch size 32，bf16 混合精度，seed=42。

### 训练过程（各轮验证集）

| Epoch | val Loss | val Accuracy | val Macro-F1 |
| --- | --- | --- | --- |
| 1 | 0.0864 | 0.9670 | 0.9362 |
| 2 | 0.1090 | 0.9705 | 0.9426 |
| 3 | 0.0878 | 0.9757 | 0.9529 |

三轮都在稳定提升，没有明显过拟合（验证 loss 在第 2 轮略升后回落）。

### 结果（NYT Test Set，n=1153）

| 指标 | 数值 |
| --- | --- |
| Accuracy | 0.9792 |
| Macro-F1 | 0.9585 |
| 训练用时 | 98.9 s（RTX 4060 Laptop，bf16） |

分类别指标：

| 类别 | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| business | 0.9286 | 0.9489 | 0.9386 | 137 |
| politics | 0.9299 | 0.9605 | 0.9450 | 152 |
| sports | 0.9965 | 0.9873 | 0.9919 | 864 |

混淆矩阵（行 = 真实，列 = 预测）：

| | business | politics | sports |
| --- | --- | --- | --- |
| **business** | 130 | 5 | 2 |
| **politics** | 5 | 146 | 1 |
| **sports** | 5 | 6 | 853 |

### 关键发现：max_len=64 让 BERT 只看到约 10% 的正文

用真实 WordPiece 分词器精确测量全语料后：

| 指标 | 数值 |
| --- | --- |
| WordPiece 长度 中位数 / 均值 / P95 / 最大 | 873 / 835 / 1329 / 6775 |
| 能完整放进 64 个 token 的文档 | **5 / 11519（0.04%）** |
| 平均能看到的正文比例 | **10.6%** |
| 中位文档能看到的比例 | 7.3% |

作业规定的 max_len=64 意味着每篇新闻只保留开头约 49 个词。BERT 依然拿到 0.9585 的
Macro-F1，说明 NYT 沿用路透社体例、**导语高度浓缩主题**，开头几十个词已含大部分判别信息；
但同时也解释了它为什么没能超过词袋——其余 90% 的正文被直接丢弃了。

## 三组方法横向对比

| 表示方法 | Accuracy | Macro-F1 | business F1 | politics F1 | sports F1 |
| --- | --- | --- | --- | --- | --- |
| **Word Frequency（词频词袋）** | **0.9896** | **0.9751** | 0.9627 | 0.9644 | 0.9983 |
| Binary BoW（二值词袋） | 0.9870 | 0.9717 | 0.9513 | 0.9677 | 0.9960 |
| Word2Vec-100d（NYT 全文，对照） | 0.9861 | 0.9676 | 0.9478 | 0.9579 | 0.9971 |
| Word2Vec-100d（NYT 训练集） | 0.9853 | 0.9633 | 0.9398 | 0.9518 | 0.9983 |
| GloVe-6B-100d（预训练） | 0.9818 | 0.9585 | 0.9254 | 0.9548 | 0.9954 |
| BERT-base-uncased（max_len=64） | 0.9792 | 0.9585 | 0.9386 | 0.9450 | 0.9919 |
| Word2Vec-100d（AG News） | 0.9809 | 0.9553 | 0.9219 | 0.9481 | 0.9960 |

（由 `python scripts/compare_results.py` 自动生成，见 `results/comparison.md`）

### 结论

1. **在这个任务上，简单词袋反而最强。** Word Frequency 以 0.9751 的 Macro-F1 居首，
   超过 109.5M 参数的 BERT（0.9585）1.7 个百分点。这并不反常，原因有三：
   词袋用了**全文**而 BERT 只看到 10.6%；词袋的 127352 维稀疏特征保留了
   **低频高判别性词**，而词嵌入被 min_count=5 截断、平均操作又进一步稀释；
   训练集有 9215 条标注样本，对这个三分类任务而言已足够支撑高维线性模型。

2. **表示能力的排序呈现清晰的梯度：词袋 > 同域词嵌入 > 跨域词嵌入。**
   词袋（0.9717~0.9751）> NYT 自训 Word2Vec（0.9633）> 预训练 GloVe（0.9585）
   > AG News 自训 Word2Vec（0.9553）。词嵌入之间的差异主要来自**领域匹配度**而非
   语料规模——GloVe 用了 60 亿 token，仍不敌只有 840 万 token 的 NYT 自训向量。

3. **所有方法的瓶颈完全一致：business 与 politics 的混淆。**
   三种表示的混淆矩阵都显示错误高度集中在这两类之间，而 sports 的 F1 一律在
   0.99 以上。这与两类新闻主题天然重叠（政治、经济新闻互相涉及）相符。
   business 的 F1 从词袋的 0.9627 一路降到 GloVe 的 0.9254 与 AG News 的 0.9219，
   说明**越是丢失细节的表示，越难区分这两个近邻类别**。

4. **BERT 的 sports 召回率反而最低（0.9873）。** 其他方法都在 0.995 以上。
   结合截断现象可以推断：只读开头 64 个 token 时，部分体育新闻的导语
   （涉及球队交易、合同、联盟政策）看起来更像商业或政治新闻。这从反方向印证了
   截断确实造成了信息损失。

5. **若想真正发挥 BERT 的优势，需要放宽 max_len。** 本次按作业要求固定在 64。
   从截断数据可以预期：把 max_len 提到 256/512（覆盖 30%~60% 正文）应能显著缩小
   与词袋的差距，甚至反超——这需要额外的对照实验来验证。

## 备注

- `datasets/` 下文件为课程提供，脚本只读不写；该目录已在 `.gitignore` 中排除，
  不入库（见「数据准备」）。
- `glove.6B.zip`（822 MB）超过 GitHub 单文件 100 MB 限制，已在 `.gitignore` 中排除。
- `embeddings/`（解压出的 GloVe 文本 347 MB + 解析缓存 153 MB）同样已忽略，
  运行 `task2_word2vec.py` 会自动从 zip 重建，无需手动准备。
- BERT 权重缓存在 `~/.cache/huggingface/`（约 440 MB），不在仓库内。
- `splits/`、`cache/`、`results/` 为运行产物；`splits/` 与 `results/` 建议提交，
  以保证划分与结果可追溯；`cache/` 体积较大已忽略。
- 复现说明：所有随机种子固定为 42（`--seed`）。Task 1/2 完全确定性；
  Task 3 走 GPU + cuDNN，个别算子存在非确定性，重复运行指标可能有
  小数点后第 3~4 位的波动，属正常现象。
