"""Task 1: Bag of Words + Logistic Regression（30 分）

实现两种词袋表示，分类器统一使用 Logistic Regression，
并在 NYT Test Set 上报告 Accuracy 与 Macro-F1：

1. **Binary Bag of Words**：只记录词是否在文档中出现（0/1）。
2. **Word Frequency**：记录每个词在文档中的出现次数。

两种表示共用同一套流程：固定划分 → nltk 分词 → 用**训练集**构建词表 →
向量化 → 训练 Logistic Regression → 在验证集/测试集上评价。
词表只由训练集构建，以避免验证集/测试集的信息泄漏。

用法::

    python task1_bow.py                     # 完整实验
    python task1_bow.py --limit 2000        # 冒烟测试（只用前 2000 条）
    python task1_bow.py --C 0.5 --max-iter 2000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from common.data import (
    DATASETS_DIR,
    LABEL_COL,
    RESULTS_DIR,
    TEXT_COL,
    load_nyt,
    load_split,
)
from common.modeling import describe_result, print_summary_table, run_experiment
from common.text import tokenize_corpus

TOKEN_CACHE_NAME = "nyt_word_tokenize_lower"
LOWERCASE = True  # GloVe(6B) 与 Word2Vec 词表均为小写，统一小写以便后续 Task 复用分词结果


def build_vocab(tokenized_train: Sequence[Sequence[str]]) -> dict[str, int]:
    """从训练语料构建词表。

    词表按字典序排序后编号，使特征下标与语料中词的出现顺序无关，
    从而保证结果可复现、便于不同表示方法之间对齐。
    """
    vocabulary: set[str] = set()
    for tokens in tokenized_train:
        vocabulary.update(tokens)
    return {word: idx for idx, word in enumerate(sorted(vocabulary))}


def vectorize(
    tokenized_docs: Sequence[Sequence[str]],
    vocab: dict[str, int],
    binary: bool,
) -> sparse.csr_matrix:
    """把分词后的文档转成稀疏词袋矩阵。

    Args:
        tokenized_docs: 每篇文档的词序列。
        vocab: 训练集词表，词表外的词（OOV）直接忽略。
        binary: True → 只记 0/1；False → 记词频。

    Returns:
        形状为 ``(n_docs, len(vocab))`` 的 CSR 稀疏矩阵。
    """
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    for tokens in tokenized_docs:
        for word, count in Counter(tokens).items():
            col = vocab.get(word)
            if col is None:  # 未登录词不计入
                continue
            indices.append(col)
            data.append(1.0 if binary else float(count))
        indptr.append(len(indices))

    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), indices, indptr),
        shape=(len(tokenized_docs), len(vocab)),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Bag of Words + Logistic Regression")
    parser.add_argument("--seed", type=int, default=42, help="数据划分随机种子")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic Regression 正则强度倒数")
    parser.add_argument("--max-iter", type=int, default=1000, help="最大迭代次数")
    parser.add_argument("--limit", type=int, default=0, help="仅使用前 N 条样本（冒烟测试用）")
    parser.add_argument("--no-token-cache", action="store_true", help="忽略分词缓存，重新分词")
    args = parser.parse_args()

    # ---------- 1. 数据与固定划分 ----------
    df = load_nyt()
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    n_samples = len(df)
    split = load_split(seed=args.seed, n_samples=n_samples)

    labels = df[LABEL_COL].to_numpy()
    class_names = sorted(set(labels.tolist()))

    print("=" * 78)
    print("Task 1: Bag-of-Words + Logistic Regression")
    print("=" * 78)
    print(f"数据集     : {DATASETS_DIR / 'nyt.csv'}  (n={n_samples})")
    print(
        f"固定划分   : seed={args.seed}  "
        f"train/val/test = {split.sizes['train']}/{split.sizes['val']}/{split.sizes['test']}"
    )
    print(f"类别       : {len(class_names)} 类 -> {class_names}")
    print(f"超参数     : C={args.C}, max_iter={args.max_iter}")

    # ---------- 2. 分词（全语料一次，带缓存） ----------
    started = time.perf_counter()
    tokenized = tokenize_corpus(
        df[TEXT_COL].tolist(),
        cache_name=TOKEN_CACHE_NAME,
        lowercase=LOWERCASE,
        use_cache=not args.no_token_cache,
    )
    print(f"分词耗时   : {time.perf_counter() - started:.1f}s（含缓存读写）")

    train_tokens = [tokenized[i] for i in split.train]
    val_tokens = [tokenized[i] for i in split.val]
    test_tokens = [tokenized[i] for i in split.test]
    y_train = labels[split.train]
    y_val = labels[split.val]
    y_test = labels[split.test]

    # ---------- 3. 仅用训练集构建词表 ----------
    vocab = build_vocab(train_tokens)
    print(f"词表 |V|   : {len(vocab)}（仅由训练集构建）")
    avg_len = sum(len(t) for t in train_tokens) / max(1, len(train_tokens))
    print(f"训练集平均文档长度: {avg_len:.1f} tokens")

    # ---------- 4. 两种表示 + 训练 + 评价 ----------
    experiments = [("Binary BoW", True), ("Word Frequency", False)]
    results: list[dict[str, object]] = []

    for name, binary in experiments:
        print("-" * 78)
        print(f"[{name}] 向量化 ...")
        X_train = vectorize(train_tokens, vocab, binary=binary)
        X_val = vectorize(val_tokens, vocab, binary=binary)
        X_test = vectorize(test_tokens, vocab, binary=binary)
        density = X_train.nnz / (X_train.shape[0] * X_train.shape[1])
        print(f"[{name}] 矩阵 {X_train.shape}, 非零占比 {density:.2%}, 训练中 ...")

        result = run_experiment(
            representation=name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            C=args.C,
            max_iter=args.max_iter,
        )
        results.append(result)
        print(describe_result(result))

    # ---------- 5. 汇总 ----------
    print_summary_table(results)

    payload = {
        "task": "task1_bow",
        "dataset": "datasets/nyt.csv",
        "n_samples": n_samples,
        "split": {"seed": args.seed, **split.sizes},
        "tokenizer": {"tool": "nltk.word_tokenize", "lowercase": LOWERCASE},
        "vocab_size": len(vocab),
        "class_names": class_names,
        "hyperparameters": {"model": "LogisticRegression", "C": args.C, "max_iter": args.max_iter},
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 冒烟测试结果另存，避免覆盖正式实验结果
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = RESULTS_DIR / f"task1_results{suffix}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
