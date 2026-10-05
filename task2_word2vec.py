"""Task 2: Word Embedding / Word2Vec（50 分）

三组实验，均为 100 维词向量 + 文档向量取平均 + Logistic Regression，
在 NYT Test Set 上报告 Accuracy 与 Macro-F1：

1. **Pre-trained GloVe**：glove.6B 的 ``glove.6B.100d.txt``。
2. **Word2Vec trained on AG News**：在 AG News 文本上自训 100 维 Word2Vec。
3. **Word2Vec trained on NYT**：在 NYT 文本上自训 100 维 Word2Vec。

文档向量按作业要求取文档内所有**有效单词**（在词向量表中查得到的词）的平均：
``e(d) = (1/n) * sum(e(w_i))``。

用法::

    python task2_word2vec.py                       # 三组实验全部运行
    python task2_word2vec.py --only glove nyt      # 只跑指定实验
    python task2_word2vec.py --limit 2000          # 冒烟测试
    python task2_word2vec.py --workers 4           # 加速（结果可能不再严格可复现）
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import numpy as np

from common.data import (
    LABEL_COL,
    RESULTS_DIR,
    TEXT_COL,
    load_ag,
    load_nyt,
    load_split,
)
from common.embeddings import (
    Embedding,
    embedding_from_gensim,
    ensure_glove_txt,
    load_glove,
    train_word2vec,
)
from common.modeling import describe_result, print_summary_table, run_experiment
from common.text import tokenize_corpus

NYT_TOKEN_CACHE = "nyt_word_tokenize_lower"
AG_TOKEN_CACHE = "ag_word_tokenize_lower"
LOWERCASE = True  # 与 GloVe-6B（小写词表）保持一致


def document_vectors(
    embedding: Embedding, tokenized_docs: Sequence[Sequence[str]]
) -> tuple[np.ndarray, int, int]:
    """返回 (文档向量矩阵, 总 token 数, OOV token 数)。"""
    matrix = embedding.document_matrix(tokenized_docs)
    return matrix, embedding.total_tokens, embedding.oov_tokens


def run_embedding_experiment(
    name: str,
    embedding: Embedding,
    data: dict[str, object],
    class_names: Sequence[str],
    args: argparse.Namespace,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """用给定词向量构建文档向量并训练/评价分类器。"""
    print("-" * 78)
    print(f"[{name}] 构建文档向量（词向量平均）...")

    started = time.perf_counter()
    X_train, tot_train, oov_train = document_vectors(embedding, data["train_tokens"])
    X_val, tot_val, oov_val = document_vectors(embedding, data["val_tokens"])
    X_test, tot_test, oov_test = document_vectors(embedding, data["test_tokens"])
    build_seconds = time.perf_counter() - started

    total_tokens = tot_train + tot_val + tot_test
    oov_tokens = oov_train + oov_val + oov_test
    stats: dict[str, object] = {
        "embedding_dim": embedding.dim,
        "embedding_vocab_size": len(embedding),
        "token_coverage": round(1 - oov_tokens / total_tokens, 4) if total_tokens else 0.0,
        "oov_token_ratio": round(oov_tokens / total_tokens, 4) if total_tokens else 0.0,
        "zero_vector_test_docs": int(np.all(X_test == 0, axis=1).sum()),
        "vectorize_seconds": round(build_seconds, 2),
    }
    if extra:
        stats.update(extra)

    print(
        f"[{name}] 文档向量 {X_train.shape}; 词表 {len(embedding)} 词; "
        f"token 覆盖率 {stats['token_coverage']:.2%}（{build_seconds:.1f}s）"
    )

    result = run_experiment(
        representation=name,
        X_train=X_train,
        y_train=data["y_train"],
        X_val=X_val,
        y_val=data["y_val"],
        X_test=X_test,
        y_test=data["y_test"],
        class_names=class_names,
        C=args.C,
        max_iter=args.max_iter,
        extra=stats,
    )
    print(describe_result(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Word2Vec / GloVe + Logistic Regression")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（数据划分与 Word2Vec）")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic Regression 正则强度倒数")
    parser.add_argument("--max-iter", type=int, default=1000, help="最大迭代次数")
    parser.add_argument("--limit", type=int, default=0, help="仅使用前 N 条样本（冒烟测试用）")
    parser.add_argument("--dim", type=int, default=100, help="词向量维度")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["glove", "ag", "nyt"],
        default=["glove", "ag", "nyt"],
        help="只运行指定的实验",
    )
    parser.add_argument(
        "--nyt-scope",
        choices=["train", "all", "both"],
        default="both",
        help="NYT 自训 Word2Vec 使用的语料范围：train=仅训练集（无泄漏，主实验），"
        "all=全部 NYT（附加对照），both=两者都跑",
    )
    parser.add_argument("--w2v-window", type=int, default=5)
    parser.add_argument("--w2v-min-count", type=int, default=5)
    parser.add_argument("--w2v-sg", type=int, default=0, choices=[0, 1], help="0=CBOW, 1=Skip-gram")
    parser.add_argument("--w2v-epochs", type=int, default=5)
    parser.add_argument("--w2v-negative", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Word2Vec 线程数（1 保证可复现）")
    parser.add_argument("--no-token-cache", action="store_true", help="忽略分词缓存")
    args = parser.parse_args()

    # ---------- 1. 数据与固定划分（与 Task 1 完全相同） ----------
    df = load_nyt()
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    n_samples = len(df)
    split = load_split(seed=args.seed, n_samples=n_samples)

    labels = df[LABEL_COL].to_numpy()
    class_names = sorted(set(labels.tolist()))

    print("=" * 78)
    print("Task 2: Word Embedding / Word2Vec + Logistic Regression")
    print("=" * 78)
    print(f"数据集     : datasets/nyt.csv  (n={n_samples})")
    print(
        f"固定划分   : seed={args.seed}  "
        f"train/val/test = {split.sizes['train']}/{split.sizes['val']}/{split.sizes['test']}"
    )
    print(f"类别       : {len(class_names)} 类 -> {class_names}")
    print(
        f"词向量设置 : dim={args.dim}, window={args.w2v_window}, "
        f"min_count={args.w2v_min_count}, sg={args.w2v_sg}, epochs={args.w2v_epochs}, "
        f"negative={args.w2v_negative}, workers={args.workers}"
    )
    print(f"分类器设置 : LogisticRegression(C={args.C}, max_iter={args.max_iter})")

    tokenized = tokenize_corpus(
        df[TEXT_COL].tolist(),
        cache_name=NYT_TOKEN_CACHE,
        lowercase=LOWERCASE,
        use_cache=not args.no_token_cache,
    )
    data: dict[str, object] = {
        "train_tokens": [tokenized[i] for i in split.train],
        "val_tokens": [tokenized[i] for i in split.val],
        "test_tokens": [tokenized[i] for i in split.test],
        "y_train": labels[split.train],
        "y_val": labels[split.val],
        "y_test": labels[split.test],
    }

    results: list[dict[str, object]] = []

    # ---------- 2. 实验一：预训练 GloVe ----------
    if "glove" in args.only:
        print("-" * 78)
        glove_txt = ensure_glove_txt(dim=args.dim)
        glove = load_glove(glove_txt, dim=args.dim)
        results.append(
            run_embedding_experiment(
                f"GloVe-6B-{args.dim}d (pre-trained)",
                glove,
                data,
                class_names,
                args,
                extra={"embedding_source": "glove.6B.zip / glove.6B.100d.txt"},
            )
        )
        del glove

    # ---------- 3. 实验二：在 AG News 上训练 Word2Vec ----------
    if "ag" in args.only:
        print("-" * 78)
        df_ag = load_ag()
        if args.limit:
            df_ag = df_ag.iloc[: args.limit].reset_index(drop=True)
        ag_tokens = tokenize_corpus(
            df_ag[TEXT_COL].tolist(),
            cache_name=AG_TOKEN_CACHE,
            lowercase=LOWERCASE,
            use_cache=not args.no_token_cache,
        )
        model = train_word2vec(
            ag_tokens,
            dim=args.dim,
            window=args.w2v_window,
            min_count=args.w2v_min_count,
            sg=args.w2v_sg,
            epochs=args.w2v_epochs,
            negative=args.w2v_negative,
            workers=args.workers,
            seed=args.seed,
            name="AG News",
        )
        ag_embedding = embedding_from_gensim(model, f"Word2Vec-{args.dim}d (AG News)")
        results.append(
            run_embedding_experiment(
                f"Word2Vec-{args.dim}d (AG News)",
                ag_embedding,
                data,
                class_names,
                args,
                extra={
                    "embedding_source": "自训于 datasets/ag.csv",
                    "w2v_train_docs": len(ag_tokens),
                    "w2v_train_tokens": int(sum(len(s) for s in ag_tokens)),
                },
            )
        )
        del ag_tokens, model, ag_embedding

    # ---------- 4. 实验三：在 NYT 上训练 Word2Vec ----------
    if "nyt" in args.only:
        scopes = []
        if args.nyt_scope in ("train", "both"):
            scopes.append("train")
        if args.nyt_scope in ("all", "both"):
            scopes.append("all")

        for scope in scopes:
            sentences = data["train_tokens"] if scope == "train" else tokenized
            scope_desc = "NYT train split" if scope == "train" else "NYT full"
            print("-" * 78)
            model = train_word2vec(
                sentences,
                dim=args.dim,
                window=args.w2v_window,
                min_count=args.w2v_min_count,
                sg=args.w2v_sg,
                epochs=args.w2v_epochs,
                negative=args.w2v_negative,
                workers=args.workers,
                seed=args.seed,
                name=scope_desc,
            )
            nyt_embedding = embedding_from_gensim(model, f"Word2Vec-{args.dim}d (NYT/{scope})")
            results.append(
                run_embedding_experiment(
                    f"Word2Vec-{args.dim}d ({scope_desc})",
                    nyt_embedding,
                    data,
                    class_names,
                    args,
                    extra={
                        "embedding_source": f"自训于 NYT（{scope_desc}）",
                        "w2v_train_docs": len(sentences),
                        "w2v_train_tokens": int(sum(len(s) for s in sentences)),
                    },
                )
            )
            del model, nyt_embedding

    # ---------- 5. 汇总 ----------
    print_summary_table(results)

    payload = {
        "task": "task2_word2vec",
        "dataset": "datasets/nyt.csv",
        "n_samples": n_samples,
        "split": {"seed": args.seed, **split.sizes},
        "tokenizer": {"tool": "nltk.word_tokenize", "lowercase": LOWERCASE},
        "class_names": class_names,
        "document_vector": "mean of in-vocabulary word vectors (OOV skipped)",
        "hyperparameters": {
            "model": "LogisticRegression",
            "C": args.C,
            "max_iter": args.max_iter,
            "word2vec": {
                "vector_size": args.dim,
                "window": args.w2v_window,
                "min_count": args.w2v_min_count,
                "sg": args.w2v_sg,
                "epochs": args.w2v_epochs,
                "negative": args.w2v_negative,
                "workers": args.workers,
                "seed": args.seed,
            },
        },
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = RESULTS_DIR / f"task2_results{suffix}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
