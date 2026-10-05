"""Task 3: Pre-trained Neural Models（20 分）—— BERT 微调

使用 ``google-bert/bert-base-uncased`` 在 NYT 上做文本分类微调，
按作业要求设置 **maximum sequence length = 64**、**3 epochs**，
并在 NYT Test Set 上报告 Accuracy 与 Macro-F1。

指标口径与 Task 1/2 完全一致（复用 ``common.modeling.evaluate_predictions``），
数据划分也复用同一份固定索引，保证三组方法可直接比较。

用法::

    python task3_bert.py                          # 完整实验（3 epochs）
    python task3_bert.py --limit 500 --epochs 1   # 冒烟测试
    python task3_bert.py --max-len 64 --batch-size 32 --lr 2e-5
"""

from __future__ import annotations

import os

# HuggingFace 主站在本机不可达，必须在 import transformers 之前指定镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Windows 默认不支持符号链接，关掉 huggingface_hub 的相应告警
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from common.data import LABEL_COL, RESULTS_DIR, TEXT_COL, load_nyt, load_split  # noqa: E402
from common.modeling import evaluate_predictions  # noqa: E402

MODEL_NAME = "google-bert/bert-base-uncased"
WORK_DIR = Path(__file__).resolve().parent / "cache" / "bert_runs"


class TextClassificationDataset(Dataset):
    """把 tokenizer 的输出包成 torch Dataset。"""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(texts: list[str], tokenizer, max_len: int, verbose: bool = True) -> dict:
    started = time.perf_counter()
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    if verbose:
        print(f"[分词] WordPiece 编码完成，用时 {time.perf_counter() - started:.1f}s")
    return encodings


def measure_lengths(texts: list[str], tokenizer, batch_size: int = 512) -> np.ndarray:
    """不截断地统计每篇文档的 WordPiece 长度，用于量化 max_len 的截断程度。

    分批处理以免一次性把上千万个 token id 留在内存里。
    """
    from transformers.utils import logging as hf_logging

    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # 关掉"序列超过模型上限"的提示

    lengths: list[int] = []
    try:
        for start in range(0, len(texts), batch_size):
            encodings = tokenizer(
                texts[start : start + batch_size], truncation=False, add_special_tokens=True
            )
            lengths.extend(len(ids) for ids in encodings["input_ids"])
            del encodings
    finally:
        hf_logging.set_verbosity(previous)

    return np.asarray(lengths, dtype=np.int64)


def build_training_args(
    args: argparse.Namespace, use_cuda: bool, n_train: int
) -> tuple[TrainingArguments, int]:
    """构造 TrainingArguments，并兼容不同 transformers 版本的参数差异。

    注意：transformers 5.x 移除了 ``warmup_ratio`` / ``overwrite_output_dir``，
    改为显式的 ``warmup_steps``，因此这里按当前版本支持情况动态取舍。
    """
    params = inspect.signature(TrainingArguments.__init__).parameters

    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    kwargs: dict[str, object] = {
        "output_dir": str(WORK_DIR),
        "overwrite_output_dir": True,  # 仅旧版支持，下面会按版本过滤
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "linear",
        "logging_steps": 50,
        "save_strategy": "no",
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": use_cuda,  # RTX 40 系支持 bf16，比 fp16 更稳定
        "fp16": False,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",  # transformers < 4.41 用这个名字
    }
    # 只保留当前版本真正支持的参数
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    # warmup 二选一：优先 ratio，其次 steps
    if "warmup_ratio" in kwargs and "warmup_steps" in kwargs:
        kwargs.pop("warmup_steps")
    return TrainingArguments(**kwargs), warmup_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: BERT fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数（作业规定 3）")
    parser.add_argument("--max-len", type=int, default=64, help="最大序列长度（作业规定 64）")
    parser.add_argument("--batch-size", type=int, default=32, help="训练 batch size")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="评估 batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="仅使用前 N 条样本（冒烟测试用）")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="预训练模型名")
    args = parser.parse_args()

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    print("=" * 78)
    print("Task 3: Pre-trained Neural Models —— BERT 微调")
    print("=" * 78)
    print(f"torch      : {torch.__version__}")
    print(f"设备       : {device}{'（bf16 混合精度）' if use_cuda else '（CPU，将非常慢）'}")
    print(f"预训练模型 : {args.model}")
    print(f"HF 镜像    : {os.environ.get('HF_ENDPOINT')}")

    # ---------- 1. 数据与固定划分（与 Task 1/2 完全相同） ----------
    df = load_nyt()
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    n_samples = len(df)
    split = load_split(seed=args.seed, n_samples=n_samples)

    labels_raw = df[LABEL_COL].tolist()
    class_names = sorted(set(labels_raw))
    label2id = {name: i for i, name in enumerate(class_names)}
    texts = df[TEXT_COL].tolist()

    print(
        f"\n数据集     : datasets/nyt.csv (n={n_samples})  "
        f"train/val/test = {split.sizes['train']}/{split.sizes['val']}/{split.sizes['test']}"
    )
    print(f"类别       : {class_names}")
    print(
        f"训练设置   : epochs={args.epochs}, max_len={args.max_len}, "
        f"batch={args.batch_size}, lr={args.lr}, warmup={args.warmup_ratio}, "
        f"weight_decay={args.weight_decay}"
    )

    # ---------- 2. 分词（BERT 自己的 WordPiece，与 nltk 分词无关） ----------
    print("-" * 78)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[分词器] {args.model}，词表大小 {tokenizer.vocab_size}")

    # 先精确测量真实长度，量化 max_len 造成的截断（报告用）
    started = time.perf_counter()
    full_lengths = measure_lengths(texts, tokenizer)
    visible = np.minimum(full_lengths, args.max_len)
    truncation_stats = {
        "max_seq_length": args.max_len,
        "median_wordpieces": int(np.median(full_lengths)),
        "mean_wordpieces": round(float(full_lengths.mean()), 1),
        "p95_wordpieces": int(np.percentile(full_lengths, 95)),
        "max_wordpieces": int(full_lengths.max()),
        "docs_fitting_entirely": int((full_lengths <= args.max_len).sum()),
        "docs_total": int(len(full_lengths)),
        "mean_visible_ratio": round(float((visible / full_lengths).mean()), 4),
        "median_visible_ratio": round(float(np.median(visible / full_lengths)), 4),
    }
    print(
        f"[长度] WordPiece 中位数 {truncation_stats['median_wordpieces']}，"
        f"能完整放入 {args.max_len} 的文档 "
        f"{truncation_stats['docs_fitting_entirely']}/{truncation_stats['docs_total']}；"
        f"平均只能看到全文的 {truncation_stats['mean_visible_ratio']:.1%}"
        f"（测量用时 {time.perf_counter() - started:.1f}s）"
    )

    train_enc = tokenize([texts[i] for i in split.train], tokenizer, args.max_len)
    val_enc = tokenize([texts[i] for i in split.val], tokenizer, args.max_len)
    test_enc = tokenize([texts[i] for i in split.test], tokenizer, args.max_len)

    y_train = [label2id[labels_raw[i]] for i in split.train]
    y_val = [label2id[labels_raw[i]] for i in split.val]
    y_val_names = np.array([labels_raw[i] for i in split.val])
    y_test_names = np.array([labels_raw[i] for i in split.test])

    train_ds = TextClassificationDataset(train_enc, y_train)
    val_ds = TextClassificationDataset(val_enc, y_val)
    test_ds = TextClassificationDataset(
        test_enc, [label2id[labels_raw[i]] for i in split.test]
    )

    # ---------- 3. 模型 ----------
    print("-" * 78)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(class_names), id2label={i: n for n, i in label2id.items()}
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量 {n_params / 1e6:.1f}M，分类头 num_labels={len(class_names)}")

    # ---------- 4. 训练 ----------
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
        }

    training_args, warmup_steps = build_training_args(args, use_cuda, len(train_ds))
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    print(f"[训练] 总步数 {total_steps}，warmup {warmup_steps} 步（linear schedule）")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("[训练] 开始 ...")
    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    print(f"[训练] 完成，用时 {train_seconds:.1f}s（{train_seconds / 60:.1f} 分钟）")

    epoch_metrics = [
        {
            "epoch": round(entry["epoch"], 2),
            "val_accuracy": round(entry["eval_accuracy"], 4),
            "val_macro_f1": round(entry["eval_macro_f1"], 4),
            "val_loss": round(entry["eval_loss"], 4),
        }
        for entry in trainer.state.log_history
        if "eval_accuracy" in entry
    ]
    print("[训练] 各轮验证集指标:")
    for entry in epoch_metrics:
        print(
            f"    epoch {entry['epoch']}: val Acc={entry['val_accuracy']:.4f} "
            f"Macro-F1={entry['val_macro_f1']:.4f} loss={entry['val_loss']:.4f}"
        )

    # ---------- 5. 测试集评估（口径与 Task 1/2 一致） ----------
    print("-" * 78)
    val_preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)

    # 统一转回字符串标签，才能与 evaluate_predictions 的 class_names 对齐
    val_metrics = evaluate_predictions(
        y_val_names, np.asarray(class_names)[val_preds], class_names
    )
    test_metrics = evaluate_predictions(
        y_test_names, np.asarray(class_names)[test_preds], class_names
    )

    print(
        f"[BERT] val  Acc={val_metrics['accuracy']:.4f} Macro-F1={val_metrics['macro_f1']:.4f} | "
        f"test Acc={test_metrics['accuracy']:.4f} Macro-F1={test_metrics['macro_f1']:.4f}"
    )
    print("[BERT] 测试集分类别指标:")
    print(f"    {'类别':<12}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>10}")
    for label, m in test_metrics["per_class"].items():
        print(
            f"    {label:<12}{m['precision']:>11.4f}{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}{m['support']:>10}"
        )
    cm = test_metrics["confusion_matrix"]
    print("[BERT] 测试集混淆矩阵 (行=真实, 列=预测):")
    print("    " + " " * 12 + "".join(f"{lbl:>12}" for lbl in cm["labels"]))
    for label, row in zip(cm["labels"], cm["matrix"]):
        print(f"    {label:<12}" + "".join(f"{value:>12}" for value in row))

    # ---------- 6. 保存结果 ----------
    payload = {
        "task": "task3_bert",
        "dataset": "datasets/nyt.csv",
        "n_samples": n_samples,
        "split": {"seed": args.seed, **split.sizes},
        "model": args.model,
        "tokenizer": f"{args.model} (WordPiece)",
        "class_names": class_names,
        "device": device,
        "hyperparameters": {
            "epochs": args.epochs,
            "max_seq_length": args.max_len,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "lr_scheduler": "linear",
            "optimizer": "AdamW",
            "precision": "bf16" if use_cuda else "fp32",
            "seed": args.seed,
            "n_parameters": int(n_params),
        },
        "truncation": truncation_stats,
        "train_seconds": round(train_seconds, 2),
        "train_loss": round(float(train_result.training_loss), 4),
        "epoch_metrics": epoch_metrics,
        "results": [
            {
                "representation": f"BERT-base-uncased (max_len={args.max_len}, {args.epochs} epochs)",
                "n_features": int(n_params),
                "n_train": len(train_ds),
                "train_seconds": round(train_seconds, 2),
                "n_iter": None,
                "converged": None,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_per_class": test_metrics["per_class"],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
            }
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = RESULTS_DIR / f"task3_results{suffix}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
