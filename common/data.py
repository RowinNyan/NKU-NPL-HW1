"""NYT 数据集的加载与固定划分。

作业要求：先把 NYT 数据集随机打乱，再按 80% / 10% / 10% 切分为
训练 / 验证 / 测试集，且**所有实验必须使用完全相同的划分**，以保证
不同文本表示方法之间的可比性。

因此这里把划分出的**索引**缓存到 `splits/` 目录：任何脚本（Task 1/2/3）
都通过 `load_split()` 读取同一份索引文件，只要 `seed` 不变，划分就完全一致，
即使原始 csv 的行序发生变化也不受影响。

用法::

    from common.data import load_nyt, load_split

    df = load_nyt()
    split = load_split()                 # seed=42, 80/10/10
    train_texts = df["text"].to_numpy()[split.train]
    train_labels = df["label"].to_numpy()[split.train]
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"

TEXT_COL = "text"
LABEL_COL = "label"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass(frozen=True)
class DatasetSplit:
    """三份互不重叠的样本索引，均指向原始 DataFrame 的行号。"""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int = SEED

    def __post_init__(self) -> None:
        total = len(self.train) + len(self.val) + len(self.test)
        unique = len(np.unique(np.concatenate([self.train, self.val, self.test])))
        if unique != total:
            raise ValueError("划分结果存在重叠的样本索引")
        if not (len(self.train) > len(self.val) >= 0 and len(self.test) > 0):
            raise ValueError(f"划分比例异常: {self.sizes}")

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    def subset(self, array: np.ndarray, part: str) -> np.ndarray:
        """按划分取出数组的一部分：`part` ∈ {train, val, test}。"""
        return array[getattr(self, part)]


def load_nyt(path: str | Path | None = None) -> pd.DataFrame:
    """读取 NYT 数据集，并校验 text / label 两列存在。"""
    csv_path = Path(path) if path is not None else DATASETS_DIR / "nyt.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据集: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} 缺少列 {sorted(missing)}，实际列: {list(df.columns)}")

    df = df[[TEXT_COL, LABEL_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def load_ag(path: str | Path | None = None) -> pd.DataFrame:
    """读取 AG News 数据集。

    AG News 只有 text 一列（无标签），本次作业中用于训练 Word2Vec。
    """
    csv_path = Path(path) if path is not None else DATASETS_DIR / "ag.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据集: {csv_path}")

    df = pd.read_csv(csv_path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"{csv_path.name} 缺少列 '{TEXT_COL}'，实际列: {list(df.columns)}")

    df = df[[TEXT_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def make_split(n_samples: int, seed: int = SEED) -> DatasetSplit:
    """对 `range(n_samples)` 做固定种子的随机打乱，再按 80/10/10 切分。"""
    if n_samples <= 0:
        raise ValueError("样本数为 0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)
    # 余数全部归入 test，保证三份之和恰好等于 n_samples
    train = perm[:n_train]
    val = perm[n_train : n_train + n_val]
    test = perm[n_train + n_val :]

    # 排序让划分与语料行序解耦，同时保证下游处理顺序可复现
    return DatasetSplit(
        train=np.sort(train), val=np.sort(val), test=np.sort(test), seed=seed
    )


def split_cache_path(seed: int = SEED, n_samples: int | None = None) -> Path:
    """划分缓存文件路径。

    文件名带上样本数，避免 `--limit` 之类的冒烟测试用小划分覆盖正式划分的缓存。
    """
    suffix = "" if n_samples is None else f"_n{n_samples}"
    return SPLITS_DIR / f"nyt_split_seed{seed}{suffix}.pkl"


def load_split(
    seed: int = SEED, n_samples: int | None = None, use_cache: bool = True
) -> DatasetSplit:
    """返回 NYT 的固定划分；首次调用时计算并缓存索引。

    同一 (seed, n_samples) 组合始终返回完全相同的划分。
    """
    if n_samples is None:
        n_samples = len(load_nyt())

    cache_path = split_cache_path(seed, n_samples)
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fh:
            split: DatasetSplit = pickle.load(fh)
        if sum(split.sizes.values()) == n_samples:
            return split

    split = make_split(n_samples, seed=seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(split, fh)
    return split
