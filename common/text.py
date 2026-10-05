"""分词工具：统一使用 nltk.word_tokenize，并带磁盘缓存。

Task 1/2/3 需要完全相同的分词结果，而 nltk 逐句分词较慢，
所以分词只在首次运行时执行，结果缓存到 `cache/` 目录供后续实验复用。

缓存文件名里带有语料标识、文档数与配置（如是否小写），
避免不同设置之间互相覆盖。
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Sequence
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"

_NLTK_RESOURCES = ("tokenizers/punkt", "tokenizers/punkt_tab")


def ensure_nltk_data() -> None:
    """确保 punkt 分词模型已就绪，缺失时自动下载。"""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"[nltk] 缺少 {resource}，正在下载 ...")
            nltk.download(resource.split("/")[-1], quiet=True)


def tokenize_text(text: str, lowercase: bool = True) -> list[str]:
    """把一段文本切成词序列。"""
    if lowercase:
        text = text.lower()
    return nltk.word_tokenize(text)


def tokenize_corpus(
    texts: Sequence[str],
    cache_name: str,
    lowercase: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[list[str]]:
    """对整个语料分词，并把结果缓存到 `cache/<cache_name>.pkl`。

    Args:
        texts: 文档字符串序列。
        cache_name: 缓存标识，例如 ``nyt_word_tokenize_lower``。
        lowercase: 是否先转小写（词袋与 GloVe 均为小写词表，默认 True）。
        use_cache: 设为 False 可强制重新分词。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 文件名带上文档数，避免冒烟测试（如 --limit 2000）覆盖全量语料的分词缓存
    cache_path = CACHE_DIR / f"{cache_name}_n{len(texts)}.pkl"

    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if len(cached) == len(texts):
            if verbose:
                print(f"[分词] 命中缓存 {cache_path.name}（{len(cached)} 篇）")
            return cached
        if verbose:
            print(f"[分词] 缓存条数不符（{len(cached)} != {len(texts)}），重新分词")

    ensure_nltk_data()
    total = len(texts)
    if verbose:
        print(f"[分词] nltk.word_tokenize 处理 {total} 篇文档 ...")

    tokenized: list[list[str]] = []
    started = time.perf_counter()
    step = max(1, total // 10)
    for i, text in enumerate(texts):
        tokenized.append(tokenize_text(text, lowercase=lowercase))
        if verbose and ((i + 1) % step == 0 or i + 1 == total):
            elapsed = time.perf_counter() - started
            print(
                f"[分词] {i + 1}/{total} ({100 * (i + 1) / total:5.1f}%) "
                f"用时 {elapsed:6.1f}s"
            )

    with cache_path.open("wb") as fh:
        pickle.dump(tokenized, fh)
    if verbose:
        print(f"[分词] 已缓存到 {cache_path}")
    return tokenized
