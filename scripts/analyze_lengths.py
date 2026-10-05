"""分析 NYT 文档长度，评估 BERT 的 max_len=64 会造成多少截断。

作业规定 BERT 的 maximum sequence length = 64，而 NYT 文档很长
（平均 731 个 nltk token）。这里量化"截断"的严重程度，用于报告的讨论部分。

注意：这里用 nltk token 数估算 BERT wordpiece 数，经验系数
``WORDPIECE_FACTOR``（BERT 会把未登录词继续切分为子词，通常比词数多 25%~35%）。
精确值在装上 transformers 后可用真实 tokenizer 复核。

用法:
    python scripts/analyze_lengths.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOKEN_CACHE = ROOT / "cache" / "nyt_word_tokenize_lower_n11519.pkl"
MAX_LEN = 64
WORDPIECE_FACTOR = 1.3  # nltk token -> BERT wordpiece 的经验放大系数


def main() -> None:
    with TOKEN_CACHE.open("rb") as fh:
        tokenized = pickle.load(fh)

    nltk_lens = np.array([len(t) for t in tokenized], dtype=np.int64)
    wp_lens = np.maximum(1, np.round(nltk_lens * WORDPIECE_FACTOR).astype(np.int64))

    print("=" * 70)
    print(f"NYT 文档长度分布（n={len(nltk_lens)}）")
    print("=" * 70)
    print(f"nltk token 数: 中位数 {np.median(nltk_lens):.0f}, "
          f"均值 {nltk_lens.mean():.0f}, P95 {np.percentile(nltk_lens, 95):.0f}, "
          f"最大 {nltk_lens.max()}")
    print(f"估算 wordpiece: 中位数 {np.median(wp_lens):.0f}, "
          f"均值 {wp_lens.mean():.0f}, P95 {np.percentile(wp_lens, 95):.0f} "
          f"(按系数 {WORDPIECE_FACTOR})")

    print("-" * 70)
    print(f"max_len = {MAX_LEN} 的影响")
    print("-" * 70)
    fits = int((wp_lens <= MAX_LEN).sum())
    print(f"能完整放入 {MAX_LEN} 个 token 的文档: {fits} / {len(wp_lens)} "
          f"({fits / len(wp_lens):.2%})")

    visible = np.minimum(wp_lens, MAX_LEN)
    coverage = visible / wp_lens
    print(f"文档被看到的比例: 中位数 {np.median(coverage):.1%}, "
          f"均值 {coverage.mean():.1%}, 最低 {coverage.min():.1%}")
    print(f"被截掉的 token 占比: {1 - coverage.mean():.1%}")

    # 64 个 token 大致相当于多少个 nltk token / 原始词
    approx_words = MAX_LEN / WORDPIECE_FACTOR
    print(f"{MAX_LEN} 个 wordpiece ≈ {approx_words:.0f} 个 nltk token；"
          f"中位文档 {np.median(nltk_lens):.0f} token -> 只覆盖开头约 "
          f"{approx_words / np.median(nltk_lens):.1%} 的内容")


if __name__ == "__main__":
    main()
