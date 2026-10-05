"""词向量加载与文档向量（词向量平均）计算。

Task 2 需要三种词向量，这里统一封装成 :class:`Embedding`：

* 预训练 GloVe（从 ``glove.6B.zip`` 中解压 ``glove.6B.100d.txt`` 后解析）；
* gensim 训练出的 Word2Vec 模型（AG News / NYT）。

文档向量按作业要求取**文档内所有有效单词词向量的平均**：
``e(d) = (1/n) * sum(e(w_i))``，其中"有效单词"指在词向量表中能查到的词，
查不到的（OOV）直接跳过；若一篇文档一个词都查不到，则文档向量为零向量。
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "embeddings"


class Embedding:
    """一张词向量表，支持把文档转成平均词向量。"""

    def __init__(self, matrix: np.ndarray, index: dict[str, int], name: str) -> None:
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.index = index
        self.name = name
        # 最近一次 document_matrix 的统计，供报告分析 OOV 情况
        self.coverage: float = float("nan")
        self.oov_tokens: int = 0
        self.total_tokens: int = 0

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def document_vector(self, tokens: Sequence[str]) -> np.ndarray:
        """单篇文档的平均词向量（跳过 OOV）。"""
        rows = [self.index[t] for t in tokens if t in self.index]
        if not rows:
            return np.zeros(self.dim, dtype=np.float32)
        return self.matrix[rows].mean(axis=0)

    def document_matrix(
        self, tokenized_docs: Sequence[Sequence[str]], verbose: bool = False
    ) -> np.ndarray:
        """把整个语料转成 ``(n_docs, dim)`` 的文档向量矩阵。"""
        out = np.zeros((len(tokenized_docs), self.dim), dtype=np.float32)
        total = 0
        hits = 0
        step = max(1, len(tokenized_docs) // 5)
        for i, tokens in enumerate(tokenized_docs):
            rows = [self.index[t] for t in tokens if t in self.index]
            total += len(tokens)
            hits += len(rows)
            if rows:
                out[i] = self.matrix[rows].mean(axis=0)
            if verbose and ((i + 1) % step == 0 or i + 1 == len(tokenized_docs)):
                print(f"    文档向量 {i + 1}/{len(tokenized_docs)}")

        self.total_tokens = total
        self.oov_tokens = total - hits
        self.coverage = hits / total if total else 0.0
        return out

    def stats(self) -> dict[str, object]:
        """词表规模与在最近一次语料上的覆盖率。"""
        return {
            "embedding_name": self.name,
            "embedding_dim": self.dim,
            "embedding_vocab_size": len(self),
            "token_coverage": round(self.coverage, 4),
            "oov_token_ratio": round(1.0 - self.coverage, 4),
            "oov_tokens": self.oov_tokens,
            "total_tokens": self.total_tokens,
        }


def ensure_glove_txt(
    zip_path: str | Path | None = None, dim: int = 100, out_dir: str | Path | None = None
) -> Path:
    """确保 ``glove.6B.<dim>d.txt`` 已从 zip 中解压出来，返回其路径。"""
    zip_path = Path(zip_path) if zip_path else ROOT / "glove.6B.zip"
    out_dir = Path(out_dir) if out_dir else EMBEDDINGS_DIR
    member = f"glove.6B.{dim}d.txt"
    target = out_dir / member

    if target.exists() and target.stat().st_size > 0:
        return target
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到 {zip_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GloVe] 从 {zip_path.name} 解压 {member} ...")
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise ValueError(f"{zip_path.name} 中没有 {member}")
        zf.extract(member, out_dir)
    print(f"[GloVe] 已解压到 {target}（{target.stat().st_size / 1e6:.0f} MB）")
    return target


def load_glove(
    txt_path: str | Path, dim: int = 100, use_cache: bool = True, verbose: bool = True
) -> Embedding:
    """解析 GloVe 文本格式的词向量。

    首次解析后会把矩阵与词表缓存为 ``.npy`` / ``.words.txt``，
    之后加载只需约 1 秒（解析 40 万行的原始文本约需 1 分钟）。
    """
    txt_path = Path(txt_path)
    npy_path = txt_path.with_name(txt_path.name + ".vectors.npy")
    words_path = txt_path.with_name(txt_path.name + ".words.txt")

    if use_cache and npy_path.exists() and words_path.exists():
        matrix = np.load(npy_path)
        words = words_path.read_text(encoding="utf-8").splitlines()
        if matrix.shape[0] == len(words):
            if verbose:
                print(f"[GloVe] 命中缓存 {npy_path.name} ({matrix.shape[0]} 词, {dim}d)")
            return Embedding(matrix, {w: i for i, w in enumerate(words)}, txt_path.stem)

    if verbose:
        print(f"[GloVe] 解析 {txt_path.name} ...")

    n_lines = 0
    with txt_path.open("rb") as fh:
        for _ in fh:
            n_lines += 1

    matrix = np.empty((n_lines, dim), dtype=np.float32)
    words: list[str] = []
    with txt_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.split()  # 按任意空白切分，兼容行尾多余空格
            words.append(parts[0])
            matrix[i] = np.asarray(parts[1:], dtype=np.float32)
            if verbose and (i + 1) % 100000 == 0:
                print(f"[GloVe] 已解析 {i + 1}/{n_lines}")

    np.save(npy_path, matrix)
    words_path.write_text("\n".join(words), encoding="utf-8")
    if verbose:
        print(f"[GloVe] 解析完成: {matrix.shape[0]} 词 × {matrix.shape[1]} 维，已缓存")
    return Embedding(matrix, {w: i for i, w in enumerate(words)}, txt_path.stem)


def embedding_from_gensim(model, name: str) -> Embedding:
    """把 gensim Word2Vec 模型包装成 :class:`Embedding`。"""
    wv = model.wv
    return Embedding(wv.vectors, dict(wv.key_to_index), name)


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    dim: int = 100,
    window: int = 5,
    min_count: int = 5,
    sg: int = 0,
    epochs: int = 5,
    negative: int = 5,
    workers: int = 1,
    seed: int = 42,
    name: str = "word2vec",
):
    """在给定语料上训练 Word2Vec。

    默认沿用 gensim / 原论文的常用设置（CBOW，window=5，min_count=5，5 轮，
    negative sampling=5），并把 ``workers`` 默认设为 1 以保证结果可复现。
    """
    from gensim.models import Word2Vec

    print(
        f"[Word2Vec:{name}] 训练中 ... 语料 {len(sentences)} 篇 / "
        f"{sum(len(s) for s in sentences)} tokens, dim={dim}, window={window}, "
        f"min_count={min_count}, sg={sg}, epochs={epochs}, workers={workers}"
    )
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    print(f"[Word2Vec:{name}] 完成，词表 {len(model.wv)} 词")
    return model
