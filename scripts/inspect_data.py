"""数据集结构勘察脚本（只打印结构统计，不转储正文）。

用法:
    python scripts/inspect_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:  # 换下一个编码再试
            last_err = err
    raise RuntimeError(f"无法解码 {path}") from last_err


def describe(name: str, path: Path) -> None:
    print("=" * 72)
    print(f"[{name}] {path.name}  size={path.stat().st_size / 1e6:.1f} MB")
    df = read_csv_any_encoding(path)
    print(f"shape (rows, cols) = {df.shape}")
    print(f"columns           = {list(df.columns)}")
    print(f"dtypes            = {dict(df.dtypes.astype(str))}")
    print(f"missing values    = {df.isna().sum().to_dict()}")

    text_col = "text" if "text" in df.columns else df.columns[0]
    lens = df[text_col].astype(str).str.split().str.len()
    print(
        "token 数(按空白切分) min/mean/median/p95/max = "
        f"{lens.min()} / {lens.mean():.1f} / {lens.median():.0f} / "
        f"{lens.quantile(0.95):.0f} / {lens.max()}"
    )

    if "label" in df.columns:
        vc = df["label"].value_counts().sort_index()
        print(f"类别数 = {df['label'].nunique()}")
        print("类别分布:")
        for label, count in vc.items():
            print(f"    {label!r:>10} : {count:>7} ({count / len(df):6.2%})")
    else:
        print("!! 没有 label 列")

    print(f"正文样例(截断 160 字符): {str(df[text_col].iloc[0])[:160]!r}")


def main() -> None:
    for name in ("nyt", "ag"):
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            describe(name, path)
        else:
            print(f"!! 缺少 {path}")


if __name__ == "__main__":
    main()
