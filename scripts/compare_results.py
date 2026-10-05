"""汇总 Task 1/2/3 的结果，生成横向对比表。

作业要求"能够对不同文本表示方法的性能进行基本比较和分析"，
这个脚本把 results/ 下三份 json 合并成一张表，并输出 Markdown
（可直接粘进报告）到 results/comparison.md。

用法:
    python scripts/compare_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TASK_FILES = [
    ("Task 1 · 词袋", "task1_results.json"),
    ("Task 2 · 词嵌入", "task2_results.json"),
    ("Task 3 · 预训练模型", "task3_results.json"),
]
CLASS_ORDER = ["business", "politics", "sports"]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_label, filename in TASK_FILES:
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"!! 跳过（尚未运行）: {filename}")
            continue
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        for result in payload.get("results", []):
            rows.append(
                {
                    "task": task_label,
                    "name": result["representation"],
                    "accuracy": result["test_accuracy"],
                    "macro_f1": result["test_macro_f1"],
                    "val_accuracy": result.get("val_accuracy"),
                    "val_macro_f1": result.get("val_macro_f1"),
                    "per_class": result.get("test_per_class", {}),
                    "seconds": result.get("train_seconds"),
                }
            )
    return rows


def main() -> None:
    rows = load_rows()
    if not rows:
        print("没有可汇总的结果。")
        return

    best_f1 = max(r["macro_f1"] for r in rows)
    best_acc = max(r["accuracy"] for r in rows)

    lines: list[str] = []
    lines.append("| 表示方法 | Accuracy | Macro-F1 | business F1 | politics F1 | sports F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda x: -x["macro_f1"]):
        per_class = r["per_class"]
        cells = []
        for cls in CLASS_ORDER:
            value = per_class.get(cls, {}).get("f1")
            cells.append(f"{value:.4f}" if value is not None else "-")
        acc = f"{r['accuracy']:.4f}"
        f1 = f"{r['macro_f1']:.4f}"
        if r["accuracy"] == best_acc:
            acc = f"**{acc}**"
        if r["macro_f1"] == best_f1:
            f1 = f"**{f1}**"
        lines.append(
            f"| {r['name']} | {acc} | {f1} | " + " | ".join(cells) + " |"
        )

    table = "\n".join(lines)
    print("=" * 100)
    print("Task 1/2/3 结果对比（NYT Test Set，按 Macro-F1 降序）")
    print("=" * 100)
    print(table)
    print("=" * 100)
    print(f"最佳 Accuracy : {best_acc:.4f}")
    print(f"最佳 Macro-F1 : {best_f1:.4f}")

    # 按任务分组的小结
    print("\n各 Task 最佳结果:")
    for task_label, _ in TASK_FILES:
        group = [r for r in rows if r["task"] == task_label]
        if not group:
            continue
        best = max(group, key=lambda x: x["macro_f1"])
        print(
            f"  {task_label:<20} {best['name']:<44} "
            f"Acc={best['accuracy']:.4f}  Macro-F1={best['macro_f1']:.4f}"
        )

    out_path = RESULTS_DIR / "comparison.md"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Task 1/2/3 结果对比（NYT Test Set）\n\n")
        fh.write(table + "\n")
    print(f"\nMarkdown 已保存: {out_path}")


if __name__ == "__main__":
    main()
