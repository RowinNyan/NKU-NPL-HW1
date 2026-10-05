"""逐包探测依赖是否可用，打印版本或失败原因。

用法:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import sys
import traceback

PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "nltk",
    "gensim",
    "torch",
    "transformers",
    "datasets",
    "tqdm",
]

print(f"python  {sys.version.split()[0]}  ({sys.executable})")
print("-" * 72)
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"OK      {name:<14} {version}")
    except BaseException:  # noqa: BLE001 - 探测脚本，任何异常都要报告
        lines = traceback.format_exc().strip().splitlines()
        reason = lines[-1] if lines else "unknown error"
        print(f"FAIL    {name:<14} {reason}")
