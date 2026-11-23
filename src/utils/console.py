"""Windows 控制台 UTF-8 修复。

Windows 默认 GBK 代码页，打印 ⚠ / ± / 希腊字母（λ、τ、κ 等公式符号）会抛
UnicodeEncodeError。所有脚本在 import 后立刻调用 setup_console()。
"""

from __future__ import annotations

import sys


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 非 TTY 或已重定向，忽略
