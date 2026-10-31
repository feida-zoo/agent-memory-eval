"""构建时写入的 membench 代码哈希（由 ``skill/build_skill.py`` 生成）。

这个文件**不参与**哈希计算（见 ``membench/version.py`` 的排除规则），否则会自引用。
安装后的 skill 包没有 ``.git``，所以版本标识必须在打包时定稿并硬编码到这里：
运行时直接读常量，报告里带出去即可。

``BUILD_HASH`` 为空表示"还没打包过"（开发态），此时运行时按当前代码现算，
并在报告里标注来源。
"""

from __future__ import annotations

#: 打包时计算出的 membench 包内容指纹（sha256 前 12 位）；空 = 未打包。
BUILD_HASH = "f417401df2db"

#: 该哈希的生成时间（ISO 8601，本地时区）；空 = 未打包。
BUILT_AT = "2026-10-31T14:26:35"
