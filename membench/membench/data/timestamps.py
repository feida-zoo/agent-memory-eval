"""数据集时间戳归一化：把各原始数据集的时间戳统一成 ISO 8601。

统一的是**格式**，不是**精度**：

- 原始只到日的仍然是 ``YYYY-MM-DD``（不补 00:00）；
- 原始只到分的仍然是 ``YYYY-MM-DDTHH:MM``（不补秒）；
- 原始到秒的保留到秒；
- 原始没有时区就不造时区（全部为 naive 本地时间）。

已知形态（2026-10 对原始数据集逐值实测）::

    ISO 8601        2025-01-05T10:00:00 / 2026-04-06T09:00 / 2024-03-01
                    （内置 sample / office，GateMem；空格分隔会被规范成 T）
    LongMemEval     2023/04/10 (Mon) 17:50     （星期可选；给出时校验与日期一致）
    LoCoMo          1:56 pm on 8 May, 2023     （12 小时制，含 12:xx am/pm 边界）
    BEAM            March-01-2024              （只到日；本集锚点极稀疏）

不认识的形态直接抛 :class:`ValueError`（报错带原值与来源）——宁可让
``dataset fetch`` 失败，也不让"混格式"静默进入数据集：混进去之后，适配器
的日期解析会在个别题上无声失效，比 fetch 期报错难查得多。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

__all__ = ["normalize_timestamp", "TIMESTAMP_FORMATS_NOTE"]

#: 报错信息里列出的已知形态（也供文档复用）。
TIMESTAMP_FORMATS_NOTE = (
    "ISO 8601（YYYY-MM-DD[THH:MM[:SS]]）"
    " / YYYY/MM/DD (Www) HH:MM"
    " / H:MM am|pm on D Month, YYYY"
    " / Month-DD-YYYY"
)

_MONTHS: dict[str, int] = {
    name: idx
    for idx, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}

#: ISO 8601：日期 / 到分 / 到秒；分隔符 ``T`` 或空格（统一输出为 ``T``）。
_ISO_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})"
    r"(?:[T ](?P<h>\d{2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?)?$"
)

#: LongMemEval：``2023/04/10 (Mon) 17:50``（星期可选）。
_SLASH_RE = re.compile(
    r"^(?P<y>\d{4})/(?P<mo>\d{2})/(?P<d>\d{2})"
    r"(?: \((?P<wd>[A-Za-z]{3})\))? (?P<h>\d{2}):(?P<mi>\d{2})$"
)

#: LoCoMo：``1:56 pm on 8 May, 2023``。
_PROSE_RE = re.compile(
    r"^(?P<h12>\d{1,2}):(?P<mi>\d{2}) (?P<ap>[AaPp][Mm]) on "
    r"(?P<d>\d{1,2}) (?P<mon>[A-Za-z]+), (?P<y>\d{4})$"
)

#: BEAM：``March-01-2024``（只有日期）。
_BEAM_RE = re.compile(r"^(?P<mon>[A-Za-z]+)-(?P<d>\d{1,2})-(?P<y>\d{4})$")


def _invalid(raw: str, where: str, why: str) -> ValueError:
    return ValueError(
        f"无法归一化时间戳 {raw!r}{f'（{where}）' if where else ''}：{why}。"
        f"已知形态：{TIMESTAMP_FORMATS_NOTE}；如需支持新形态，请扩展 "
        "membench/data/timestamps.py 的解析表并补测试。"
    )


def _date_part(raw: str, where: str, y: str, mo: str | int, d: str | int) -> str:
    """校验年月日并返回 ``YYYY-MM-DD``。"""
    try:
        date(int(y), int(mo), int(d))
    except ValueError as exc:
        raise _invalid(raw, where, f"日期不合法（{exc}）") from exc
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _month_number(raw: str, where: str, name: str) -> int:
    month = _MONTHS.get(name)
    if month is None:
        raise _invalid(raw, where, f"月份名不认识：{name!r}")
    return month


def normalize_timestamp(value: Any, *, where: str = "") -> str | None:
    """把任意已知形态的时间戳归一为 ISO 8601 字符串（精度保持原样）。

    Args:
        value: 原始时间戳（``str`` / ``None``；None 或空串归一为 ``None``）。
        where: 出错信息里的来源标注（如 ``"dialogues[2].sessions[0]"``）。

    Returns:
        ISO 8601 字符串，或 ``None``（原本就没有时间戳）。

    Raises:
        ValueError: 非空但形态不认识，或日期/星期等自洽校验失败。
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    match = _ISO_RE.match(raw)
    if match:
        day = _date_part(raw, where, match["y"], match["mo"], match["d"])
        if match["h"] is None:
            return day
        hh, mi = int(match["h"]), int(match["mi"])
        if hh > 23 or mi > 59:
            raise _invalid(raw, where, f"时刻越界（{hh:02d}:{mi:02d}）")
        stamp = f"{day}T{hh:02d}:{mi:02d}"
        if match["s"] is not None:
            ss = int(match["s"])
            if ss > 59:
                raise _invalid(raw, where, f"秒越界（{ss:02d}）")
            stamp += f":{ss:02d}"
        return stamp

    match = _SLASH_RE.match(raw)
    if match:
        day = _date_part(raw, where, match["y"], match["mo"], match["d"])
        if match["wd"] is not None:
            real = date(int(match["y"]), int(match["mo"]), int(match["d"])).strftime("%a")
            if real != match["wd"]:
                raise _invalid(raw, where, f"星期与日期不符（{match['wd']} vs {real}）")
        hh, mi = int(match["h"]), int(match["mi"])
        if hh > 23 or mi > 59:
            raise _invalid(raw, where, f"时刻越界（{hh:02d}:{mi:02d}）")
        return f"{day}T{hh:02d}:{mi:02d}"

    match = _PROSE_RE.match(raw)
    if match:
        month = _month_number(raw, where, match["mon"])
        day = _date_part(raw, where, match["y"], month, match["d"])
        h12, mi = int(match["h12"]), int(match["mi"])
        if not 1 <= h12 <= 12 or mi > 59:
            raise _invalid(raw, where, f"12 小时制时刻越界（{h12}:{mi:02d}）")
        hour = h12 % 12 + (12 if match["ap"].lower() == "pm" else 0)
        return f"{day}T{hour:02d}:{mi:02d}"

    match = _BEAM_RE.match(raw)
    if match:
        month = _month_number(raw, where, match["mon"])
        return _date_part(raw, where, match["y"], month, match["d"])

    raise _invalid(raw, where, "形态不认识")
