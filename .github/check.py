# -*- coding: utf-8 -*-
"""
CI 用的检查脚本 1：关键行为断言 + 路线配置真跑通。

单独写成文件而不是内联 heredoc —— 因为 GitHub Actions 里
`run: python - <<'PY'` 的 heredoc 内容一旦缩进就会被当成 Python 的
IndentationError（shell 的 heredoc 内容不能缩进，除非用 <<- 配 tab）。
放到文件里更稳，也方便本地直接跑。
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, ".")

from patrol_core import PatrolMission, SimNavBackend, State            # noqa: E402
from patrol_ros.patrol_node import detector_from_route, route_from_dict  # noqa: E402

FAILED = []


def check(cond, msg):
    if cond:
        print("  [PASS] " + msg)
    else:
        FAILED.append(msg)
        print("  [FAIL] " + msg)


# ---------------------------------------------------------------- 1. 单测结果
print("== 1. 单元测试结果 ==")
txt = pathlib.Path("test_output.txt").read_text(encoding="utf-8", errors="replace")
check("FAILED" not in txt and "OK" in txt, "单元测试全通过")

# 关键用例必须存在（防止以后被误删）
must_have = [
    "test_loop_without_max_laps_is_rejected",     # 无限循环必须被拒
    "test_waypoint_failure_retries_then_skips",   # 单点失败要跳过而不是整趟崩
    "test_low_battery_goes_charging_and_resumes",  # 低电回充 + 断点续巡
    "test_abort_then_resume_from_breakpoint",     # 中止后续巡
    "test_debounce_swallows_single_spike_in_patrol",  # 去抖要吞掉单次毛刺
    "test_recovery_debounced",                    # 恢复去抖
    "test_infinite_loop_config_rejected",         # 配置层也要挡住无限循环
    "test_patrol_route_actually_runs",            # 配置能真跑完
]
for k in must_have:
    check(k in txt, "关键用例存在：" + k)

# ---------------------------------------------------------------- 2. 路线配置
print("\n== 2. 路线配置加载并真跑通 ==")
for f in ("config/patrol_route.json", "config/guide_route.json"):
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    r = route_from_dict(data)
    check(r.validate() == [], "%s 校验通过（%d 个点）" % (r.name, len(r)))
    det = detector_from_route(r)
    m = PatrolMission(r, SimNavBackend(speed_mps=2.0), det,
                      reader=lambda item: 10.0)
    st = m.run_once()
    check(st == State.DONE, "%s 跑通，结束状态 %s" % (r.name, st.value))
    check(m.progress.failed == 0, "%s 无失败点" % r.name)

# ---------------------------------------------------------------- 3. 演示输出
print("\n== 3. 演示输出 ==")
d = pathlib.Path("demo_output.txt").read_text(encoding="utf-8", errors="replace")
for key in ("巡检报告", "导览记录", "low_battery", "续巡", "critical"):
    check(key in d, "演示输出包含：" + key)

# ---------------------------------------------------------------- 汇总
print()
if FAILED:
    print("共 %d 项检查失败：" % len(FAILED))
    for f in FAILED:
        print("  - " + f)
    raise SystemExit(1)
print("全部检查通过。")
