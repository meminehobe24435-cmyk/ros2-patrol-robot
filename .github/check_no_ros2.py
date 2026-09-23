# -*- coding: utf-8 -*-
"""
CI 用的检查脚本 2：确认**没有 ROS2 的环境下也不应该崩**。

这个检查本身就是一条设计断言：
    "ROS2 层要与核心逻辑解耦，缺了 ROS2 也要能 import、并给出可读的提示，
     而不是甩一个 ImportError 堆栈出来。"
"""
import sys

sys.path.insert(0, ".")

failures = []


def check(cond, msg):
    if cond:
        print("  [PASS] " + msg)
    else:
        failures.append(msg)
        print("  [FAIL] " + msg)


print("== 无 ROS2 环境的降级行为 ==")

# 1) 核心包必须能 import（它对 ROS2 零依赖）
import patrol_core                                            # noqa: E402
check(hasattr(patrol_core, "PatrolMission"), "patrol_core 可 import（零 ROS2 依赖）")

# 2) patrol_ros 也必须能 import —— 不能因为缺 rclpy 就在 import 阶段炸掉
import patrol_ros                                             # noqa: E402
check(hasattr(patrol_ros, "route_from_dict"), "patrol_ros 可 import")

from patrol_ros.nav_backend import HAS_ROS2, Ros2NavBackend    # noqa: E402
check(HAS_ROS2 is False, "CI 环境确实没有 ROS2（HAS_ROS2=False）")

# 3) 用 ROS2 后端时应给出**明确提示**，而不是难懂的报错
try:
    Ros2NavBackend()
except RuntimeError as exc:
    msg = str(exc)
    check("ROS2" in msg and "SimNavBackend" in msg,
          "占位实现给出明确提示：%s" % msg.splitlines()[0])
except Exception as exc:                                        # noqa: BLE001
    check(False, "抛出的不是 RuntimeError 而是 %s: %s" % (type(exc).__name__, exc))
else:
    check(False, "无 ROS2 时应当抛 RuntimeError")

# 4) 路线配置解析（ROS2 层里唯一不依赖 rclpy 的部分）必须可用
try:
    import json
    from patrol_ros.patrol_node import detector_from_route
    with open("config/patrol_route.json", encoding="utf-8") as f:
        r = patrol_ros.route_from_dict(json.load(f))
    check(len(r) == 6, "无 ROS2 时仍能解析路线配置（%d 个点）" % len(r))
    check(len(detector_from_route(r).channels) == 5, "仍能自动生成检测通道")
except Exception as exc:                                        # noqa: BLE001
    check(False, "解析路线配置失败：%s" % exc)

print()
if failures:
    print("共 %d 项失败：" % len(failures))
    for f in failures:
        print("  - " + f)
    raise SystemExit(1)
print("全部检查通过。")
