# -*- coding: utf-8 -*-
"""
patrol_ros.patrol_node — 巡检/导览主节点

职责（薄封装，逻辑都在 patrol_core）：
  · 参数：路线文件、是否循环、圈数、低电量阈值
  · 订阅：/battery_state（电量）、/patrol/start（启动/暂停/继续命令）
  · 发布：/patrol/state（状态）、/patrol/alarm（告警）、/patrol/progress（进度）
  · 服务：/patrol/query（查询当前状态与最近告警）、/patrol/report（导出报告）
  · Action：可选，把"跑一趟巡检"暴露成 Action 供上位机调用

⚠️ 本文件**未在本机运行验证**（开发环境无 ROS2），是照 rclpy 接口规范编写的。
   可单测的部分（任务编排/状态机/异常判定/报告）全部放在 patrol_core，
   由 tests/ 的 46 项单测覆盖。
"""
from __future__ import annotations

import json
import threading
from typing import Optional

from patrol_core import (ActionKind, AnomalyDetector, ChannelRule, PatrolMission,
                         Pose2D, Route, State, Waypoint, WaypointKind,
                         build_report, to_json, to_text)
from patrol_core.report import save

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Bool
    from std_srvs.srv import Trigger

    HAS_ROS2 = True
except Exception:                        # noqa: BLE001
    HAS_ROS2 = False


def route_from_dict(data: dict) -> Route:
    """
    从 JSON/dict 加载路线。

    把路线做成**外部配置**而不是写死在代码里，是现场的基本要求：
    换一条巡检路线只需要改 JSON，不用重新编译部署。
    """
    kind_map = {k.value: k for k in WaypointKind}
    act_map = {a.value: a for a in ActionKind}

    r = Route(name=data.get("name", "未命名路线"),
              loop=bool(data.get("loop", False)),
              max_laps=int(data.get("max_laps", 1)),
              return_home=bool(data.get("return_home", True)),
              kind=kind_map.get(data.get("kind", "patrol"), WaypointKind.PATROL))

    for w in data.get("waypoints", []):
        pose = w.get("pose", {})
        r.add(Waypoint(
            name=w["name"],
            pose=Pose2D(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
                        float(pose.get("yaw", 0.0))),
            kind=kind_map.get(w.get("kind", "patrol"), WaypointKind.PATROL),
            action=act_map.get(w.get("action", "none"), ActionKind.NONE),
            dwell_s=float(w.get("dwell_s", 3.0)),
            thresholds={k: float(v) for k, v in (w.get("thresholds") or {}).items()},
            lower_bounds={k: float(v) for k, v in (w.get("lower_bounds") or {}).items()},
            speech=w.get("speech", ""),
            retry=int(w.get("retry", 1)),
            timeout_s=float(w.get("timeout_s", 60.0)),
        ))
    return r


def detector_from_route(route: Route, debounce_up: int = 2,
                        debounce_down: int = 3) -> AnomalyDetector:
    """按路线里声明的检测项自动建检测器（不用重复配一遍）。"""
    d = AnomalyDetector()
    seen = set()
    for w in route.waypoints:
        for item, up in w.thresholds.items():
            if item in seen:
                continue
            seen.add(item)
            d.add_rule(ChannelRule(item, upper=up, debounce_up=debounce_up,
                                   debounce_down=debounce_down))
        for item, lo in w.lower_bounds.items():
            if item in seen:
                continue
            seen.add(item)
            d.add_rule(ChannelRule(item, lower=lo, debounce_up=debounce_up,
                                   debounce_down=debounce_down))
    return d


if HAS_ROS2:

    class PatrolNode(Node):
        """巡检/导览主节点。"""

        def __init__(self) -> None:
            super().__init__("patrol_node")

            # ---------------- 参数 ----------------
            self.declare_parameter("route_file", "")
            self.declare_parameter("low_battery_ratio", 0.2)
            self.declare_parameter("debounce_up", 2)
            self.declare_parameter("debounce_down", 3)
            self.declare_parameter("report_prefix", "/tmp/patrol_report")

            route_file = self.get_parameter("route_file").value
            self.low_battery_ratio = float(self.get_parameter("low_battery_ratio").value)
            deb_up = int(self.get_parameter("debounce_up").value)
            deb_down = int(self.get_parameter("debounce_down").value)
            self.report_prefix = self.get_parameter("report_prefix").value

            if route_file:
                with open(route_file, encoding="utf-8") as f:
                    route = route_from_dict(json.load(f))
            else:
                route = self._default_route()
                self.get_logger().warn("未提供 route_file，使用内置示例路线")

            self.route = route
            self.detector = detector_from_route(route, deb_up, deb_down)

            # ---------------- 通讯 ----------------
            self.backend = None            # 由 launch 注入（ROS2 或仿真）
            self.mission: Optional[PatrolMission] = None
            self.battery_ratio = 1.0
            self._lock = threading.Lock()

            self.pub_state = self.create_publisher(String, "patrol/state", 10)
            self.pub_alarm = self.create_publisher(String, "patrol/alarm", 10)
            self.pub_progress = self.create_publisher(String, "patrol/progress", 10)

            self.create_subscription(Bool, "patrol/command/start",
                                     self._on_start, 10)
            self.create_subscription(Bool, "patrol/command/abort",
                                     self._on_abort, 10)

            self.create_service(Trigger, "patrol/query", self._on_query)
            self.create_service(Trigger, "patrol/report", self._on_report)

            self.create_timer(1.0, self._tick)
            self.get_logger().info("巡检节点已启动，路线：%s（%d 个点）"
                                   % (route.name, len(route)))

        # ------------------------------------------------------------ 默认路线
        @staticmethod
        def _default_route() -> Route:
            r = Route(name="默认巡检路线", loop=False, return_home=True)
            r.add(Waypoint("原点", Pose2D(0, 0, 0), WaypointKind.HOME))
            r.add(Waypoint("点1", Pose2D(5, 0, 0), action=ActionKind.THERMAL,
                           thresholds={"温度1": 60.0}))
            r.add(Waypoint("点2", Pose2D(5, 5, 0), action=ActionKind.METER,
                           thresholds={"电流1": 100.0}))
            r.add(Waypoint("充电桩", Pose2D(0, 5, 0), WaypointKind.CHARGE))
            return r

        # ------------------------------------------------------------ 回调
        def set_backend(self, backend) -> None:
            """由 launch / main 注入导航后端（真实的或仿真的）。"""
            self.backend = backend

        def _on_start(self, msg) -> None:
            if msg.data:
                self.start_patrol()
            else:
                self.abort_patrol()

        def _on_abort(self, msg) -> None:
            if msg.data:
                self.abort_patrol()

        def start_patrol(self) -> None:
            if self.backend is None:
                self.get_logger().error("未注入导航后端，无法开始巡检")
                return
            if self.mission is not None and self.mission.state not in (
                    State.DONE, State.IDLE, State.PAUSED):
                self.get_logger().warn("已有巡检在进行中")
                return

            with self._lock:
                self.mission = PatrolMission(
                    self.route, self.backend, self.detector,
                    reader=self._read_sensor,
                    speaker=self._speak,
                    low_battery_ratio=self.low_battery_ratio)
            self.get_logger().info("开始巡检")
            # 在后台线程跑，避免阻塞 executor
            threading.Thread(target=self._run_mission, daemon=True).start()

        def abort_patrol(self) -> None:
            if self.mission is not None:
                self.mission.abort()
                self.get_logger().warn("收到中止请求（将记录断点，可续巡）")

        # ------------------------------------------------------------ 传感器/播报
        def _read_sensor(self, item: str) -> float:
            """
            读一个检测项。

            真实部署时这里应该：
              · 从**最新的缓存值**取（话题回调里更新），而不是现去等一帧
              · 缓存过期（比如 5 秒没更新）要抛异常，让上层记为"读数失败"
            这里给出骨架，具体话题名按现场改。
            """
            with self._lock:
                v = getattr(self, "_latest", {}).get(item)
            if v is None:
                raise IOError("检测项 %s 暂无数据（话题未接入或已超时）" % item)
            return float(v)

        def update_sensor(self, item: str, value: float) -> None:
            """供传感器订阅回调调用：更新缓存并喂给检测器。"""
            with self._lock:
                if not hasattr(self, "_latest"):
                    self._latest = {}
                self._latest[item] = float(value)
            alarm = self.detector.update(item, float(value))
            if alarm is not None:
                self._publish_alarm(alarm)

        def _speak(self, text: str) -> None:
            """导览播报：真实部署时发到 TTS 话题/服务。"""
            self.get_logger().info("播报：%s" % text)

        # ------------------------------------------------------------ 发布
        def _publish_alarm(self, alarm) -> None:
            m = String()
            m.data = json.dumps({
                "item": alarm.item, "value": alarm.value,
                "threshold": alarm.threshold,
                "severity": alarm.severity.value,
                "direction": alarm.direction,
            }, ensure_ascii=False)
            self.pub_alarm.publish(m)

        def _tick(self) -> None:
            """周期发布状态与进度。"""
            if self.mission is None:
                return
            s = String()
            s.data = self.mission.state.value
            self.pub_state.publish(s)

            p = String()
            p.data = json.dumps({
                "lap": self.mission.progress.lap,
                "index": self.mission.progress.index,
                "visited": self.mission.progress.visited,
                "succeeded": self.mission.progress.succeeded,
                "failed": self.mission.progress.failed,
                "skipped": self.mission.progress.skipped,
                "alarms": self.mission.progress.alarms,
                "battery": self.battery_ratio,
            }, ensure_ascii=False)
            self.pub_progress.publish(p)

        # ------------------------------------------------------------ 服务
        def _on_query(self, request, response):
            if self.mission is None:
                response.success = False
                response.message = "尚未开始巡检"
                return response
            response.success = True
            response.message = to_text(build_report(self.mission))
            return response

        def _on_report(self, request, response):
            if self.mission is None:
                response.success = False
                response.message = "尚未开始巡检，无报告"
                return response
            rep = build_report(self.mission)
            paths = save(rep, self.report_prefix)
            response.success = True
            response.message = "已导出：" + json.dumps(paths, ensure_ascii=False)
            return response

        # ------------------------------------------------------------ 执行
        def _run_mission(self) -> None:
            try:
                self.mission.run_once(battery_ratio=self.battery_ratio)
            except Exception as exc:                       # noqa: BLE001
                self.get_logger().error("巡检异常终止：%s" % exc)
            finally:
                s = String()
                s.data = self.mission.state.value
                self.pub_state.publish(s)

else:

    class PatrolNode(object):            # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "未检测到 ROS2（rclpy 不可用）。任务逻辑请用 patrol_core 直接跑，"
                "见 demo.py 与 tests/。")
