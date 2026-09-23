# -*- coding: utf-8 -*-
"""
patrol_core.waypoint — 点位模型与路线定义

「巡检」和「导览」在机器人层面其实是同一件事：
    **按顺序走到若干点位，到点执行一个动作，然后去下一个点。**
差别只在"到点做什么"：
    · 巡检：读表 / 拍照 / 测温，判断是否超限，超限就告警
    · 导览：播报讲解词，等待游客，讲完去下一个展点

所以这里用同一套模型描述两种任务，只靠 kind 区分。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class WaypointKind(str, Enum):
    """点位类型。"""

    PATROL = "patrol"       # 巡检点：到点执行检测
    GUIDE = "guide"         # 导览点：到点播报讲解
    CHARGE = "charge"       # 充电桩
    HOME = "home"           # 原点/待命点
    TRANSIT = "transit"     # 纯路过点（不执行动作，只用于绕障）


class ActionKind(str, Enum):
    """到点后要执行的动作类型。"""

    NONE = "none"           # 不做事（路过点）
    CAPTURE = "capture"     # 拍照
    THERMAL = "thermal"     # 测温
    METER = "meter"         # 读表（如指针表/数字表）
    SPEAK = "speak"         # 语音播报
    WAIT = "wait"           # 等待（等人/等稳定）


@dataclass(frozen=True)
class Pose2D:
    """二维位姿（米 / 弧度）。ROS2 里对应 geometry_msgs/Pose2D。"""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def yaw_to(self, other: "Pose2D") -> float:
        """从本点朝 other 的朝向（弧度，归一化到 (-pi, pi]）。"""
        return normalize_angle(math.atan2(other.y - self.y, other.x - self.x))


def normalize_angle(a: float) -> float:
    """把角度归一化到 (-pi, pi]。"""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


@dataclass
class Waypoint:
    """一个点位。"""

    name: str
    pose: Pose2D
    kind: WaypointKind = WaypointKind.PATROL
    action: ActionKind = ActionKind.NONE
    dwell_s: float = 3.0                 # 到点停留时长（秒）
    # 该点要检测的项目：{项目名: 上限}，超上限即告警（巡检用）
    thresholds: Dict[str, float] = field(default_factory=dict)
    # 告警判定需要的**下限**（如温度过低、电量不足）
    lower_bounds: Dict[str, float] = field(default_factory=dict)
    speech: str = ""                     # 导览讲解词
    retry: int = 1                       # 导航失败重试次数
    timeout_s: float = 60.0              # 单点导航超时

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("点位名不能为空")
        if self.dwell_s < 0:
            raise ValueError("停留时长不能为负")
        if self.retry < 0:
            raise ValueError("重试次数不能为负")
        if self.timeout_s <= 0:
            raise ValueError("超时必须为正")

    @property
    def is_action_point(self) -> bool:
        """是否需要执行动作（路过点不算）。"""
        return self.action != ActionKind.NONE

    def __repr__(self) -> str:
        return "Waypoint(%s, %.2f/%.2f, %s/%s)" % (
            self.name, self.pose.x, self.pose.y, self.kind.value, self.action.value)


@dataclass
class Route:
    """一条路线（巡检路线或导览路线）。"""

    name: str
    waypoints: List[Waypoint] = field(default_factory=list)
    loop: bool = True                    # 是否循环
    max_laps: int = 0                    # 0 表示不限圈数（配合 loop）
    return_home: bool = True             # 结束后是否回原点
    kind: WaypointKind = WaypointKind.PATROL

    def add(self, wp: Waypoint) -> "Route":
        if any(w.name == wp.name for w in self.waypoints):
            raise ValueError("点位名重复：%s" % wp.name)
        self.waypoints.append(wp)
        return self

    def __len__(self) -> int:
        return len(self.waypoints)

    def total_path_length(self) -> float:
        """按当前顺序累计路径长度（不含返航段）。"""
        total = 0.0
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            total += a.pose.distance_to(b.pose)
        if self.loop and len(self.waypoints) > 1:
            total += self.waypoints[-1].pose.distance_to(self.waypoints[0].pose)
        return total

    def validate(self) -> List[str]:
        """静态检查，返回问题列表（空表示没问题）。"""
        problems: List[str] = []
        if not self.waypoints:
            problems.append("路线为空")
        # loop=True 却不限圈数 = 无限循环。真实部署里这意味着机器人一直跑下去，
        # 必须显式给出圈数才允许循环。
        if self.loop and self.max_laps <= 0:
            problems.append("循环路线必须指定 max_laps（否则会无限循环）")
        charge = [w for w in self.waypoints if w.kind == WaypointKind.CHARGE]
        if len(charge) > 1:
            problems.append("充电点超过一个：%s" % ", ".join(w.name for w in charge))
        for w in self.waypoints:
            if w.kind == WaypointKind.GUIDE and not w.speech:
                problems.append("导览点缺少讲解词：%s" % w.name)
            if w.kind == WaypointKind.PATROL and not w.thresholds and not w.lower_bounds \
                    and w.action in (ActionKind.NONE,):
                problems.append("巡检点没有任何检测项：%s" % w.name)
            if w.timeout_s <= 0:
                problems.append("超时非法：%s" % w.name)
        return problems


def order_by_nearest(route: Route, start: Pose2D) -> Route:
    """
    最近邻排序（贪心），用于**缩短巡检路程**。

    现场意义：巡检点一多，"按录入顺序走"会绕很多冤枉路。
    贪心最近邻是 O(n²)，对几十个点足够，且不需要 TSP 精确解
    （真实场景还要考虑通道宽度与单行线，精确最优反而没意义）。
    """
    remaining = list(route.waypoints)
    ordered: List[Waypoint] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda w: cur.distance_to(w.pose))
        ordered.append(nxt)
        remaining.remove(nxt)
        cur = nxt.pose
    out = Route(name=route.name + "(nearest)", waypoints=ordered,
                loop=route.loop, max_laps=route.max_laps,
                return_home=route.return_home, kind=route.kind)
    return out
