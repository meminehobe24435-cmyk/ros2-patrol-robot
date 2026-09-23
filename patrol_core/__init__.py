# -*- coding: utf-8 -*-
"""
patrol_core — 巡检/导览机器人的**核心逻辑层**（纯 Python，零依赖）

分层设计（这是本项目最重要的一个决定）：

    ┌────────────────────────────────────────────┐
    │  patrol_ros/   ROS2 节点层（rclpy 薄封装）  │  ← 接真实机器人
    │  话题 / 服务 / Action / TF / Nav2           │
    └───────────────────┬────────────────────────┘
                        │ 只调用下面的接口
    ┌───────────────────▼────────────────────────┐
    │  patrol_core/  核心逻辑（本包）             │  ← 可完整单测，不需要 ROS2
    │  waypoint  点位与路线                       │
    │  mission   任务编排与状态机                 │
    │  anomaly   异常判定（去抖 + 分级）          │
    │  backend   导航后端抽象（ROS2 / 仿真）      │
    │  report    巡检报告                         │
    └────────────────────────────────────────────┘

为什么要这么分：
  · ROS2 环境重、依赖多，**任务逻辑不该被它绑住**
  · 把导航抽象成 NavBackend 后，"单点失败重试""低电断点续巡""告警去抖"
    这些**现场最容易出问题的地方**都能在毫秒级跑完的单元测试里覆盖
  · 没有 ROS2 也能用 SimNavBackend 端到端跑通整条巡检流程
"""

from .waypoint import (Waypoint, WaypointKind, ActionKind, Pose2D, Route,
                       normalize_angle, order_by_nearest)
from .backend import (NavBackend, NavResult, NavFeedback,
                      SimNavBackend, ScriptedNavBackend)
from .anomaly import (AnomalyDetector, ChannelRule, Alarm, Severity,
                      State as ChannelStateEnum)
from .mission import PatrolMission, State, Progress, StepRecord
from .report import build_report, to_json, to_csv, to_text, save

__version__ = "1.0.0"

__all__ = [
    "Waypoint", "WaypointKind", "ActionKind", "Pose2D", "Route",
    "normalize_angle", "order_by_nearest",
    "NavBackend", "NavResult", "NavFeedback", "SimNavBackend", "ScriptedNavBackend",
    "AnomalyDetector", "ChannelRule", "Alarm", "Severity", "ChannelStateEnum",
    "PatrolMission", "State", "Progress", "StepRecord",
    "build_report", "to_json", "to_csv", "to_text", "save",
]
