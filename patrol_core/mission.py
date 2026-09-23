# -*- coding: utf-8 -*-
"""
patrol_core.mission — 巡检/导览任务编排与状态机

一次任务要处理的事（也是现场最容易出错的地方）：
  · 按顺序导航到每个点，**单点失败不能整趟崩**（要重试，再不行跳过并记录）
  · 每段导航都要能被打断（急停、低电量、上位机取消）
  · 到点执行动作、读检测项、判异常
  · 电量低要**中断任务去充电**，充完能**断点续巡**（不是从头再来）
  · 全程要有可读的状态与日志，便于现场排查

这里把状态机写成显式的状态 + 转移，而不是一堆 if —— 这样：
  1) 可以画出状态图给客户/同事看
  2) 每个转移都能单测
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from .anomaly import Alarm, AnomalyDetector, ChannelRule
from .backend import NavBackend, NavResult
from .waypoint import ActionKind, Pose2D, Route, Waypoint, WaypointKind


class State(str, Enum):
    """机器人任务状态机。"""

    IDLE = "idle"                # 待命
    PATROLLING = "patrolling"    # 正在前往下一个点
    INSPECTING = "inspecting"    # 到点执行动作/检测
    GUIDING = "guiding"          # 到点播报引导
    PAUSED = "paused"            # 暂停（人工干预）
    LOW_BATTERY = "low_battery"  # 低电量，去充电
    CHARGING = "charging"        # 充电中
    RETURNING = "returning"      # 返航
    FAULT = "fault"              # 故障停机
    DONE = "done"                # 任务完成


@dataclass
class Progress:
    """任务进度。"""

    lap: int = 0
    index: int = 0               # 当前路线中的下标
    visited: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    alarms: int = 0
    resumed_from: int = 0        # 断点续巡的起点

    def reset_index(self) -> None:
        self.index = 0


@dataclass
class StepRecord:
    """单个点位的执行记录（巡检报告的一行）。"""

    waypoint: str
    ok: bool
    attempts: int = 1
    nav_result: str = ""
    dwell_s: float = 0.0
    readings: Dict[str, float] = field(default_factory=dict)
    alarms: List[Alarm] = field(default_factory=list)
    speech: str = ""
    note: str = ""


class PatrolMission:
    """
    一次巡检/导览任务。

    依赖注入：
      · backend：导航后端（ROS2 或仿真）
      · reader ：读数回调 ``reader(item) -> float``；不传则用内置模拟读数
      · speaker：播报回调 ``speaker(text)``；不传则只记录
    这样在真实 ROS2 节点里，reader 从话题取数据、speaker 调 TTS；
    在单测里则传假实现 —— 同一套任务逻辑两边都能跑。
    """

    def __init__(self,
                 route: Route,
                 backend: NavBackend,
                 detector: Optional[AnomalyDetector] = None,
                 reader: Optional[Callable[[str], float]] = None,
                 speaker: Optional[Callable[[str], None]] = None,
                 low_battery_ratio: float = 0.2,
                 resume_ratio: float = 0.8,
                 clock: Optional[Callable[[], float]] = None):
        self.route = route
        self.backend = backend
        self.detector = detector or AnomalyDetector()
        self.reader = reader or (lambda item: 0.0)
        self.speaker = speaker
        self.low_battery_ratio = low_battery_ratio
        self.resume_ratio = resume_ratio
        self._now = clock or time.monotonic

        self.state = State.IDLE
        self.progress = Progress()
        self.records: List[StepRecord] = []
        self.log: List[str] = []
        self.abort_requested = False
        self._pending_resume_index: Optional[int] = None

        problems = route.validate()
        if problems:
            raise ValueError("路线校验失败：" + "；".join(problems))

    # ------------------------------------------------------------ 状态与日志
    def _set_state(self, s: State, why: str = "") -> None:
        self.state = s
        self.log.append("[%s] -> %s%s" % (
            time.strftime("%H:%M:%S"), s.value, ("（%s）" % why) if why else ""))

    @property
    def current(self) -> Optional[Waypoint]:
        if 0 <= self.progress.index < len(self.route.waypoints):
            return self.route.waypoints[self.progress.index]
        return None

    def abort(self) -> None:
        """请求中止（急停/上位机取消）。下一次检查点生效。"""
        self.abort_requested = True
        self.backend.cancel()

    # ------------------------------------------------------------ 主循环
    def run_once(self, battery_ratio: float = 1.0) -> State:
        """
        跑完整条路线（按 max_laps 决定圈数）。

        这里写成**同步但分步**的形式：每一步都检查
        abort / 电量 / 导航结果，便于单测与控制。
        """
        if self.state in (State.PATROLLING, State.INSPECTING, State.GUIDING):
            return self.state

        self._set_state(State.PATROLLING, "开始任务")
        laps = 0
        n = len(self.route.waypoints)
        if n == 0:
            self._set_state(State.DONE, "空路线")
            return self.state

        # 断点续巡：从上次中断的点继续
        if self._pending_resume_index is not None:
            self.progress.index = self._pending_resume_index
            self.progress.resumed_from = self._pending_resume_index
            self._pending_resume_index = None
            self.log.append("从第 %d 个点续巡" % (self.progress.index + 1))

        while True:
            while self.progress.index < n:
                if self.abort_requested:
                    self.abort_requested = False
                    self._set_state(State.PAUSED, "收到中止请求")
                    self._pending_resume_index = self.progress.index
                    return self.state

                if battery_ratio <= self.low_battery_ratio:
                    self._set_state(State.LOW_BATTERY, "电量 %.0f%%" % (battery_ratio * 100))
                    self._pending_resume_index = self.progress.index
                    self._go_charge()
                    battery_ratio = 1.0
                    self._set_state(State.PATROLLING, "充电完成，续巡")

                wp = self.route.waypoints[self.progress.index]
                ok = self._execute_waypoint(wp)
                self.progress.visited += 1
                if ok:
                    self.progress.succeeded += 1
                else:
                    self.progress.failed += 1
                self.progress.index += 1

            laps += 1
            self.progress.lap = laps
            if not self.route.loop:
                break
            if self.route.max_laps and laps >= self.route.max_laps:
                break
            self.progress.index = 0

        if self.route.return_home:
            self._set_state(State.RETURNING, "返航")
            home = next((w for w in self.route.waypoints if w.kind == WaypointKind.HOME), None)
            if home is not None:
                self.backend.goto(home.pose, home.timeout_s)

        self._set_state(State.DONE, "任务完成")
        return self.state

    # ------------------------------------------------------------ 单点执行
    def _execute_waypoint(self, wp: Waypoint) -> bool:
        rec = StepRecord(waypoint=wp.name, ok=False)
        self._set_state(State.PATROLLING, "前往 %s" % wp.name)

        # 导航（带重试）
        result = NavResult.ABORTED
        for attempt in range(1, wp.retry + 2):
            result = self.backend.goto(wp.pose, wp.timeout_s, label=wp.name)
            rec.attempts = attempt
            rec.nav_result = result.value
            if result == NavResult.SUCCEEDED:
                break
            self.log.append("  导航失败（%s），第 %d 次尝试" % (result.value, attempt))

        if result != NavResult.SUCCEEDED:
            rec.ok = False
            rec.note = "导航失败：%s" % result.value
            self.records.append(rec)
            self.progress.skipped += 1
            self.log.append("  跳过 %s（%s）" % (wp.name, result.value))
            return False

        # 到点执行动作
        if wp.action == ActionKind.SPEAK:
            self._set_state(State.GUIDING, "讲解 %s" % wp.name)
            rec.speech = wp.speech
            if self.speaker is not None and wp.speech:
                self.speaker(wp.speech)
        else:
            self._set_state(State.INSPECTING, "检测 %s" % wp.name)

        rec.dwell_s = wp.dwell_s
        self._do_action(wp, rec)
        rec.ok = True
        self.records.append(rec)
        return True

    def _do_action(self, wp: Waypoint, rec: StepRecord) -> None:
        """执行到点动作并读检测项。"""
        items = list(wp.thresholds.keys()) + list(wp.lower_bounds.keys())
        for item in items:
            try:
                v = float(self.reader(item))
            except Exception as exc:                       # noqa: BLE001
                rec.note = (rec.note + "；" if rec.note else "") + "读数失败(%s): %s" % (item, exc)
                continue
            rec.readings[item] = v
            if item in self.detector.channels:
                alarm = self.detector.update(item, v)
                if alarm is not None:
                    rec.alarms.append(alarm)
                    self.progress.alarms += 1

        if wp.action in (ActionKind.CAPTURE, ActionKind.THERMAL, ActionKind.METER):
            rec.note = (rec.note + "；" if rec.note else "") + "已执行 %s" % wp.action.value

    # ------------------------------------------------------------ 充电
    def _go_charge(self) -> None:
        charge = next((w for w in self.route.waypoints
                       if w.kind == WaypointKind.CHARGE), None)
        if charge is None:
            self.log.append("  路线中没有充电点，忽略低电量")
            return
        self._set_state(State.RETURNING, "去充电桩")
        r = self.backend.goto(charge.pose, charge.timeout_s, label=charge.name)
        if r != NavResult.SUCCEEDED:
            self._set_state(State.FAULT, "无法到达充电桩：%s" % r.value)
            raise RuntimeError("无法到达充电桩：%s" % r.value)
        self._set_state(State.CHARGING, "充电中")
