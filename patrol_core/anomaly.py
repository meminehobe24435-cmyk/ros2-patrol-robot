# -*- coding: utf-8 -*-
"""
patrol_core.anomaly — 异常判定与告警分级

现场真正难的不是"读到一个超标值"，而是**判断它到底算不算异常**：
  · 传感器抖动会瞬间超限 —— 直接报会刷屏、运维就不看了
  · 真故障往往是"连续多次超限"或"持续超限超过 N 秒"
  · 不同项目的严重程度不一样（温度超 5℃ 和 超 30℃ 不是一回事）

所以这里做三件事：
  1) **去抖**：连续 N 次超限才确认（避免单点毛刺）
  2) **分级**：按超出幅度分 WARNING / CRITICAL
  3) **恢复判定**：恢复正常也要连续 N 次，避免告警反复横跳
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class State(str, Enum):
    NORMAL = "normal"
    PENDING = "pending"       # 已超限但还没连续够次数（去抖中）
    ALARMING = "alarming"
    RECOVERING = "recovering" # 已恢复但还没连续够次数


@dataclass
class Alarm:
    """一条告警。"""

    item: str
    value: float
    threshold: float
    severity: Severity
    direction: str            # "high" / "low"
    message: str = ""

    def __repr__(self) -> str:
        return "Alarm(%s=%.3f %s %.3f, %s)" % (
            self.item, self.value, ">" if self.direction == "high" else "<",
            self.threshold, self.severity.value)


@dataclass
class ChannelRule:
    """一个检测项的判定规则。"""

    item: str
    upper: Optional[float] = None          # 上限，超过即超限
    lower: Optional[float] = None          # 下限，低于即超限
    critical_ratio: float = 1.5            # 超出幅度的多少倍算 CRITICAL
    critical_abs: Optional[float] = None   # 绝对阈值（如温度 > 80 直接 CRITICAL）
    debounce_up: int = 3                   # 连续多少次超限才告警
    debounce_down: int = 3                 # 连续多少次恢复才解除
    unit: str = ""

    def check(self, value: float) -> Optional[Alarm]:
        """单次判定（不做去抖）。返回 Alarm 或 None。"""
        if self.upper is not None and value > self.upper:
            return Alarm(self.item, value, self.upper,
                         self._severity(value, self.upper, "high"),
                         "high", self._msg(value, "超过上限"))
        if self.lower is not None and value < self.lower:
            return Alarm(self.item, value, self.lower,
                         self._severity(value, self.lower, "low"),
                         "low", self._msg(value, "低于下限"))
        return None

    def _severity(self, value: float, limit: float, direction: str) -> Severity:
        if self.critical_abs is not None:
            if direction == "high" and value >= self.critical_abs:
                return Severity.CRITICAL
            if direction == "low" and value <= self.critical_abs:
                return Severity.CRITICAL
        span = abs(limit) if abs(limit) > 1e-9 else 1.0
        excess = abs(value - limit)
        return Severity.CRITICAL if excess >= span * (self.critical_ratio - 1.0) \
            else Severity.WARNING

    def _msg(self, value: float, what: str) -> str:
        return "%s %.3f%s %s %.3f%s" % (
            self.item, value, self.unit, what,
            self.upper if "上限" in what else self.lower, self.unit)


@dataclass
class ChannelState:
    """一个通道的运行状态（去抖计数等）。"""

    rule: ChannelRule
    state: State = State.NORMAL
    over_count: int = 0
    ok_count: int = 0
    last_value: Optional[float] = None
    alarm: Optional[Alarm] = None
    alarm_count: int = 0          # 累计确认告警次数


class AnomalyDetector:
    """
    多通道异常检测器（带去抖与恢复判定）。

    用法：
        det = AnomalyDetector()
        det.add_rule(ChannelRule("温度", upper=60, unit="℃", critical_abs=80))
        for v in readings: det.update("温度", v)   # 返回本通道的当前告警或 None
    """

    def __init__(self) -> None:
        self._ch: Dict[str, ChannelState] = {}
        self._events: List[Alarm] = []       # 确认过的告警历史

    def add_rule(self, rule: ChannelRule) -> None:
        if rule.item in self._ch:
            raise ValueError("通道已存在：%s" % rule.item)
        if rule.upper is None and rule.lower is None:
            raise ValueError("通道 %s 至少要有一个上下限" % rule.item)
        if rule.debounce_up < 1 or rule.debounce_down < 1:
            raise ValueError("去抖次数必须 >= 1")
        self._ch[rule.item] = ChannelState(rule=rule)

    @property
    def channels(self) -> List[str]:
        return list(self._ch.keys())

    def state_of(self, item: str) -> State:
        return self._ch[item].state

    def alarm_of(self, item: str) -> Optional[Alarm]:
        return self._ch[item].alarm

    @property
    def events(self) -> List[Alarm]:
        return list(self._events)

    def alarm_count_of(self, item: str) -> int:
        """该通道累计确认过的告警次数。

        注意：state_of() 返回的是**状态枚举**，拿不到计数 ——
        这是本项目里踩到的一个命名歧义（State vs ChannelState），
        所以单独提供一个明确命名的访问器。
        """
        return self._ch[item].alarm_count

    def active_alarms(self) -> List[Alarm]:
        """
        当前生效的告警。
        **RECOVERING（恢复去抖中）也算有告警** ——
        这正是"恢复去抖"的意义：在连续 N 次确认恢复正常之前，警报不能撤。
        """
        return [c.alarm for c in self._ch.values()
                if c.state in (State.ALARMING, State.RECOVERING)
                and c.alarm is not None]

    def update(self, item: str, value: float) -> Optional[Alarm]:
        """
        喂一个读数。返回**当前生效**的告警（None 表示正常）。
        只有去抖通过后才会返回告警 —— 单次毛刺会被吃掉。
        """
        if item not in self._ch:
            raise KeyError("未注册的通道：%s" % item)
        c = self._ch[item]
        c.last_value = value
        hit = c.rule.check(value)

        if c.state in (State.NORMAL, State.RECOVERING):
            if hit is not None:
                c.over_count += 1
                c.ok_count = 0
                if c.over_count >= c.rule.debounce_up:
                    c.state = State.ALARMING
                    c.alarm = hit
                    c.alarm_count += 1
                    self._events.append(hit)
                else:
                    c.state = State.PENDING
            else:
                c.over_count = 0
                c.state = State.NORMAL
                c.alarm = None
        elif c.state == State.PENDING:
            if hit is not None:
                c.over_count += 1
                if c.over_count >= c.rule.debounce_up:
                    c.state = State.ALARMING
                    c.alarm = hit
                    c.alarm_count += 1
                    self._events.append(hit)
            else:
                c.over_count = 0
                c.state = State.NORMAL
        elif c.state == State.ALARMING:
            if hit is not None:
                c.alarm = hit                    # 更新为最新值（严重度可能升级）
            else:
                c.ok_count += 1
                if c.ok_count >= c.rule.debounce_down:
                    c.state = State.NORMAL
                    c.alarm = None
                    c.ok_count = 0
                else:
                    c.state = State.RECOVERING

        return c.alarm if c.state in (State.ALARMING, State.RECOVERING) else None

    def summary(self) -> Dict[str, int]:
        """按严重度统计已确认的告警。"""
        out = {s.value: 0 for s in Severity}
        for a in self._events:
            out[a.severity.value] += 1
        return out
