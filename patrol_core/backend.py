# -*- coding: utf-8 -*-
"""
patrol_core.backend — 导航后端抽象

这一层是整个项目"能与 ROS2 解耦"的关键：

    PatrolMission（任务逻辑）只依赖 NavBackend 接口
        ├── Ros2NavBackend   —— 真实 ROS2：Action 调 Nav2 的 NavigateToPose
        └── SimNavBackend    —— 纯 Python 仿真：按速度积分推进，可注入失败/超时

好处：
  · 任务编排、状态机、异常判定、报告生成**全部可以脱离 ROS2 单测**
  · 没有 ROS2 环境也能端到端跑通完整巡检流程（本仓库的 CI 就是这么跑的）
  · 换导航栈（Nav2 / 自研 / 别的中间件）只换后端，上层不动

这个模式和我在别处用的 ISerialLink 是同一个思路：
**把"会变的硬件/中间件"挡在接口后面，把"不变的业务逻辑"露出来。**
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from .waypoint import Pose2D, normalize_angle


class NavResult(str, Enum):
    """一次导航的结果。"""

    SUCCEEDED = "succeeded"
    ABORTED = "aborted"          # 规划失败/被取消
    TIMEOUT = "timeout"          # 超时
    REJECTED = "rejected"        # 目标不可达
    BLOCKED = "blocked"          # 被障碍挡住（现场最常见）


@dataclass
class NavFeedback:
    """导航过程中的反馈（对应 ROS2 Action 的 feedback）。"""

    remaining_m: float = 0.0
    elapsed_s: float = 0.0
    progress: float = 0.0        # 0~1


class NavBackend(ABC):
    """导航后端接口。"""

    @abstractmethod
    def goto(self, target: Pose2D, timeout_s: float) -> NavResult:
        """导航到目标点。阻塞直到成功/失败/超时。"""

    @abstractmethod
    def cancel(self) -> None:
        """取消当前导航（急停/任务中止时调用）。"""

    @abstractmethod
    def pose(self) -> Pose2D:
        """当前位姿。"""

    def distance_to(self, target: Pose2D) -> float:
        return self.pose().distance_to(target)

    def close(self) -> None:                      # noqa: D401
        """释放资源（默认无操作）。"""
        return None


class SimNavBackend(NavBackend):
    """
    纯 Python 仿真后端。

    模拟真实导航会遇到的几件事，用于把手上的逻辑测扎实：
      · **移动需要时间**（按速度与距离算），所以超时逻辑是真的会被触发
      · **可以指定某些点失败**（blocked / aborted / timeout / rejected）
      · **可以注入随机扰动**（复现"偶发失败"这类现场问题）
    """

    def __init__(self,
                 start: Pose2D = Pose2D(0.0, 0.0, 0.0),
                 speed_mps: float = 0.6,
                 fail_at: Optional[dict] = None,
                 seed: int = 20240922,
                 jitter: float = 0.0):
        self._pose = start
        self.speed = speed_mps
        self.fail_at = dict(fail_at or {})     # {点位名: NavResult} 或 {序号: NavResult}
        self.jitter = jitter
        self._rand = random.Random(seed)
        self._cancelled = False
        self.visits: List[str] = []            # 走过的点位名（测试断言用）
        self.calls = 0

    # ---------------------------------------------------------------- 接口
    def goto(self, target: Pose2D, timeout_s: float,
             label: str = "") -> NavResult:
        self.calls += 1
        # 先判后清：如果调用 goto 之前刚被 cancel，这一次应该直接返回 ABORTED，
        # 而不是把取消标志悄悄清掉（原来就是这个顺序错的）
        if self._cancelled:
            self._cancelled = False
            return NavResult.ABORTED
        self._cancelled = False

        dist = self._pose.distance_to(target)
        # 先按距离算需要多久；再和超时比 —— 这样"距离太长导致超时"是自然发生的
        need = (dist / self.speed) if self.speed > 0 else float("inf")
        if need > timeout_s:
            self._pose = self._advance_towards(target, timeout_s * self.speed)
            return NavResult.TIMEOUT

        # 注入失败（按标签或按调用序号）
        forced = self.fail_at.get(label, self.fail_at.get(self.calls))
        if forced is not None:
            return NavResult(forced) if not isinstance(forced, NavResult) else forced

        # 随机扰动：模拟偶发失败
        if self.jitter > 0 and self._rand.random() < self.jitter:
            return NavResult.BLOCKED

        if self._cancelled:
            return NavResult.ABORTED

        self._pose = Pose2D(target.x, target.y, target.yaw)
        if label:
            self.visits.append(label)
        return NavResult.SUCCEEDED

    def cancel(self) -> None:
        self._cancelled = True

    def pose(self) -> Pose2D:
        return self._pose

    # ---------------------------------------------------------------- 辅助
    def _advance_towards(self, target: Pose2D, meters: float) -> Pose2D:
        d = self._pose.distance_to(target)
        if d <= 0:
            return self._pose
        ratio = min(1.0, meters / d)
        return Pose2D(self._pose.x + (target.x - self._pose.x) * ratio,
                      self._pose.y + (target.y - self._pose.y) * ratio,
                      self._pose.yaw)

    def set_pose(self, pose: Pose2D) -> None:
        """直接把机器人放到指定位置（测试用）。"""
        self._pose = pose


class ScriptedNavBackend(NavBackend):
    """
    按脚本走的后端：给一串预设结果，依次返回。

    用于精确测试状态机分支（比如"连续两次 BLOCKED 后应触发重规划/放弃"），
    比用随机数可控得多。
    """

    def __init__(self, results: List[NavResult],
                 start: Pose2D = Pose2D(0.0, 0.0, 0.0)):
        self._results = list(results)
        self._pose = start
        self.idx = 0
        self.targets: List[Pose2D] = []

    def goto(self, target: Pose2D, timeout_s: float, label: str = "") -> NavResult:
        self.targets.append(target)
        if self.idx < len(self._results):
            r = self._results[self.idx]
        else:
            r = NavResult.SUCCEEDED
        self.idx += 1
        if r == NavResult.SUCCEEDED:
            self._pose = Pose2D(target.x, target.y, target.yaw)
        return r

    def cancel(self) -> None:
        pass

    def pose(self) -> Pose2D:
        return self._pose
