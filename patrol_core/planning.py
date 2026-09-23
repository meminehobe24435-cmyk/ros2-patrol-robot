# -*- coding: utf-8 -*-
"""
patrol_core.planning — 路径规划与定位融合

对应岗位要求的「定位导航、路径规划、传感器融合」。

三块内容：

1) **全局路径规划**：A*（带八邻域与禁止切角）与 Dijkstra
   —— A* 用启发式快，Dijkstra 用来对照验证（两者结果代价应一致）

2) **路径平滑**：栅格路径是锯齿状的，机器人照着走会左右晃。
   这里做"视线可达"的捷径裁剪（string pulling）与按曲率的稀疏化。

3) **传感器融合定位**：EKF 融合轮式里程计（相对准、会累积漂移）
   与 IMU 航向（短期准、会漂）。这是现场最常用的最小组合。

为什么自己写而不是全交给 Nav2：
   现场调试 90% 的时间花在"路径为什么绕"和"定位为什么飘"上，
   知道这两件事内部怎么算，才谈得上"运行参数优化"（职责③）。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .slam import GridMap

__all__ = [
    "astar", "dijkstra", "plan_path", "smooth_path", "tighten_path",
    "path_length", "EKF2D", "odom_motion_model",
]


# =============================================================== 全局规划
def _neighbors(g: GridMap, gx: int, gy: int, allow_diag: bool = True,
               cost_occ: float = 0.6, cost_unknown: float = 0.5):
    """返回可通行邻居 (nx, ny, 代价)。"""
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if allow_diag:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in dirs:
        nx, ny = gx + dx, gy + dy
        if not g.inside(nx, ny):
            continue
        p = g.prob(nx, ny)
        if p is None:
            continue
        if p >= cost_occ:
            continue                       # 占据格不可通行
        # 禁止"贴着角"斜穿：两侧正交格若有一个是障碍，就不许斜走
        if dx != 0 and dy != 0:
            a = g.prob(gx + dx, gy)
            b = g.prob(gx, gy + dy)
            if (a is not None and a >= cost_occ) or (b is not None and b >= cost_occ):
                continue
        step = math.hypot(dx, dy)
        if p >= cost_unknown:
            step *= 1.5                    # 未知区域加代价，让路径优先走已知区
        yield nx, ny, step


def astar(grid: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
          allow_diag: bool = True) -> Optional[List[Tuple[int, int]]]:
    """
    A* 全局规划。返回栅格路径（含起点与终点），无解返回 None。

    启发式用**八邻域一致的欧氏距离**：
    用曼哈顿距离配八邻域会**高估**，A* 就不再最优，路径会显得"绕"。
    """
    if not grid.inside(*start) or not grid.inside(*goal):
        return None
    if grid.is_occupied(*goal) or grid.is_occupied(*start):
        return None

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    openq = [(h(start, goal), 0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {start: 0.0}
    closed = set()

    while openq:
        _, gcur, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        if cur in closed:
            continue
        closed.add(cur)

        for nx, ny, step in _neighbors(grid, cur[0], cur[1], allow_diag):
            nxt = (nx, ny)
            if nxt in closed:
                continue
            ng = gcur + step
            if ng < gscore.get(nxt, float("inf")):
                gscore[nxt] = ng
                came[nxt] = cur
                heapq.heappush(openq, (ng + h(nxt, goal), ng, nxt))
    return None


def dijkstra(grid: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
             allow_diag: bool = True) -> Optional[List[Tuple[int, int]]]:
    """Dijkstra（无启发式）。用于与 A* 对照 —— 两者最优代价应一致。"""
    if not grid.inside(*start) or not grid.inside(*goal):
        return None
    if grid.is_occupied(*goal) or grid.is_occupied(*start):
        return None

    openq = [(0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    dist: Dict[Tuple[int, int], float] = {start: 0.0}
    closed = set()

    while openq:
        dcur, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        if cur in closed:
            continue
        closed.add(cur)

        for nx, ny, step in _neighbors(grid, cur[0], cur[1], allow_diag):
            nxt = (nx, ny)
            nd = dcur + step
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                came[nxt] = cur
                heapq.heappush(openq, (nd, nxt))
    return None


def path_cost(grid: GridMap, path: Sequence[Tuple[int, int]]) -> float:
    """按同样的边代价规则算路径代价（用于对照 A* 与 Dijkstra）。"""
    total = 0.0
    for a, b in zip(path, path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        step = math.hypot(dx, dy)
        p = grid.prob(b[0], b[1])
        if p is not None and p >= 0.5:
            step *= 1.5
        total += step
    return total


def path_length(path: Sequence[Tuple[int, int]], resolution: float = 1.0) -> float:
    """路径长度（世界单位）。"""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total * resolution


# =============================================================== 路径平滑
def _line_of_sight(grid: GridMap, a: Tuple[int, int], b: Tuple[int, int],
                   cost_occ: float = 0.6) -> bool:
    """两点之间是否直通（Bresenham 检查，任一路径格为障碍即不可达）。"""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else (-1 if x1 < x0 else 0)
    sy = 1 if y1 > y0 else (-1 if y1 < y0 else 0)
    err = dx - dy
    x, y = x0, y0
    while True:
        if (x, y) != a:
            p = grid.prob(x, y)
            if p is None or p >= cost_occ:
                return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def tighten_path(grid: GridMap, path: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    捷径裁剪（string pulling）：尽量用"直线可达"的两个点替换中间的一串点。

    现场意义：栅格 A* 出来的路径是锯齿状的，直接让机器人跟会左右摆；
    裁成几段长直线后，跟随明显平顺。
    """
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_of_sight(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def smooth_path(points: Sequence[Tuple[float, float]], iterations: int = 20,
                alpha: float = 0.5, fixed_ends: bool = True
                ) -> List[Tuple[float, float]]:
    """
    对**世界坐标**路点做拉普拉斯平滑（保持端点）。
    注意：平滑后必须再检查是否撞到障碍 —— 这一步调用方负责（见 plan_path）。
    """
    pts = [list(p) for p in points]
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    for _ in range(iterations):
        new = [list(p) for p in pts]
        for i in range(1, len(pts) - 1):
            if fixed_ends and i == 0:
                continue
            new[i][0] = pts[i][0] + alpha * ((pts[i - 1][0] + pts[i + 1][0]) / 2 - pts[i][0])
            new[i][1] = pts[i][1] + alpha * ((pts[i - 1][1] + pts[i + 1][1]) / 2 - pts[i][1])
        pts = new
    return [tuple(p) for p in pts]


def plan_path(grid: GridMap, start_world: Tuple[float, float],
              goal_world: Tuple[float, float], allow_diag: bool = True,
              tighten: bool = True) -> Optional[List[Tuple[float, float]]]:
    """
    一站式：世界坐标进 -> 世界坐标路径出（可通行、已裁剪）。
    """
    s = grid.world_to_grid(*start_world)
    g = grid.world_to_grid(*goal_world)
    gp = astar(grid, s, g, allow_diag)
    if gp is None:
        return None
    if tighten:
        gp = tighten_path(grid, gp)
    return [grid.grid_to_world(x, y) for x, y in gp]


# =============================================================== EKF 定位
def odom_motion_model(pose: Tuple[float, float, float],
                      v: float, w: float, dt: float) -> Tuple[float, float, float]:
    """
    差速轮里程计运动模型（**精确圆弧解**）。

    pose=(x,y,yaw)，v=线速度，w=角速度。

    为什么不用"中点积分"：中点积分只是近似。
    以 v=1, w=π/2, dt=1 为例：
        真实圆弧半径 r = v/w = 0.6366，转 90° 后的弦长 = 2r·sin(45°) = 0.9003
        中点积分给出 cos(π/4)·1 + sin(π/4)·1 -> 弦长 1.0000（差 10%）
    转弯越急、周期越长，误差越大。真实里程计用的是精确解，所以这里也用它。
    """
    x, y, th = pose
    if abs(w) < 1e-9:
        return (x + v * dt * math.cos(th), y + v * dt * math.sin(th), th)
    r = v / w
    th_new = th + w * dt
    return (x + r * (math.sin(th_new) - math.sin(th)),
            y - r * (math.cos(th_new) - math.cos(th)),
            _wrap(th_new))


def _wrap(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


class EKF2D:
    """
    二维位姿 EKF：融合**里程计**（x, y, yaw 观测）与 **IMU 航向**。

    状态 x = [x, y, yaw]
    现场为什么需要它：
      · 纯里程计：转弯打滑会累积航向误差，走一圈回来能偏十几度
      · 纯 IMU：陀螺零偏积分后航向会缓慢漂
      · 两者融合后，短时靠 IMU、长时靠里程计，明显更稳

    简化之处（README 里也写明）：这里只做"航向用 IMU 修正、
    位置用里程计"的紧耦合，没有做完整的 SLAM 后端优化。
    """

    def __init__(self, init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 q_xy: float = 0.02 ** 2, q_yaw: float = math.radians(1.0) ** 2,
                 r_odom_xy: float = 0.05 ** 2,
                 r_odom_yaw: float = math.radians(5.0) ** 2,
                 r_imu_yaw: float = math.radians(0.5) ** 2):
        self.x = [init_pose[0], init_pose[1], _wrap(init_pose[2])]
        # 协方差（3x3，行主序）
        self.P = [[0.05 ** 2, 0, 0], [0, 0.05 ** 2, 0], [0, 0, math.radians(3.0) ** 2]]
        self.q_xy = q_xy
        self.q_yaw = q_yaw
        self.r_odom_xy = r_odom_xy
        self.r_odom_yaw = r_odom_yaw
        self.r_imu_yaw = r_imu_yaw
        self.updates = 0

    # ---------------------------------------------------------- 预测
    def predict(self, v: float, w: float, dt: float) -> None:
        """用运动模型推进状态与协方差（这里用简化雅可比）。"""
        x, y, th = self.x
        th_mid = th + w * dt / 2.0
        self.x = [x + v * dt * math.cos(th_mid),
                  y + v * dt * math.sin(th_mid),
                  _wrap(th + w * dt)]

        # F = ∂f/∂x
        F = [[1.0, 0.0, -v * dt * math.sin(th_mid)],
             [0.0, 1.0, v * dt * math.cos(th_mid)],
             [0.0, 0.0, 1.0]]
        self.P = self._FPFt_plus_Q(F, dt)

    def _FPFt_plus_Q(self, F, dt):
        P = self.P
        # FP
        FP = [[sum(F[i][k] * P[k][j] for k in range(3)) for j in range(3)]
              for i in range(3)]
        # (FP)F^T
        FPFT = [[sum(FP[i][k] * F[j][k] for k in range(3)) for j in range(3)]
                for i in range(3)]
        Q = [[self.q_xy * dt, 0, 0],
             [0, self.q_xy * dt, 0],
             [0, 0, self.q_yaw * dt]]
        self.P = [[FPFT[i][j] + Q[i][j] for j in range(3)] for i in range(3)]
        return self.P

    # ---------------------------------------------------------- 更新
    def update(self, z: Sequence[float], R: Sequence[Sequence[float]]) -> None:
        """
        标准 EKF 更新。z 与状态同维（这里都是 3 维），H = I。
        传入的 R 是 3x3 观测噪声协方差。
        """
        H = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        # S = HPH^T + R = P + R
        S = [[self.P[i][j] + R[i][j] for j in range(3)] for i in range(3)]
        Si = self._inv3(S)
        if Si is None:
            return                                  # 奇异：跳过这次更新
        # K = P H^T S^-1 = P S^-1
        K = [[sum(self.P[i][k] * Si[k][j] for k in range(3)) for j in range(3)]
             for i in range(3)]
        # 残差（角度要归一化，否则 ±π 附近会"绕远路"修正）
        y = [_wrap(z[i] - self.x[i]) for i in range(3)]
        self.x = [_wrap(self.x[i] + sum(K[i][j] * y[j] for j in range(3)))
                  if i == 2 else self.x[i] + sum(K[i][j] * y[j] for j in range(3))
                  for i in range(3)]
        # P = (I - K H) P
        IKH = [[(1.0 if i == j else 0.0) - K[i][j] for j in range(3)] for i in range(3)]
        self.P = [[sum(IKH[i][k] * self.P[k][j] for k in range(3)) for j in range(3)]
                  for i in range(3)]
        self.updates += 1

    def update_odom(self, z_xy_yaw: Sequence[float]) -> None:
        R = [[self.r_odom_xy, 0, 0],
             [0, self.r_odom_xy, 0],
             [0, 0, self.r_odom_yaw]]
        self.update(z_xy_yaw, R)

    def update_imu_yaw(self, yaw: float) -> None:
        """只用 IMU 的航向做一次更新（位置观测噪声给很大，等于不修正）。"""
        big = 1e6
        R = [[big, 0, 0], [0, big, 0], [0, 0, self.r_imu_yaw]]
        self.update([self.x[0], self.x[1], yaw], R)

    # ---------------------------------------------------------- 工具
    @staticmethod
    def _inv3(m):
        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        A = e * i - f * h
        B = -(d * i - f * g)
        C = d * h - e * g
        det = a * A + b * B + c * C
        if abs(det) < 1e-18:
            return None
        inv = [[A, -(b * i - c * h), (b * f - c * e)],
               [B, (a * i - c * g), -(a * f - c * d)],
               [C, -(a * h - b * g), (a * e - b * d)]]
        return [[inv[r][cc] / det for cc in range(3)] for r in range(3)]

    def pose(self) -> Tuple[float, float, float]:
        return (self.x[0], self.x[1], _wrap(self.x[2]))

    def yaw_sigma_deg(self) -> float:
        return math.degrees(math.sqrt(max(0.0, self.P[2][2])))

    def pos_sigma(self) -> float:
        return math.sqrt(max(0.0, self.P[0][0] + self.P[1][1]))
