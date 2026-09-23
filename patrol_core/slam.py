# -*- coding: utf-8 -*-
"""
patrol_core.slam — 2D 占据栅格建图（log-odds 概率更新）

对应岗位要求的「SLAM 建图」与职责③「地图创建」。

为什么用 **log-odds** 而不是直接存概率：
    · 概率相乘会很快下溢（0.001^100 直接变 0），取对数后变成**相加**
    · 更新公式从 p = p_occ*p_prev / (...) 变成简单的 l += l_occ
    · 概率与 log-odds 的换算只在**读地图时**做一次

实现要点（都是现场容易出问题的地方）：
    · 射线投射用 **Bresenham**，命中点加占据证据、沿途格子加空闲证据
    · 每个格子的 log-odds 要**夹在上下限内**（否则长期不动的地方会被"刷白/刷黑"）
    · 命中点若落在机器人自身附近要**跳过**（典型是打到自己的脚/机身）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# 概率 <-> log-odds
def prob_to_logodds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def logodds_to_prob(l: float) -> float:
    return 1.0 - 1.0 / (1.0 + math.exp(l))


@dataclass
class GridMap:
    """占据栅格地图。分辨率单位：米/格。"""

    width: int
    height: int
    resolution: float = 0.05
    origin_x: float = 0.0          # 栅格 (0,0) 对应的世界坐标
    origin_y: float = 0.0
    log_odds: List[float] = field(default_factory=list)
    visits: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("地图尺寸必须为正")
        if self.resolution <= 0:
            raise ValueError("分辨率必须为正")
        if not self.log_odds:
            self.log_odds = [0.0] * (self.width * self.height)
            self.visits = [0] * (self.width * self.height)

    # ---------------------------------------------------------- 坐标换算
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor((x - self.origin_x) / self.resolution)),
                int(math.floor((y - self.origin_y) / self.resolution)))

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return (self.origin_x + (gx + 0.5) * self.resolution,
                self.origin_y + (gy + 0.5) * self.resolution)

    def inside(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def _idx(self, gx: int, gy: int) -> int:
        return gy * self.width + gx

    # ---------------------------------------------------------- 读写
    def get_logodds(self, gx: int, gy: int) -> Optional[float]:
        if not self.inside(gx, gy):
            return None
        return self.log_odds[self._idx(gx, gy)]

    def prob(self, gx: int, gy: int) -> Optional[float]:
        l = self.get_logodds(gx, gy)
        return None if l is None else logodds_to_prob(l)

    def is_occupied(self, gx: int, gy: int, thresh: float = 0.6) -> bool:
        p = self.prob(gx, gy)
        return bool(p is not None and p >= thresh)

    def is_free(self, gx: int, gy: int, thresh: float = 0.4) -> bool:
        p = self.prob(gx, gy)
        return bool(p is not None and p <= thresh)

    def is_unknown(self, gx: int, gy: int) -> bool:
        return self.visits[self._idx(gx, gy)] == 0 if self.inside(gx, gy) else False

    # ---------------------------------------------------------- 更新
    def update_cell(self, gx: int, gy: int, delta: float,
                    l_min: float = -4.0, l_max: float = 4.0) -> bool:
        """给一个格子累加 log-odds 证据（带上下限夹取）。"""
        if not self.inside(gx, gy):
            return False
        i = self._idx(gx, gy)
        v = self.log_odds[i] + delta
        # 夹取：不夹的话，长期不动的墙会被反复"刷黑"到无法修正
        self.log_odds[i] = min(max(v, l_min), l_max)
        self.visits[i] += 1
        return True

    def occupied_ratio(self, thresh: float = 0.6) -> float:
        seen = sum(1 for v in self.visits if v > 0)
        if seen == 0:
            return 0.0
        occ = sum(1 for i, v in enumerate(self.visits)
                  if v > 0 and logodds_to_prob(self.log_odds[i]) >= thresh)
        return occ / seen

    def known_ratio(self) -> float:
        if not self.visits:
            return 0.0
        return sum(1 for v in self.visits if v > 0) / len(self.visits)

    def to_ascii(self, occ_thresh: float = 0.6, free_thresh: float = 0.4) -> str:
        """画成 ASCII，便于在日志里直接看（现场排查很有用）。"""
        rows = []
        for gy in range(self.height - 1, -1, -1):     # y 轴向上
            line = []
            for gx in range(self.width):
                if self.is_unknown(gx, gy):
                    line.append(' ')
                elif self.is_occupied(gx, gy, occ_thresh):
                    line.append('#')
                elif self.is_free(gx, gy, free_thresh):
                    line.append('.')
                else:
                    line.append('?')
            rows.append(''.join(line))
        return '\n'.join(rows)


class OccupancyMapper:
    """
    把激光扫描 + 位姿累积成占据栅格。

    数据流：一帧扫描（角度 + 距离）+ 该时刻位姿 -> 射线投射 -> 更新栅格
    """

    def __init__(self, grid: GridMap,
                 p_occ: float = 0.70, p_free: float = 0.30,
                 l_min: float = -4.0, l_max: float = 4.0,
                 min_range: float = 0.15, max_range: float = 12.0,
                 self_filter_r: float = 0.20):
        self.grid = grid
        self.l_occ = prob_to_logodds(p_occ)
        self.l_free = prob_to_logodds(p_free)
        self.l_min = l_min
        self.l_max = l_max
        self.min_range = min_range
        self.max_range = max_range
        self.self_filter_r = self_filter_r      # 小于这个距离的命中点视为打到自身
        self.scans = 0
        self.hits = 0
        self.misses = 0

    def add_scan(self, pose: Tuple[float, float, float],
                 ranges: Sequence[float],
                 angles: Optional[Sequence[float]] = None,
                 angle_min: float = -math.pi, angle_step: float = None) -> int:
        """
        并入一帧扫描。pose = (x, y, yaw)。
        ranges 里 <=0 或 >max_range 的点视为"没打到"（只清空沿途，不加占据）。
        返回本帧更新的格子数。
        """
        if angle_step is None:
            if angles is not None and len(angles) > 1:
                angle_step = angles[1] - angles[0]
            else:
                angle_step = 2 * math.pi / max(1, len(ranges))
        px, py, pyaw = pose
        gx0, gy0 = self.grid.world_to_grid(px, py)
        updated = 0

        for k, r in enumerate(ranges):
            a = (angles[k] if angles is not None else angle_min + k * angle_step) + pyaw
            invalid = (r is None) or (r <= 0.0) or (r > self.max_range)
            rr = self.max_range if invalid else r
            ex = px + rr * math.cos(a)
            ey = py + rr * math.sin(a)
            gx1, gy1 = self.grid.world_to_grid(ex, ey)

            # 命中点太近 -> 打到自身，跳过该束
            if (not invalid) and r < self.self_filter_r:
                continue

            self._ray(gx0, gy0, gx1, gy1, mark_end=(not invalid))
            updated += 1

        self.scans += 1
        return updated

    def _ray(self, x0: int, y0: int, x1: int, y1: int, mark_end: bool) -> None:
        """Bresenham 射线：沿途标空闲，终点（若命中）标占据。"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else (-1 if x1 < x0 else 0)
        sy = 1 if y1 > y0 else (-1 if y1 < y0 else 0)
        err = dx - dy
        x, y = x0, y0
        while True:
            at_end = (x == x1 and y == y1)
            if at_end:
                if mark_end:
                    if self.grid.update_cell(x, y, self.l_occ, self.l_min, self.l_max):
                        self.hits += 1
                else:
                    if self.grid.update_cell(x, y, self.l_free, self.l_min, self.l_max):
                        self.misses += 1
                return
            if self.grid.update_cell(x, y, self.l_free, self.l_min, self.l_max):
                self.misses += 1
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            # 走出地图就停（不做无限循环）
            if not self.grid.inside(x, y):
                return


def make_room_map(width=200, height=200, resolution=0.05,
                  walls: Optional[List[Tuple[float, float, float, float]]] = None,
                  wall_thickness: Optional[float] = None) -> GridMap:
    """
    造一张"真值"地图（用于测试与演示）。

    注意：这是**给仿真用的真值**，与建图结果要分开 ——
    把真值和建出来的图混在一起是评审里最常见的自欺。
    """
    g = GridMap(width=width, height=height, resolution=resolution)
    # 默认墙厚 = 1 格。做成 3 格厚会让"建图召回率"这个指标失真 ——
    # 激光只打到最外一层，里面两层永远收不到占据证据（这是测试里踩到的）。
    if wall_thickness is None:
        wall_thickness = resolution
    if walls is None:
        w = width * resolution
        h = height * resolution
        walls = [(0.5, 0.5, w - 0.5, 0.5), (w - 0.5, 0.5, w - 0.5, h - 0.5),
                 (w - 0.5, h - 0.5, 0.5, h - 0.5), (0.5, h - 0.5, 0.5, 0.5),
                 (w * 0.5, h * 0.3, w * 0.5, h * 0.7)]
    # half = 向两侧各扩几格。这里**不能**用 max(1, ...) 兜底 ——
    # 那会把「1 格厚的薄墙」强行变成 3 格厚，于是激光只打得到最外一层，
    # "建图召回率"这个指标就失真了（写测试时踩到的）。
    half = int(round(wall_thickness / (2 * resolution)))
    if half < 0:
        half = 0
    for (x1, y1, x2, y2) in walls:
        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(length / (resolution / 2)))
        for i in range(steps + 1):
            t = i / steps
            wx = x1 + (x2 - x1) * t
            wy = y1 + (y2 - y1) * t
            gx, gy = g.world_to_grid(wx, wy)
            for ddx in range(-half, half + 1):
                for ddy in range(-half, half + 1):
                    g.update_cell(gx + ddx, gy + ddy, 50.0, -50.0, 50.0)
    return g


def raycast_ground_truth(truth: GridMap, pose: Tuple[float, float, float],
                         n_beams: int = 180, angle_min: float = -math.pi,
                         angle_max: float = math.pi,
                         max_range: float = 12.0,
                         step: float = None) -> List[float]:
    """
    在真值地图上做射线投射，得到一帧"仿真激光"。

    这样测建图时**不需要真的连雷达**，而且因为真值已知，
    可以定量评估建图质量（而不是"看着像"）。
    """
    if step is None:
        step = truth.resolution / 2.0
    px, py, pyaw = pose
    out: List[float] = []
    for k in range(n_beams):
        a = pyaw + angle_min + (angle_max - angle_min) * (k / max(1, n_beams - 1))
        ca, sa = math.cos(a), math.sin(a)
        r = 0.0
        hit = False
        while r < max_range:
            r += step
            gx, gy = truth.world_to_grid(px + r * ca, py + r * sa)
            if not truth.inside(gx, gy):
                break
            if truth.is_occupied(gx, gy, 0.9):
                hit = True
                break
        out.append(r if hit else max_range)
    return out
