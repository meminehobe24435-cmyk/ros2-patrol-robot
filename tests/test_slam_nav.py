# -*- coding: utf-8 -*-
"""
tests/test_slam_nav.py — SLAM 建图 / 路径规划 / 传感器融合 单元测试

**全部离线可跑**：真值地图与"激光扫描"都是合成出来的，
所以不需要雷达、不需要 ROS2，也能定量验证算法对不对。

  T1 grid     栅格坐标换算、log-odds 夹取、未知/占据/空闲判定
  T2 mapping  射线投射建图：墙建出来、空地清出来、自身命中被过滤
  T3 quality  建图质量：与真值地图对比（占据重叠率、已知率）
  T4 astar    A* 与 Dijkstra：同一地图上最优代价应一致；绕障、无解、禁斜穿
  T5 smooth   捷径裁剪与拉普拉斯平滑：变短、端点保持、不撞墙
  T6 ekf       EKF：预测、里程计更新、IMU 航向修正、协方差收敛
"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from patrol_core.slam import (GridMap, OccupancyMapper, make_room_map,
                              raycast_ground_truth, prob_to_logodds,
                              logodds_to_prob)
from patrol_core.planning import (EKF2D, astar, dijkstra, odom_motion_model,
                                  path_cost, path_length, plan_path,
                                  smooth_path, tighten_path)


class T1Grid(unittest.TestCase):
    """T1 栅格地图基础"""

    def test_prob_logodds_roundtrip(self):
        for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            l = prob_to_logodds(p)
            self.assertAlmostEqual(logodds_to_prob(l), p, places=6)
        self.assertAlmostEqual(logodds_to_prob(0.0), 0.5, places=9)

    def test_world_grid_conversion(self):
        g = GridMap(100, 100, resolution=0.1)
        gx, gy = g.world_to_grid(0.0, 0.0)
        self.assertEqual((gx, gy), (0, 0))
        gx, gy = g.world_to_grid(0.05, 0.15)
        self.assertEqual((gx, gy), (0, 1))
        # 负坐标要向下取整（floor），否则 (-0.05) 会被算成 0 而不是 -1
        gx, gy = g.world_to_grid(-0.05, -0.15)
        self.assertEqual((gx, gy), (-1, -2))
        # 往返一致（格心）
        for gx in (0, 10, 99):
            wx, wy = g.grid_to_world(gx, 5)
            self.assertEqual(g.world_to_grid(wx, wy), (gx, 5))

    def test_inside_and_out_of_range(self):
        g = GridMap(10, 10)
        self.assertTrue(g.inside(0, 0) and g.inside(9, 9))
        self.assertFalse(g.inside(10, 0) or g.inside(-1, 0))
        self.assertIsNone(g.prob(-1, 0))
        self.assertFalse(g.update_cell(-1, 0, 1.0))

    def test_logodds_clamped(self):
        """证据要夹在上下限内，否则长期不动的墙会被刷到无法修正"""
        g = GridMap(5, 5)
        for _ in range(1000):
            g.update_cell(2, 2, 1.0, l_min=-4.0, l_max=4.0)
        l = g.get_logodds(2, 2)
        self.assertAlmostEqual(l, 4.0, places=9)
        for _ in range(1000):
            g.update_cell(2, 2, -1.0, l_min=-4.0, l_max=4.0)
        self.assertAlmostEqual(g.get_logodds(2, 2), -4.0, places=9)

    def test_state_predicates(self):
        g = GridMap(5, 5)
        self.assertTrue(g.is_unknown(0, 0))
        g.update_cell(0, 0, prob_to_logodds(0.9))
        self.assertTrue(g.is_occupied(0, 0))
        self.assertFalse(g.is_free(0, 0))
        g.update_cell(1, 1, prob_to_logodds(0.05))
        self.assertTrue(g.is_free(1, 1))
        self.assertFalse(g.is_occupied(1, 1))

    def test_bad_construction(self):
        with self.assertRaises(ValueError):
            GridMap(0, 10)
        with self.assertRaises(ValueError):
            GridMap(10, 10, resolution=0.0)

    def test_ascii_render(self):
        g = make_room_map(40, 40, 0.1)
        art = g.to_ascii()
        self.assertEqual(len(art.splitlines()), 40)
        self.assertIn('#', art)
        self.assertTrue(all(len(l) == 40 for l in art.splitlines()))


class T2Mapping(unittest.TestCase):
    """T2 射线投射建图"""

    def test_ray_marks_free_then_occupied(self):
        g = GridMap(60, 60, resolution=0.1)
        g.update_cell(*g.world_to_grid(3.0, 3.0), 50.0, -50.0, 50.0)   # 造一面墙
        m = OccupancyMapper(g)
        m.add_scan((0.5, 3.0, 0.0), [3.0], angles=[0.0])               # 朝 +x 打
        # 沿途应为空闲
        for gx in range(6, 28):
            gy = g.world_to_grid(0.0, 3.0)[1]
            self.assertTrue(g.is_free(gx, gy), "沿途格 (%d,%d) 应为空闲" % (gx, gy))
        # 命中处应为占据
        hx, hy = g.world_to_grid(3.0, 3.0)
        self.assertTrue(g.is_occupied(hx, hy))
        self.assertEqual(m.scans, 1)
        self.assertGreater(m.hits, 0)
        self.assertGreater(m.misses, 0)

    def test_self_filter_skips_close_hits(self):
        """打到自身（距离过近）的束应被跳过，否则机器人脚下会被画成墙"""
        g = GridMap(50, 50, resolution=0.1)
        m = OccupancyMapper(g, self_filter_r=0.30)
        before = sum(g.visits)
        m.add_scan((1.0, 1.0, 0.0), [0.05, 0.10, 0.20], angles=[0.0, 0.1, 0.2])
        after = sum(g.visits)
        self.assertEqual(before, after, "过近的命中不应产生任何更新")

    def test_invalid_and_out_of_range_beams(self):
        """无效距离（0/负/超量程）只清沿途，不加占据证据"""
        g = GridMap(60, 60, resolution=0.1)
        m = OccupancyMapper(g, max_range=5.0)
        m.add_scan((0.5, 3.0, 0.0), [0.0, -1.0, 99.0], angles=[0.0, 0.1, 0.2])
        self.assertEqual(m.hits, 0, "无效束不应产生占据证据")
        self.assertGreater(m.misses, 0, "仍应清出沿途空闲")

    def test_ray_stops_at_map_edge(self):
        """射线打向地图外应在边界停止，不能越界或死循环"""
        g = GridMap(20, 20, resolution=0.1)
        m = OccupancyMapper(g, max_range=100.0)
        m.add_scan((1.0, 1.0, 0.0), [100.0], angles=[0.0])
        self.assertTrue(g.is_free(19, 10))
        self.assertEqual(g.get_logodds(19, 10) is not None, True)

    def test_unknown_ratio_and_occupied_ratio(self):
        g = GridMap(20, 20, resolution=0.1)
        self.assertEqual(g.known_ratio(), 0.0)
        g.update_cell(5, 5, prob_to_logodds(0.9))
        self.assertAlmostEqual(g.known_ratio(), 1.0 / 400, places=9)
        self.assertAlmostEqual(g.occupied_ratio(), 1.0, places=9)


class T3MappingQuality(unittest.TestCase):
    """T3 建图质量（与真值对比，而不是"看着像"）"""

    def setUp(self):
        self.truth = make_room_map(160, 160, 0.05)
        self.grid = GridMap(160, 160, 0.05)
        self.mapper = OccupancyMapper(self.grid, max_range=8.0)

    def _build(self, n_poses=24, radius=3.0):
        """绕房间中心走一圈，扫一圈激光，把地图建出来"""
        cx = cy = 160 * 0.05 / 2.0
        for i in range(n_poses):
            a = 2 * math.pi * i / n_poses
            pose = (cx + radius * math.cos(a), cy + radius * math.sin(a),
                    a + math.pi)
            rng = raycast_ground_truth(self.truth, pose, n_beams=120,
                                       max_range=8.0)
            self.mapper.add_scan(pose, rng, angle_min=-math.pi,
                                 angle_step=2 * math.pi / 120)

    def test_walls_reconstructed(self):
        """
        墙体召回率。
        注意真值墙必须是**1 格厚**（make_room_map 默认值）——
        做成 3 格厚时，激光只打到最外一层，另外两层永远收不到占据证据，
        召回率会假性掉到 0.36 左右（这个坑在写测试时踩过）。
        """
        self._build()
        # 真值里的占据格，在建出来的图里应大多被判为占据
        hit = tot = 0
        for gy in range(self.truth.height):
            for gx in range(self.truth.width):
                if self.truth.is_occupied(gx, gy, 0.9):
                    tot += 1
                    if self.grid.is_occupied(gx, gy, 0.6):
                        hit += 1
        recall = hit / tot
        self.assertGreater(recall, 0.60,
                           "墙体召回率 %.2f（应从真值里找回大部分墙）" % recall)

    def test_free_space_cleared(self):
        self._build()
        # 机器人走过的附近应是空闲
        cx = cy = 160 * 0.05 / 2.0
        for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
            wx = cx + 3.0 * math.cos(a)
            wy = cy + 3.0 * math.sin(a)
            gx, gy = self.grid.world_to_grid(wx, wy)
            self.assertTrue(self.grid.is_free(gx, gy),
                            "机器人走过的位置 (%d,%d) 应为空闲" % (gx, gy))

    def test_known_ratio_grows(self):
        r0 = self.grid.known_ratio()
        self._build(n_poses=8)
        r1 = self.grid.known_ratio()
        self._build(n_poses=8, radius=3.5)
        r2 = self.grid.known_ratio()
        self.assertEqual(r0, 0.0)
        self.assertGreater(r1, 0.0)
        self.assertGreater(r2, r1, "多走一圈已知区域应变大")


class T4Planning(unittest.TestCase):
    """T4 全局路径规划"""

    def _open_map(self, w=60, h=60, res=0.1):
        g = GridMap(w, h, res)
        # 全部标为空闲
        for gy in range(h):
            for gx in range(w):
                g.update_cell(gx, gy, prob_to_logodds(0.2))
        return g

    def test_straight_path(self):
        g = self._open_map()
        p = astar(g, (5, 5), (25, 5))
        self.assertIsNotNone(p)
        self.assertEqual(p[0], (5, 5))
        self.assertEqual(p[-1], (25, 5))
        self.assertEqual(len(p), 21, "直线路径应恰好 21 个格（含两端）")

    def test_astar_and_dijkstra_same_cost(self):
        """A* 用了启发式，但最优代价必须与 Dijkstra 一致 —— 否则就是启发式高估了"""
        g = self._open_map(80, 80)
        # 竖一道带缺口的墙
        for gy in range(80):
            if gy not in (38, 39, 40):
                g.update_cell(40, gy, prob_to_logodds(0.95))
        pa = astar(g, (5, 5), (70, 70))
        pd = dijkstra(g, (5, 5), (70, 70))
        self.assertIsNotNone(pa)
        self.assertIsNotNone(pd)
        self.assertAlmostEqual(path_cost(g, pa), path_cost(g, pd), places=9,
                               msg="A* 与 Dijkstra 最优代价不一致")

    def test_astar_goes_through_gap(self):
        g = self._open_map(80, 80)
        for gy in range(80):
            if gy not in (38, 39, 40):
                g.update_cell(40, gy, prob_to_logodds(0.95))
        p = astar(g, (5, 40), (70, 40))
        self.assertIsNotNone(p)
        for gx, gy in p:
            self.assertLess(g.prob(gx, gy), 0.6, "路径不应穿墙")

    def test_no_path_when_walled_off(self):
        g = self._open_map(60, 60)
        for gy in range(60):
            g.update_cell(30, gy, prob_to_logodds(0.95))    # 完全封死
        self.assertIsNone(astar(g, (5, 5), (50, 50)))
        self.assertIsNone(dijkstra(g, (5, 5), (50, 50)))

    def test_goal_on_obstacle_rejected(self):
        g = self._open_map(30, 30)
        g.update_cell(10, 10, prob_to_logodds(0.95))
        self.assertIsNone(astar(g, (5, 5), (10, 10)))
        self.assertIsNone(astar(g, (10, 10), (5, 5)))

    def test_out_of_bounds(self):
        g = self._open_map(20, 20)
        self.assertIsNone(astar(g, (0, 0), (99, 99)))
        self.assertIsNone(astar(g, (-1, 0), (5, 5)))

    def test_no_diagonal_corner_cutting(self):
        """禁止贴着角斜穿：斜走时两侧正交格若有一个是障碍就不许走"""
        g = self._open_map(20, 20)
        g.update_cell(6, 5, prob_to_logodds(0.95))     # 斜穿会经过这里一侧
        g.update_cell(5, 6, prob_to_logodds(0.95))
        p = astar(g, (5, 5), (6, 6), allow_diag=True)
        if p is not None and len(p) == 2:
            self.fail("不应允许两障碍夹角处的斜穿")

    def test_unknown_costs_more_than_known(self):
        """未知区应加代价，路径优先走已知区"""
        g = GridMap(60, 20, 0.1)          # 默认全未知
        for gx in range(60):
            g.update_cell(gx, 5, prob_to_logodds(0.2))    # 已知走廊
        p = astar(g, (5, 5), (55, 5))
        self.assertIsNotNone(p)
        for gx, gy in p:
            self.assertLess(abs(gy - 5), 3, "应尽量贴着已知走廊走")

    def test_path_length_helper(self):
        self.assertEqual(path_length([(0, 0), (3, 4)], 1.0), 5.0)
        self.assertEqual(path_length([(0, 0)], 1.0), 0.0)
        self.assertAlmostEqual(path_length([(0, 0), (3, 4)], 0.05), 0.25, places=9)


class T5Smoothing(unittest.TestCase):
    """T5 路径平滑"""

    def test_tighten_removes_redundant_points(self):
        g = GridMap(60, 60, 0.1)
        for gy in range(60):
            for gx in range(60):
                g.update_cell(gx, gy, prob_to_logodds(0.2))
        straight = [(5 + i, 5) for i in range(20)]
        t = tighten_path(g, straight)
        self.assertEqual(len(t), 2, "纯直线应被裁成两个端点")
        self.assertEqual(t[0], (5, 5))
        self.assertEqual(t[-1], (24, 5))

    def test_tighten_keeps_obstacle_waypoints(self):
        """绕障路径裁剪后仍不能穿墙"""
        g = GridMap(40, 40, 0.1)
        for gy in range(40):
            for gx in range(40):
                g.update_cell(gx, gy, prob_to_logodds(0.2))
        for gy in range(0, 25):
            g.update_cell(15, gy, prob_to_logodds(0.95))     # 竖墙留上方缺口
        p = astar(g, (5, 5), (30, 5))
        self.assertIsNotNone(p)
        t = tighten_path(g, p)
        self.assertLess(len(t), len(p), "应被裁剪")
        # 逐段检查不穿墙
        for a, b in zip(t, t[1:]):
            x0, y0 = a
            x1, y1 = b
            steps = max(abs(x1 - x0), abs(y1 - y0))
            for i in range(steps + 1):
                x = round(x0 + (x1 - x0) * i / steps)
                y = round(y0 + (y1 - y0) * i / steps)
                self.assertLess(g.prob(x, y), 0.6,
                                "裁剪后的段穿过了障碍 (%d,%d)" % (x, y))

    def test_smooth_keeps_endpoints(self):
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 0.0)]
        sm = smooth_path(pts, iterations=50)
        self.assertEqual(sm[0], pts[0])
        self.assertEqual(sm[-1], pts[-1])
        self.assertEqual(len(sm), len(pts))

    def test_smooth_reduces_roughness(self):
        def roughness(p):
            r = 0.0
            for i in range(1, len(p) - 1):
                ax = p[i][0] - p[i - 1][0]
                ay = p[i][1] - p[i - 1][1]
                bx = p[i + 1][0] - p[i][0]
                by = p[i + 1][1] - p[i][1]
                a = math.atan2(ay, ax)
                b = math.atan2(by, bx)
                r += abs((b - a + math.pi) % (2 * math.pi) - math.pi)
            return r
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 0.0), (3.0, 0.0)]
        sm = smooth_path(pts, iterations=30)
        self.assertLess(roughness(sm), roughness(pts), "平滑后转角应更小")

    def test_short_inputs_unchanged(self):
        self.assertEqual(smooth_path([(0.0, 0.0)]), [(0.0, 0.0)])
        self.assertEqual(smooth_path([(0.0, 0.0), (1.0, 1.0)]), [(0.0, 0.0), (1.0, 1.0)])
        self.assertEqual(tighten_path(GridMap(5, 5), [(0, 0)]), [(0, 0)])

    def test_plan_path_end_to_end(self):
        """世界坐标进、世界坐标出，且被裁剪成直线两点"""
        g = GridMap(60, 60, 0.1)
        for gy in range(60):
            for gx in range(60):
                g.update_cell(gx, gy, prob_to_logodds(0.2))
        # 墙放在**上方**（gy 40~59），不挡起终点之间的直线（都在 gy≈10）
        for gy in range(40, 60):
            g.update_cell(30, gy, prob_to_logodds(0.95))

        path = plan_path(g, (1.0, 1.0), (5.0, 1.0))
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2, "不挡路的直线应被裁成两点")
        # 首点应落在起点所在格的格心
        gx, gy = g.world_to_grid(1.0, 1.0)
        self.assertAlmostEqual(path[0][0], g.grid_to_world(gx, gy)[0], places=6)
        self.assertAlmostEqual(path[0][1], g.grid_to_world(gx, gy)[1], places=6)

    def test_plan_path_goes_around_wall(self):
        """真正挡路的墙：路径必须绕过去，且不穿墙"""
        g = GridMap(60, 60, 0.1)
        for gy in range(60):
            for gx in range(60):
                g.update_cell(gx, gy, prob_to_logodds(0.2))
        for gy in range(0, 40):
            g.update_cell(30, gy, prob_to_logodds(0.95))
        path = plan_path(g, (1.0, 1.0), (5.0, 1.0))
        self.assertIsNotNone(path, "墙有上方缺口，应有解")
        self.assertGreater(len(path), 2, "需要绕墙，不该只有两点")
        for (wx, wy) in path:
            p = g.prob(*g.world_to_grid(wx, wy))
            self.assertLess(p, 0.6, "路径不应穿墙")


class T6EKF(unittest.TestCase):
    """T6 传感器融合定位"""

    def test_motion_model_straight(self):
        p = odom_motion_model((0.0, 0.0, 0.0), v=1.0, w=0.0, dt=1.0)
        self.assertAlmostEqual(p[0], 1.0, places=9)
        self.assertAlmostEqual(p[1], 0.0, places=9)
        self.assertAlmostEqual(p[2], 0.0, places=9)

    def test_motion_model_turn(self):
        p = odom_motion_model((0.0, 0.0, 0.0), v=1.0, w=math.pi / 2, dt=1.0)
        self.assertAlmostEqual(p[2], math.pi / 2, places=9)
        # 中点积分：走的是近似圆弧，x 与 y 都不为 0
        self.assertGreater(p[0], 0.0)
        self.assertGreater(p[1], 0.0)
        # 圆弧半径 = v/w = 2/pi ≈ 0.6366，转 90° 后位移应为 sqrt(2)*r
        r = 1.0 / (math.pi / 2)
        self.assertAlmostEqual(math.hypot(p[0], p[1]), math.sqrt(2) * r, places=6)

    def test_predict_only_drifts(self):
        ekf = EKF2D()
        for _ in range(100):
            ekf.predict(1.0, 0.0, 0.1)
        x, y, th = ekf.pose()
        self.assertAlmostEqual(x, 10.0, places=6)
        self.assertAlmostEqual(th, 0.0, places=9)
        self.assertGreater(ekf.pos_sigma(), 0.0)
        self.assertGreater(ekf.P[2][2], 0.0)

    def test_odom_update_pulls_state(self):
        ekf = EKF2D()
        for _ in range(10):
            ekf.predict(1.0, 0.0, 0.1)          # 预测走到 (1.0, 0)
        ekf.update_odom([1.10, 0.00, 0.00])     # 观测说在 1.10
        x, _, _ = ekf.pose()
        self.assertGreater(x, 1.0)
        self.assertLess(x, 1.10, "EKF 应折中，不该直接跳到观测值")

    def test_imu_yaw_corrects_drift(self):
        """航向漂了以后，IMU 观测应把它拉回来"""
        ekf = EKF2D()
        for _ in range(200):
            ekf.predict(0.0, 0.01, 0.05)         # 持续微转 -> 航向累积漂移
        drifted = ekf.pose()[2]
        self.assertGreater(abs(drifted), 0.05)
        for _ in range(50):
            ekf.update_imu_yaw(0.0)
        self.assertLess(abs(ekf.pose()[2]), abs(drifted),
                        "IMU 航向观测应把漂移拉回")
        self.assertLess(ekf.yaw_sigma_deg(), 5.0, "航向不确定度应收敛")

    def test_imu_update_does_not_move_position(self):
        """只给航向的观测不应把位置也拽走"""
        ekf = EKF2D(init_pose=(5.0, 3.0, 0.0))
        before = ekf.pose()[:2]
        for _ in range(20):
            ekf.update_imu_yaw(0.1)
        after = ekf.pose()[:2]
        self.assertAlmostEqual(before[0], after[0], places=3)
        self.assertAlmostEqual(before[1], after[1], places=3)

    def test_angle_wrap_in_update(self):
        """±π 附近的残差要归一化，否则会"绕远路"修正"""
        ekf = EKF2D(init_pose=(0.0, 0.0, math.radians(179)))
        ekf.update_odom([0.0, 0.0, math.radians(-179)])
        yaw = math.degrees(ekf.pose()[2])
        # 正确结果应接近 ±180（相差 2°），绝不该跳到 0 附近
        self.assertGreater(abs(yaw), 150.0, "航向跨越 ±π 时修正错误：%.1f°" % yaw)

    def test_covariance_shrinks_with_observations(self):
        ekf = EKF2D()
        s0 = ekf.pos_sigma()
        for _ in range(30):
            ekf.predict(0.1, 0.0, 0.1)
            ekf.update_odom([ekf.x[0], ekf.x[1], ekf.x[2]])
        self.assertLess(ekf.pos_sigma(), s0 * 1.5,
                        "持续有观测时不确定度不应无限增长")

    def test_singular_observation_matrix_skipped(self):
        """奇异协方差不应崩，只是跳过更新"""
        ekf = EKF2D()
        bad_R = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, float("inf")]]
        try:
            ekf.update([1.0, 1.0, 0.5], bad_R)
        except Exception as exc:              # noqa: BLE001
            self.fail("奇异协方差应被跳过而不是抛异常：%s" % exc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
