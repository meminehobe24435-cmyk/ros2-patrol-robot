# -*- coding: utf-8 -*-
"""
tests/test_route_config.py — 路线配置加载测试（不需要 ROS2）

这一层很关键：patrol_ros.route_from_dict 是**ROS2 节点层里唯一不依赖 rclpy 的部分**，
所以它可以（也应该）被单测覆盖。现场的常见故障就是"路线 JSON 写错了"，
让它在加载阶段就报出来，比上机跑到一半才发现要好得多。
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from patrol_core import (ActionKind, PatrolMission, SimNavBackend, State,
                         WaypointKind)
from patrol_ros.patrol_node import detector_from_route, route_from_dict


def load(name):
    with open(os.path.join(ROOT, "config", name), encoding="utf-8") as f:
        return json.load(f)


class RouteConfigTests(unittest.TestCase):
    """路线 JSON 的加载与校验"""

    def test_patrol_route_loads(self):
        r = route_from_dict(load("patrol_route.json"))
        self.assertEqual(r.name, "车间巡检 A 线")
        self.assertEqual(len(r), 6)
        self.assertFalse(r.loop)
        self.assertEqual(r.validate(), [])

        # 点位类型与动作解析正确
        by_name = {w.name: w for w in r.waypoints}
        self.assertEqual(by_name["原点"].kind, WaypointKind.HOME)
        self.assertEqual(by_name["配电柜-1"].action, ActionKind.THERMAL)
        self.assertEqual(by_name["总电表"].action, ActionKind.METER)
        self.assertEqual(by_name["充电桩"].kind, WaypointKind.CHARGE)

        # 阈值与下限
        self.assertEqual(by_name["配电柜-1"].thresholds["柜1温度"], 60.0)
        self.assertEqual(by_name["总电表"].lower_bounds["电压"], 380.0)
        self.assertEqual(by_name["总电表"].pose.yaw, 1.5708)

    def test_guide_route_loads(self):
        r = route_from_dict(load("guide_route.json"))
        self.assertEqual(r.kind, WaypointKind.GUIDE)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.validate(), [])
        for w in r.waypoints:
            if w.kind == WaypointKind.GUIDE:
                self.assertTrue(w.speech, "导览点必须有讲解词：%s" % w.name)
                self.assertEqual(w.action, ActionKind.SPEAK)

    def test_detector_built_from_route(self):
        """检测器应由路线自动生成，不用重复配一遍"""
        r = route_from_dict(load("patrol_route.json"))
        det = detector_from_route(r, debounce_up=1)
        self.assertEqual(sorted(det.channels),
                         sorted(["水泵振动", "总电流", "电压", "柜1温度", "柜2温度"]))
        # 上限 / 下限方向正确
        a = det.update("柜1温度", 99.0)
        self.assertIsNotNone(a)
        self.assertEqual(a.direction, "high")
        b = det.update("电压", 300.0)
        self.assertIsNotNone(b)
        self.assertEqual(b.direction, "low")

    def test_detector_dedups_same_item(self):
        """两个点位检测同一项时，检测器只建一条通道（否则会重复告警）"""
        data = {
            "name": "重复项", "loop": False, "max_laps": 1,
            "waypoints": [
                {"name": "a", "pose": {"x": 0, "y": 0},
                 "thresholds": {"温度": 50.0}},
                {"name": "b", "pose": {"x": 1, "y": 0},
                 "thresholds": {"温度": 50.0}},
            ],
        }
        det = detector_from_route(route_from_dict(data))
        self.assertEqual(det.channels, ["温度"])

    def test_patrol_route_actually_runs(self):
        """配置能加载还不够，得能真跑完"""
        r = route_from_dict(load("patrol_route.json"))
        det = detector_from_route(r)
        m = PatrolMission(r, SimNavBackend(speed_mps=2.0), det,
                          reader=lambda item: 10.0)
        self.assertEqual(m.run_once(), State.DONE)
        self.assertEqual(m.progress.failed, 0)
        self.assertEqual(m.progress.visited, 6)

    def test_guide_route_speaks_all_points(self):
        r = route_from_dict(load("guide_route.json"))
        spoken = []
        m = PatrolMission(r, SimNavBackend(speed_mps=2.0),
                          speaker=spoken.append)
        m.run_once()
        self.assertEqual(len(spoken), 4, "四个展点应各播报一次")

    def test_bad_route_is_rejected_at_load(self):
        """坏配置要在加载阶段就暴露，而不是跑到一半才崩"""
        bad = {
            "name": "坏路线", "loop": False, "max_laps": 1,
            "waypoints": [
                {"name": "g", "kind": "guide", "pose": {"x": 0, "y": 0},
                 "action": "speak"},           # 导览点缺讲解词
            ],
        }
        r = route_from_dict(bad)
        self.assertTrue(any("讲解词" in p for p in r.validate()))
        with self.assertRaises(ValueError):
            PatrolMission(r, SimNavBackend())

    def test_infinite_loop_config_rejected(self):
        """loop=true 却不给圈数 -> 必须被拒（否则机器人会一直跑）"""
        bad = {
            "name": "无限循环", "loop": True, "max_laps": 0,
            "waypoints": [{"name": "a", "pose": {"x": 0, "y": 0}}],
        }
        r = route_from_dict(bad)
        self.assertTrue(any("max_laps" in p for p in r.validate()))

    def test_missing_name_raises(self):
        bad = {"name": "x", "waypoints": [{"pose": {"x": 0, "y": 0}}]}
        with self.assertRaises(KeyError):
            route_from_dict(bad)

    def test_defaults_applied(self):
        """只给最少字段时，其余应有合理默认值"""
        r = route_from_dict({"name": "极简",
                             "waypoints": [{"name": "a", "pose": {"x": 1, "y": 2}}]})
        w = r.waypoints[0]
        self.assertEqual(w.kind, WaypointKind.PATROL)
        self.assertEqual(w.action, ActionKind.NONE)
        self.assertEqual(w.dwell_s, 3.0)
        self.assertEqual(w.retry, 1)
        self.assertEqual(w.timeout_s, 60.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
