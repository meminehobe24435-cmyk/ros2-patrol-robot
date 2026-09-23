# -*- coding: utf-8 -*-
"""
tests/test_patrol.py — 巡检/导览机器人核心逻辑单元测试

不需要 ROS2：导航由 SimNavBackend / ScriptedNavBackend 提供，
所以"单点失败重试""低电断点续巡""告警去抖"这些**现场最容易出问题的地方**
都能在这里毫秒级覆盖。

  T1 waypoint  点位/路线模型、校验、最近邻排序
  T2 anomaly   去抖、分级、恢复判定
  T3 backend   仿真后端行为（超时/注入失败/取消）
  T4 mission   任务编排：正常巡完、单点失败、重试、低电续巡、中止、
               导览模式、多圈、返航
  T5 report    报告三种格式导出
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patrol_core import (ActionKind, Alarm, AnomalyDetector, ChannelRule,
                         NavResult, PatrolMission, Pose2D, Route, Severity,
                         SimNavBackend, ScriptedNavBackend, State, Waypoint,
                         WaypointKind, build_report, normalize_angle,
                         order_by_nearest, to_csv, to_json, to_text)
from patrol_core.report import save


def mk_patrol_route(loop=False, with_charge=True):
    r = Route(name="车间巡检", loop=loop, kind=WaypointKind.PATROL)
    r.add(Waypoint("原点", Pose2D(0, 0, 0), WaypointKind.HOME))
    r.add(Waypoint("配电柜A", Pose2D(3, 0, 0), action=ActionKind.THERMAL,
                   thresholds={"温度A": 60.0}, dwell_s=2))
    r.add(Waypoint("配电柜B", Pose2D(6, 0, 0), action=ActionKind.METER,
                   thresholds={"电流B": 100.0}, dwell_s=2))
    r.add(Waypoint("水泵房", Pose2D(6, 4, 0), action=ActionKind.CAPTURE,
                   thresholds={"振动": 5.0}, dwell_s=3))
    if with_charge:
        r.add(Waypoint("充电桩", Pose2D(0, 4, 0), WaypointKind.CHARGE))
    return r


def mk_detector():
    d = AnomalyDetector()
    for item, up in (("温度A", 60.0), ("电流B", 100.0), ("振动", 5.0)):
        d.add_rule(ChannelRule(item, upper=up, debounce_up=2, debounce_down=2))
    return d


class T1Waypoint(unittest.TestCase):
    """T1 点位与路线"""

    def test_pose_distance_and_yaw(self):
        a, b = Pose2D(0, 0), Pose2D(3, 4)
        self.assertAlmostEqual(a.distance_to(b), 5.0, places=9)
        self.assertAlmostEqual(a.yaw_to(b), 0.9272952180016122, places=6)

    def test_normalize_angle(self):
        import math
        self.assertAlmostEqual(normalize_angle(0.0), 0.0, places=12)
        self.assertAlmostEqual(normalize_angle(3 * math.pi), math.pi, places=9)
        # 归一到 (-π, π]：-3π 先加 2π 得 -π，-π 不满足 > -π，再加 2π 得 π
        self.assertAlmostEqual(normalize_angle(-3 * math.pi), math.pi, places=9)
        self.assertAlmostEqual(normalize_angle(-math.pi / 2), -math.pi / 2, places=9)
        self.assertTrue(-math.pi < normalize_angle(100.0) <= math.pi)

    def test_waypoint_validation(self):
        with self.assertRaises(ValueError):
            Waypoint("", Pose2D())
        with self.assertRaises(ValueError):
            Waypoint("x", Pose2D(), dwell_s=-1)
        with self.assertRaises(ValueError):
            Waypoint("x", Pose2D(), retry=-1)
        with self.assertRaises(ValueError):
            Waypoint("x", Pose2D(), timeout_s=0)

    def test_duplicate_name_rejected(self):
        r = Route("r")
        r.add(Waypoint("a", Pose2D()))
        with self.assertRaises(ValueError):
            r.add(Waypoint("a", Pose2D(1, 1)))

    def test_is_action_point(self):
        self.assertFalse(Waypoint("t", Pose2D(), action=ActionKind.NONE).is_action_point)
        self.assertTrue(Waypoint("c", Pose2D(), action=ActionKind.CAPTURE).is_action_point)

    def test_route_validate_detects_problems(self):
        r = Route("空")
        self.assertIn("路线为空", r.validate())

        r2 = Route("缺讲解词")
        r2.add(Waypoint("展点", Pose2D(), WaypointKind.GUIDE, ActionKind.SPEAK))
        self.assertTrue(any("讲解词" in p for p in r2.validate()))

        r3 = Route("双充电桩")
        r3.add(Waypoint("c1", Pose2D(), WaypointKind.CHARGE))
        r3.add(Waypoint("c2", Pose2D(1, 1), WaypointKind.CHARGE))
        self.assertTrue(any("充电点超过一个" in p for p in r3.validate()))

        r4 = Route("无检测项")
        r4.add(Waypoint("p", Pose2D(), WaypointKind.PATROL, ActionKind.NONE))
        self.assertTrue(any("没有任何检测项" in p for p in r4.validate()))

        self.assertEqual(mk_patrol_route().validate(), [])

    def test_path_length(self):
        r = Route("直线", loop=False)
        r.add(Waypoint("a", Pose2D(0, 0)))
        r.add(Waypoint("b", Pose2D(3, 4)))
        self.assertAlmostEqual(r.total_path_length(), 5.0, places=9)
        r.loop = True
        self.assertAlmostEqual(r.total_path_length(), 10.0, places=9)

    def test_order_by_nearest_shortens_path(self):
        """乱序的点按最近邻重排后，总路程不应变长"""
        r = Route("乱序", loop=False)
        for n, (x, y) in (("A", (0, 0)), ("B", (10, 0)), ("C", (1, 0)), ("D", (9, 0))):
            r.add(Waypoint(n, Pose2D(x, y)))
        before = r.total_path_length()
        after = order_by_nearest(r, Pose2D(0, 0))
        self.assertLess(after.total_path_length(), before)
        self.assertEqual([w.name for w in after.waypoints], ["A", "C", "D", "B"])


class T2Anomaly(unittest.TestCase):
    """T2 异常判定与去抖"""

    def test_single_spike_is_filtered(self):
        """单次毛刺不应告警（去抖 N=3）"""
        d = AnomalyDetector()
        d.add_rule(ChannelRule("温度", upper=60, debounce_up=3))
        self.assertIsNone(d.update("温度", 200.0))
        self.assertEqual(d.state_of("温度").value, "pending")
        self.assertEqual(d.events, [])

    def test_consecutive_exceed_triggers(self):
        d = AnomalyDetector()
        d.add_rule(ChannelRule("温度", upper=60, debounce_up=3))
        d.update("温度", 70)
        d.update("温度", 70)
        a = d.update("温度", 70)
        self.assertIsNotNone(a)
        self.assertEqual(d.state_of("温度").value, "alarming")
        self.assertEqual(len(d.events), 1)

    def test_recovery_debounced(self):
        """恢复也要连续 N 次，避免告警反复横跳"""
        d = AnomalyDetector()
        d.add_rule(ChannelRule("温度", upper=60, debounce_up=2, debounce_down=3))
        d.update("温度", 70); d.update("温度", 70)
        self.assertTrue(d.active_alarms())
        d.update("温度", 10)
        self.assertEqual(d.state_of("温度").value, "recovering")
        self.assertTrue(d.active_alarms(), "恢复去抖期间告警仍应存在")
        d.update("温度", 10); d.update("温度", 10)
        self.assertEqual(d.state_of("温度").value, "normal")
        self.assertEqual(d.active_alarms(), [])

    def test_critical_by_absolute_threshold(self):
        d = AnomalyDetector()
        d.add_rule(ChannelRule("温度", upper=60, debounce_up=1, critical_abs=80))
        a = d.update("温度", 65)
        self.assertEqual(a.severity, Severity.WARNING)
        a = d.update("温度", 85)
        self.assertEqual(a.severity, Severity.CRITICAL)

    def test_critical_by_ratio(self):
        d = AnomalyDetector()
        d.add_rule(ChannelRule("压力", upper=100, debounce_up=1, critical_ratio=1.5))
        self.assertEqual(d.update("压力", 110).severity, Severity.WARNING)
        # 超出幅度 >= 100*(1.5-1) = 50 -> CRITICAL
        self.assertEqual(d.update("压力", 155).severity, Severity.CRITICAL)

    def test_lower_bound(self):
        d = AnomalyDetector()
        d.add_rule(ChannelRule("电量", lower=20, debounce_up=1))
        a = d.update("电量", 15)
        self.assertIsNotNone(a)
        self.assertEqual(a.direction, "low")

    def test_lower_bound_critical(self):
        """
        下限 20、critical_abs=5：
          · 读数 19（超出幅度 1，远小于 20*0.5=10）-> WARNING
          · 读数 10（超出幅度 10，恰好达到 20*(1.5-1)=10）-> CRITICAL（比例规则）
          · 读数 3（低于 critical_abs=5）-> CRITICAL（绝对阈值规则）
        """
        d = AnomalyDetector()
        d.add_rule(ChannelRule("电量", lower=20, debounce_up=1, critical_abs=5))
        self.assertEqual(d.update("电量", 19).severity, Severity.WARNING)
        self.assertEqual(d.update("电量", 10).severity, Severity.CRITICAL)
        self.assertEqual(d.update("电量", 3).severity, Severity.CRITICAL)

    def test_unknown_channel_raises(self):
        d = AnomalyDetector()
        with self.assertRaises(KeyError):
            d.update("没有的通道", 1.0)

    def test_rule_validation(self):
        d = AnomalyDetector()
        with self.assertRaises(ValueError):
            d.add_rule(ChannelRule("x"))
        with self.assertRaises(ValueError):
            d.add_rule(ChannelRule("y", upper=1, debounce_up=0))
        d.add_rule(ChannelRule("ok", upper=1))
        with self.assertRaises(ValueError):
            d.add_rule(ChannelRule("ok", upper=1))

    def test_summary_counts(self):
        d = AnomalyDetector()
        d.add_rule(ChannelRule("a", upper=10, debounce_up=1))
        d.add_rule(ChannelRule("b", upper=10, debounce_up=1, critical_abs=100))
        d.update("a", 20)
        d.update("b", 200)
        s = d.summary()
        self.assertEqual(s["warning"] + s["critical"], 2)

    def test_repeated_alarm_counts_events(self):
        d = AnomalyDetector()
        d.add_rule(ChannelRule("a", upper=10, debounce_up=1, debounce_down=1))
        d.update("a", 20)
        d.update("a", 5)
        d.update("a", 20)
        self.assertEqual(len(d.events), 2)
        self.assertEqual(d.alarm_count_of("a"), 2)


class T3Backend(unittest.TestCase):
    """T3 仿真导航后端"""

    def test_move_takes_time_and_can_timeout(self):
        """距离远、超时短 -> 应判超时，并停在半路"""
        b = SimNavBackend(start=Pose2D(0, 0), speed_mps=0.5)
        r = b.goto(Pose2D(10, 0), timeout_s=4.0, label="远点")
        self.assertEqual(r, NavResult.TIMEOUT)
        self.assertLess(b.pose().x, 10.0)
        self.assertGreater(b.pose().x, 0.0)

    def test_reach_when_enough_time(self):
        b = SimNavBackend(start=Pose2D(0, 0), speed_mps=1.0)
        r = b.goto(Pose2D(3, 4), timeout_s=10.0, label="近点")
        self.assertEqual(r, NavResult.SUCCEEDED)
        self.assertAlmostEqual(b.pose().x, 3.0, places=9)
        self.assertEqual(b.visits, ["近点"])

    def test_inject_failure_by_label(self):
        b = SimNavBackend(fail_at={"坏点": NavResult.BLOCKED})
        self.assertEqual(b.goto(Pose2D(1, 0), 10, label="坏点"), NavResult.BLOCKED)
        self.assertEqual(b.goto(Pose2D(1, 0), 10, label="好点"), NavResult.SUCCEEDED)

    def test_inject_failure_by_call_index(self):
        b = SimNavBackend(fail_at={1: "aborted"})
        self.assertEqual(b.goto(Pose2D(1, 0), 10, label="a"), NavResult.ABORTED)
        self.assertEqual(b.goto(Pose2D(1, 0), 10, label="a"), NavResult.SUCCEEDED)

    def test_cancel(self):
        b = SimNavBackend()
        b.cancel()
        self.assertEqual(b.goto(Pose2D(1, 0), 10), NavResult.ABORTED)

    def test_jitter_produces_blocked(self):
        b = SimNavBackend(jitter=1.0)         # 100% 概率
        self.assertEqual(b.goto(Pose2D(1, 0), 10), NavResult.BLOCKED)

    def test_scripted_backend(self):
        b = ScriptedNavBackend([NavResult.BLOCKED, NavResult.TIMEOUT,
                                NavResult.SUCCEEDED])
        self.assertEqual(b.goto(Pose2D(1, 0), 5), NavResult.BLOCKED)
        self.assertEqual(b.goto(Pose2D(1, 0), 5), NavResult.TIMEOUT)
        self.assertEqual(b.goto(Pose2D(1, 0), 5), NavResult.SUCCEEDED)
        # 脚本用完之后默认成功
        self.assertEqual(b.goto(Pose2D(1, 0), 5), NavResult.SUCCEEDED)
        self.assertEqual(len(b.targets), 4)


class T4Mission(unittest.TestCase):
    """T4 任务编排与状态机"""

    def _run(self, route, backend, readings=None, **kw):
        readings = readings or {}
        det = mk_detector()
        m = PatrolMission(route, backend, det,
                          reader=lambda item: readings.get(item, 20.0), **kw)
        return m

    def test_normal_patrol_completes(self):
        b = SimNavBackend(speed_mps=10.0)
        m = self._run(mk_patrol_route(), b)
        st = m.run_once()
        self.assertEqual(st, State.DONE)
        self.assertEqual(m.progress.visited, 5)
        self.assertEqual(m.progress.failed, 0)
        self.assertEqual(m.progress.alarms, 0)
        self.assertEqual(len(m.records), 5)

    def test_waypoint_failure_retries_then_skips(self):
        """单点两次都失败 -> 跳过并记录，**后面的点继续走**（不能整趟崩）"""
        route = mk_patrol_route(with_charge=False)
        b = SimNavBackend(speed_mps=10.0, fail_at={1: NavResult.BLOCKED,
                                                   2: NavResult.BLOCKED})
        m = self._run(route, b)
        st = m.run_once()
        self.assertEqual(st, State.DONE)
        # 第 1 个点（原点后的第一个）重试 2 次都失败
        failed = [r for r in m.records if not r.ok]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].attempts, 2)
        self.assertIn("blocked", failed[0].nav_result)
        self.assertEqual(m.progress.skipped, 1)
        # 关键：后面的点仍被到访
        self.assertGreaterEqual(m.progress.succeeded, 3)

    def test_retry_succeeds_second_time(self):
        route = mk_patrol_route(with_charge=False)
        b = SimNavBackend(speed_mps=10.0, fail_at={2: NavResult.BLOCKED})
        m = self._run(route, b)
        m.run_once()
        ok_recs = [r for r in m.records if r.ok]
        self.assertTrue(any(r.attempts == 2 for r in ok_recs),
                        "应有记录显示第二次尝试成功")

    def test_alarm_generated_when_over_threshold(self):
        """
        注意用 debounce_up=1 的检测器：
        mk_detector() 默认去抖 2 次，而巡检每个点只读一次，
        所以用默认检测器时**单次超限不会告警**（那正是去抖在起作用）。
        """
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route(with_charge=False)
        det = AnomalyDetector()
        det.add_rule(ChannelRule("温度A", upper=60.0, debounce_up=1))
        m = PatrolMission(route, b, det,
                          reader=lambda item: {"温度A": 999.0}.get(item, 20.0))
        m.run_once()
        self.assertGreater(m.progress.alarms, 0)
        self.assertTrue(any(r.alarms for r in m.records))

    def test_debounce_swallows_single_spike_in_patrol(self):
        """反过来的用例：去抖 2 次时，只读一次的超限**不应**告警"""
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route(with_charge=False)
        m = self._run(route, b, readings={"温度A": 999.0})   # mk_detector: debounce_up=2
        m.run_once()
        self.assertEqual(m.progress.alarms, 0,
                         "单次读数不应触发告警（去抖 N=2）")

    def test_low_battery_goes_charging_and_resumes(self):
        """低电量 -> 去充电 -> **从断点续巡**（不是从头再来）"""
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route()
        m = self._run(route, b)
        st = m.run_once(battery_ratio=0.1)
        self.assertEqual(st, State.DONE)
        # 充电点应被访问过
        self.assertIn("充电桩", b.visits)
        # 状态日志里应出现低电量与续巡
        joined = "\n".join(m.log)
        self.assertIn("low_battery", joined)
        self.assertIn("续巡", joined)

    def test_abort_then_resume_from_breakpoint(self):
        """中止后应记录断点，再次 run_once 从断点继续"""
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route(with_charge=False)
        m = self._run(route, b)

        # 用一个 reader 在第 2 个点时请求中止
        calls = {"n": 0}

        def reader(item):
            calls["n"] += 1
            if calls["n"] == 2:
                m.abort()
            return 20.0

        m.reader = reader
        st = m.run_once()
        self.assertEqual(st, State.PAUSED)
        self.assertIsNotNone(m._pending_resume_index)

        # 恢复：应设置 resumed_from
        b2 = SimNavBackend(speed_mps=10.0)
        m.backend = b2
        st2 = m.run_once()
        self.assertEqual(st2, State.DONE)
        self.assertGreater(m.progress.resumed_from, 0)

    def test_empty_route_is_rejected(self):
        """空路线属于配置错误，构造时就该被拒（而不是跑一遍什么都没做）"""
        b = SimNavBackend()
        with self.assertRaises(ValueError):
            PatrolMission(Route("空"), b)

    def test_route_validation_blocks_construction(self):
        r = Route("坏")
        r.add(Waypoint("g", Pose2D(), WaypointKind.GUIDE, ActionKind.SPEAK))
        with self.assertRaises(ValueError):
            PatrolMission(r, SimNavBackend())

    def test_loop_without_max_laps_is_rejected(self):
        """loop=True 但不给圈数 -> 必须被拒（否则会无限循环，真实部署会跑不停）"""
        r = Route("无限循环", loop=True)
        r.add(Waypoint("a", Pose2D()))
        self.assertTrue(any("max_laps" in p for p in r.validate()))
        with self.assertRaises(ValueError):
            PatrolMission(r, SimNavBackend())

    def test_loop_with_max_laps(self):
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route(loop=True, with_charge=False)
        route.max_laps = 3
        m = self._run(route, b)
        m.run_once()
        self.assertEqual(m.progress.lap, 3)
        self.assertEqual(m.progress.visited, 4 * 3)

    def test_guide_mode_speaks(self):
        """导览模式：到点调用 speaker 播报讲解词"""
        spoken = []
        r = Route("展厅导览", kind=WaypointKind.GUIDE, loop=False)
        r.add(Waypoint("入口", Pose2D(0, 0), WaypointKind.HOME))
        r.add(Waypoint("展品1", Pose2D(2, 0), WaypointKind.GUIDE, ActionKind.SPEAK,
                       speech="这是第一件展品，产自 1998 年", dwell_s=5))
        r.add(Waypoint("展品2", Pose2D(4, 0), WaypointKind.GUIDE, ActionKind.SPEAK,
                       speech="第二件展品介绍了工艺流程", dwell_s=5))
        b = SimNavBackend(speed_mps=10.0)
        m = PatrolMission(r, b, speaker=spoken.append)
        st = m.run_once()
        self.assertEqual(st, State.DONE)
        self.assertEqual(len(spoken), 2)
        self.assertIn("第一件展品", spoken[0])
        self.assertTrue(any(rec.speech for rec in m.records))

    def test_return_home_at_end(self):
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route(loop=False)
        m = self._run(route, b)
        m.run_once()
        self.assertIn("原点", b.visits)

    def test_reader_exception_does_not_crash(self):
        """读数异常不应让整趟任务崩掉"""
        def bad_reader(item):
            raise IOError("传感器离线")
        b = SimNavBackend(speed_mps=10.0)
        m = PatrolMission(mk_patrol_route(with_charge=False), b,
                          mk_detector(), reader=bad_reader)
        st = m.run_once()
        self.assertEqual(st, State.DONE)
        self.assertTrue(any("读数失败" in r.note for r in m.records))

    def test_state_transitions_logged(self):
        b = SimNavBackend(speed_mps=10.0)
        m = self._run(mk_patrol_route(with_charge=False), b)
        m.run_once()
        joined = "\n".join(m.log)
        for s in ("patrolling", "inspecting", "done"):
            self.assertIn(s, joined, "缺少状态 %s 的日志" % s)


class T5Report(unittest.TestCase):
    """T5 报告导出"""

    def _report(self):
        b = SimNavBackend(speed_mps=10.0)
        route = mk_patrol_route(with_charge=False)
        det = AnomalyDetector()
        det.add_rule(ChannelRule("温度A", upper=60.0, debounce_up=1))
        m = PatrolMission(route, b, det,
                          reader=lambda item: {"温度A": 999.0}.get(item, 20.0))
        m.run_once()
        return build_report(m)

    def test_json_roundtrip(self):
        rep = self._report()
        txt = to_json(rep)
        back = json.loads(txt)
        self.assertEqual(back["progress"]["visited"], rep["progress"]["visited"])
        self.assertIn("steps", back)

    def test_text_contains_summary(self):
        txt = to_text(self._report())
        self.assertIn("机器人巡检报告", txt)
        self.assertIn("车间巡检", txt)
        self.assertIn("告警", txt)

    def test_csv_has_bom_and_rows(self):
        csv_txt = to_csv(self._report())
        self.assertTrue(csv_txt.startswith("\ufeff"), "CSV 应带 BOM 以便 Excel 打开")
        lines = [l for l in csv_txt.split("\n") if l.strip()]
        self.assertGreater(len(lines), 3)
        self.assertIn("点位", lines[0])

    def test_save_three_formats(self):
        rep = self._report()
        with tempfile.TemporaryDirectory() as d:
            prefix = os.path.join(d, "patrol")
            paths = save(rep, prefix)
            for k, p in paths.items():
                self.assertTrue(os.path.exists(p), "缺少 %s" % k)
                self.assertGreater(os.path.getsize(p), 10)
            with open(paths["json"], encoding="utf-8") as f:
                self.assertEqual(json.load(f)["title"], rep["title"])

    def test_report_records_alarms_with_severity(self):
        rep = self._report()
        self.assertGreater(rep["alarm_summary"]["critical"] +
                           rep["alarm_summary"]["warning"], 0)
        items = rep["alarm_items"]
        self.assertIn("温度A", items)


if __name__ == "__main__":
    unittest.main(verbosity=2)
