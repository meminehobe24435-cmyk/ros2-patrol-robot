# -*- coding: utf-8 -*-
"""
demo.py — 巡检 / 导览两种模式的端到端演示（不需要 ROS2）

跑法：
    python demo.py            # 跑全部场景
    python demo.py patrol     # 只跑巡检
    python demo.py guide      # 只跑导览
    python demo.py fault      # 只跑故障与续巡
"""
import sys

# Windows 控制台默认 GBK，打印非 ASCII（如摄氏度符号）会抛 UnicodeEncodeError，
# 这里统一重配为 UTF-8 并容错，避免演示因编码问题中断。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from patrol_core import (ActionKind, AnomalyDetector, ChannelRule, PatrolMission,
                         Pose2D, Route, SimNavBackend, Waypoint, WaypointKind,
                         NavResult, build_report, to_text, save)


def banner(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


# ---------------------------------------------------------------- 巡检路线
def build_patrol_route():
    """车间巡检路线：配电柜测温 + 电表读数 + 水泵振动"""
    r = Route(name="车间巡检 A 线", loop=False, return_home=True)
    r.add(Waypoint("原点", Pose2D(0.0, 0.0, 0.0), WaypointKind.HOME))
    r.add(Waypoint("配电柜-1", Pose2D(6.0, 0.0, 0.0), action=ActionKind.THERMAL,
                   thresholds={"柜1温度": 60.0}, dwell_s=3.0))
    r.add(Waypoint("配电柜-2", Pose2D(12.0, 0.0, 0.0), action=ActionKind.THERMAL,
                   thresholds={"柜2温度": 60.0}, dwell_s=3.0))
    r.add(Waypoint("总电表", Pose2D(12.0, 5.0, 0.0), action=ActionKind.METER,
                   thresholds={"总电流": 100.0}, lower_bounds={"电压": 380.0},
                   dwell_s=2.0))
    r.add(Waypoint("水泵房", Pose2D(6.0, 5.0, 0.0), action=ActionKind.CAPTURE,
                   thresholds={"水泵振动": 5.0}, dwell_s=4.0))
    r.add(Waypoint("充电桩", Pose2D(0.0, 5.0, 0.0), WaypointKind.CHARGE))
    return r


def build_patrol_detector():
    d = AnomalyDetector()
    d.add_rule(ChannelRule("柜1温度", upper=60.0, unit="℃", debounce_up=1,
                           critical_abs=80.0))
    d.add_rule(ChannelRule("柜2温度", upper=60.0, unit="℃", debounce_up=1,
                           critical_abs=80.0))
    d.add_rule(ChannelRule("总电流", upper=100.0, unit="A", debounce_up=1,
                           critical_ratio=1.5))
    d.add_rule(ChannelRule("电压", lower=380.0, unit="V", debounce_up=1,
                           critical_abs=360.0))
    d.add_rule(ChannelRule("水泵振动", upper=5.0, unit="mm/s", debounce_up=1))
    return d


def scenario_patrol():
    banner("场景一：巡检模式（含超限告警）")
    # 用"会读数"的模拟传感器：柜2 温度偏高、电压偏低
    readings = {
        "柜1温度": 42.5,
        "柜2温度": 88.0,      # 超上限 60 且 >= critical_abs 80 -> CRITICAL
        "总电流": 76.0,
        "电压": 372.0,        # 低于下限 380 但 > critical_abs 360 -> WARNING
        "水泵振动": 2.1,
    }
    backend = SimNavBackend(start=Pose2D(0, 0), speed_mps=1.2)
    mission = PatrolMission(build_patrol_route(), backend, build_patrol_detector(),
                            reader=lambda item: readings.get(item, 0.0))
    mission.run_once()
    rep = build_report(mission, "车间巡检报告")
    print(to_text(rep))
    paths = save(rep, "patrol_report")
    print("\n已导出：" + "、".join(paths.values()))
    return mission


# ---------------------------------------------------------------- 导览路线
def build_guide_route():
    """展厅导览路线：到点播报讲解词，并等待游客"""
    r = Route(name="展厅导览 B 线", loop=False, return_home=True,
              kind=WaypointKind.GUIDE)
    r.add(Waypoint("接待点", Pose2D(0.0, 0.0, 0.0), WaypointKind.HOME))
    r.add(Waypoint("序厅", Pose2D(3.0, 0.0, 0.0), WaypointKind.GUIDE,
                   ActionKind.SPEAK, dwell_s=20.0,
                   speech="欢迎参观，序厅介绍的是本馆的整体布局与参观动线。"))
    r.add(Waypoint("工艺展区", Pose2D(8.0, 0.0, 0.0), WaypointKind.GUIDE,
                   ActionKind.SPEAK, dwell_s=35.0,
                   speech="这里是工艺展区，展示从原材料到成品的完整工艺流程。"))
    r.add(Waypoint("产品展区", Pose2D(8.0, 6.0, 0.0), WaypointKind.GUIDE,
                   ActionKind.SPEAK, dwell_s=30.0,
                   speech="产品展区陈列了历代主力产品，右侧是最新发布的一代。"))
    r.add(Waypoint("休息区", Pose2D(0.0, 6.0, 0.0), WaypointKind.GUIDE,
                   ActionKind.SPEAK, dwell_s=15.0,
                   speech="参观到这里结束，休息区提供饮水，出口在您的左前方。"))
    return r


def scenario_guide():
    banner("场景二：导览模式（到点语音讲解）")
    spoken = []
    backend = SimNavBackend(start=Pose2D(0, 0), speed_mps=1.0)
    mission = PatrolMission(build_guide_route(), backend,
                            speaker=lambda text: spoken.append(text))
    mission.run_once()
    print("讲解播报记录：")
    for i, text in enumerate(spoken, 1):
        print("  %d. %s" % (i, text))
    rep = build_report(mission, "展厅导览记录")
    print()
    print(to_text(rep))
    return mission


# ------------------------------------------------------------ 故障与续巡
def scenario_fault():
    banner("场景三：故障处理与断点续巡")
    route = build_patrol_route()

    # 1) 导航失败：第 2 个点两次都被挡住 -> 跳过，但后面的点继续走
    backend = SimNavBackend(speed_mps=1.2,
                            fail_at={"配电柜-2": NavResult.BLOCKED})
    m1 = PatrolMission(route, backend, build_patrol_detector(),
                       reader=lambda item: 30.0)
    m1.run_once()
    skipped = [r for r in m1.records if not r.ok]
    print("① 单点被挡：尝试 %d 次后跳过，其余点仍完成（成功 %d / 跳过 %d）"
          % (skipped[0].attempts if skipped else 0,
             m1.progress.succeeded, m1.progress.skipped))
    print("   失败点：%s（%s）" % (skipped[0].waypoint, skipped[0].nav_result)
          if skipped else "   无失败点")

    # 2) 低电量 -> 去充电 -> 从断点续巡
    backend2 = SimNavBackend(speed_mps=1.2)
    m2 = PatrolMission(route, backend2, build_patrol_detector(),
                       reader=lambda item: 30.0)
    m2.run_once(battery_ratio=0.12)
    print("\n② 低电量续巡：")
    print("   到访充电桩：%s" % ("充电桩" in backend2.visits))
    print("   状态日志：")
    for line in m2.log:
        if any(k in line for k in ("low_battery", "charging", "续巡", "done")):
            print("     " + line)

    # 3) 中止 -> 断点续巡
    backend3 = SimNavBackend(speed_mps=1.2)
    m3 = PatrolMission(route, backend3, build_patrol_detector(),
                       reader=lambda item: 30.0)
    n = {"i": 0}

    def reader(item):
        n["i"] += 1
        if n["i"] == 3:
            m3.abort()
        return 30.0

    m3.reader = reader
    st = m3.run_once()
    print("\n③ 中止后状态：%s，断点在第 %d 个点"
          % (st.value, (m3._pending_resume_index or 0) + 1))
    m3.backend = SimNavBackend(start=backend3.pose(), speed_mps=1.2)
    m3.run_once()
    print("   续巡完成：resumed_from = 第 %d 个点，最终状态 %s"
          % (m3.progress.resumed_from + 1, m3.state.value))
    return m1, m2, m3


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(__doc__)
    if which in ("all", "patrol"):
        scenario_patrol()
    if which in ("all", "guide"):
        scenario_guide()
    if which in ("all", "fault"):
        scenario_fault()
    print("\n全部演示结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
