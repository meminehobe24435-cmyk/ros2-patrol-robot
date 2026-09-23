# -*- coding: utf-8 -*-
"""
patrol_core.report — 巡检/导览报告生成

现场交付物就是这个：一张表 + 一段结论。
格式上刻意做成三种，因为三种人要看：
  · 运维/客户看**文本摘要**（邮件里贴一段就行）
  · 工程师看 **JSON**（接自己的看板）
  · 管理/留档要 **CSV**（Excel 打开）
"""
from __future__ import annotations

import csv
import io
import json
import time
from typing import Dict, List, Optional

from .anomaly import Alarm, Severity
from .mission import PatrolMission, StepRecord


def _alarm_dict(a: Alarm) -> Dict:
    return {
        "item": a.item,
        "value": a.value,
        "threshold": a.threshold,
        "severity": a.severity.value,
        "direction": a.direction,
    }


def build_report(mission: PatrolMission, title: str = "机器人巡检报告") -> Dict:
    """把任务执行结果整理成结构化报告。"""
    recs: List[StepRecord] = mission.records
    alarms = [a for r in recs for a in r.alarms]
    by_sev = {s.value: 0 for s in Severity}
    for a in alarms:
        by_sev[a.severity.value] += 1

    failed_items = [r.waypoint for r in recs if not r.ok]
    alarm_items = sorted({a.item for a in alarms})

    return {
        "title": title,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "route": mission.route.name,
        "state": mission.state.value,
        "progress": {
            "laps": mission.progress.lap,
            "visited": mission.progress.visited,
            "succeeded": mission.progress.succeeded,
            "failed": mission.progress.failed,
            "skipped": mission.progress.skipped,
            "alarms": len(alarms),
            "resumed_from": mission.progress.resumed_from,
        },
        "alarm_summary": by_sev,
        "alarm_items": alarm_items,
        "failed_waypoints": failed_items,
        "steps": [
            {
                "waypoint": r.waypoint,
                "ok": r.ok,
                "attempts": r.attempts,
                "nav_result": r.nav_result,
                "dwell_s": r.dwell_s,
                "readings": r.readings,
                "alarms": [_alarm_dict(a) for a in r.alarms],
                "speech": r.speech,
                "note": r.note,
            }
            for r in recs
        ],
        "log": list(mission.log),
    }


def to_json(report: Dict, indent: int = 2) -> str:
    return json.dumps(report, ensure_ascii=False, indent=indent)


def to_csv(report: Dict, bom: bool = True) -> str:
    """
    导出 CSV。默认带 UTF-8 BOM —— Windows 上 Excel 打开中文不乱码，
    这是现场最容易被投诉的一个小细节。
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["点位", "结果", "尝试次数", "导航结果", "停留(s)", "读数", "告警", "讲解", "备注"])
    for s in report["steps"]:
        readings = "; ".join("%s=%.3f" % (k, v) for k, v in s["readings"].items())
        alarms = "; ".join("%s(%s)" % (a["item"], a["severity"]) for a in s["alarms"])
        w.writerow([
            s["waypoint"],
            "成功" if s["ok"] else "失败",
            s["attempts"],
            s["nav_result"],
            "%.1f" % s["dwell_s"],
            readings,
            alarms,
            s["speech"],
            s["note"],
        ])
    p = report["progress"]
    w.writerow([])
    w.writerow(["# 汇总", "圈数", p["laps"], "到访", p["visited"],
                "成功", p["succeeded"], "失败", p["failed"], "告警", p["alarms"]])
    out = buf.getvalue()
    return ("\ufeff" + out) if bom else out


def to_text(report: Dict) -> str:
    """给人看的摘要（邮件/钉钉里直接贴）。"""
    p = report["progress"]
    lines = [
        "=" * 58,
        "%s（%s）" % (report["title"], report["generated_at"]),
        "=" * 58,
        "路线：%s　结束状态：%s" % (report["route"], report["state"]),
        "圈数 %d｜到访 %d｜成功 %d｜失败 %d｜告警 %d"
        % (p["laps"], p["visited"], p["succeeded"], p["failed"], p["alarms"]),
    ]
    if p["resumed_from"]:
        lines.append("断点续巡：从第 %d 个点继续" % (p["resumed_from"] + 1))

    sv = report["alarm_summary"]
    if any(sv.values()):
        lines.append("告警分级：CRITICAL %d｜WARNING %d" % (sv["critical"], sv["warning"]))
    if report["failed_waypoints"]:
        lines.append("失败点位：" + "、".join(report["failed_waypoints"]))

    lines.append("-" * 58)
    for s in report["steps"]:
        flag = "OK " if s["ok"] else "FAIL"
        extra = ""
        if s["readings"]:
            extra = "  " + " ".join("%s=%.2f" % (k, v) for k, v in s["readings"].items())
        if s["alarms"]:
            extra += "  [!] " + "、".join("%s(%s)" % (a["item"], a["severity"]) for a in s["alarms"])
        lines.append("[%s] %-16s%s%s" % (flag, s["waypoint"], extra,
                                        ("  备注:" + s["note"]) if s["note"] else ""))
    lines.append("=" * 58)
    return "\n".join(lines)


def save(report: Dict, path_prefix: str) -> Dict[str, str]:
    """三种格式各存一份，返回 {格式: 路径}。"""
    paths = {
        "json": path_prefix + ".json",
        "csv": path_prefix + ".csv",
        "txt": path_prefix + ".txt",
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        f.write(to_json(report))
    with open(paths["csv"], "w", encoding="utf-8-sig", newline="") as f:
        f.write(to_csv(report, bom=False))
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write(to_text(report))
    return paths
