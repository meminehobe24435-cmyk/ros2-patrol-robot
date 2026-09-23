# -*- coding: utf-8 -*-
"""
patrol_ros.nav_backend — 接真实 Nav2 的导航后端

把 patrol_core 的 NavBackend 接口实现成 ROS2 版本：
    调用 Nav2 的 ``NavigateToPose`` Action，把 Action 结果映射成 NavResult。

⚠️ 重要说明（README 里也写了）：
    本文件**没有在本机运行验证过** —— 开发环境没有 ROS2。
    核心逻辑（任务编排/状态机/异常判定/报告）走的是 SimNavBackend，
    在 tests/ 里有 46 项单测覆盖；**ROS2 这一层是照接口规范写的，需要上机验证**。

设计上刻意做薄：只做"ROS2 世界 <-> patrol_core 世界"的翻译，
不含任何业务判断 —— 所以即使这层有 bug，也不会污染已经测过的任务逻辑。
"""
from __future__ import annotations

import math
import time
from typing import Optional

from patrol_core.backend import NavBackend, NavResult
from patrol_core.waypoint import Pose2D

try:                                     # 只有在装了 ROS2 的环境里才 import
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose

    HAS_ROS2 = True
except Exception:                        # noqa: BLE001
    HAS_ROS2 = False


def yaw_to_quaternion(yaw: float):
    """偏航角 -> 四元数（只绕 Z 轴，ROS2 里最常用的一种）。"""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """四元数 -> 偏航角。"""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


if HAS_ROS2:

    class Ros2NavBackend(Node, NavBackend):
        """
        基于 Nav2 ``NavigateToPose`` Action 的导航后端。

        现场要处理的几件事都在这里：
          · Action 是异步的 -> 内部用 spin_until_future_complete / 自旋等待
          · 超时要能**主动取消**目标，否则机器人会一直往那个点走
          · 被取消 / 被抢占 -> 映射成 ABORTED
          · 导航被障碍挡住 -> Nav2 返回 ABORTED，这里映射成 BLOCKED 便于上层统计
        """

        #: Nav2 的 Action 结果状态码（与 nav2_msgs/action/NavigateToPose 一致）
        STATUS_SUCCEEDED = 4
        STATUS_ABORTED = 6
        STATUS_CANCELED = 5

        def __init__(self, node_name: str = "patrol_nav_backend",
                     action_name: str = "navigate_to_pose",
                     frame_id: str = "map") -> None:
            Node.__init__(self, node_name)
            self.frame_id = frame_id
            self._client = ActionClient(self, NavigateToPose, action_name)
            self._goal_handle = None
            self._current_pose = Pose2D()
            self._cancel_requested = False

            # 订阅当前位姿（Nav2 一般会发布 amcl_pose；也可换 tf2）
            try:
                from geometry_msgs.msg import PoseWithCovarianceStamped

                self.create_subscription(PoseWithCovarianceStamped, "amcl_pose",
                                         self._on_pose, 10)
            except Exception:            # noqa: BLE001
                self.get_logger().warn("未订阅到 amcl_pose，位姿将保持初值")

        # ------------------------------------------------------------ 内部
        def _on_pose(self, msg) -> None:
            p = msg.pose.pose
            self._current_pose = Pose2D(p.position.x, p.position.y,
                                        quaternion_to_yaw(p.orientation.x,
                                                          p.orientation.y,
                                                          p.orientation.z,
                                                          p.orientation.w))

        def _make_goal(self, target: Pose2D):
            goal = NavigateToPose.Goal()
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(target.x)
            ps.pose.position.y = float(target.y)
            qx, qy, qz, qw = yaw_to_quaternion(target.yaw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            goal.pose = ps
            return goal

        # ------------------------------------------------------------ 接口
        def goto(self, target: Pose2D, timeout_s: float,
                 label: str = "") -> NavResult:
            self._cancel_requested = False
            if not self._client.wait_for_server(timeout_sec=min(5.0, timeout_s)):
                self.get_logger().error("Nav2 Action 服务不可用")
                return NavResult.REJECTED

            send_future = self._client.send_goal_async(self._make_goal(target))
            rclpy.spin_until_future_complete(self, send_future)
            self._goal_handle = send_future.result()
            if self._goal_handle is None or not self._goal_handle.accepted:
                return NavResult.REJECTED

            result_future = self._goal_handle.get_result_async()
            deadline = time.monotonic() + timeout_s
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                if self._cancel_requested:
                    self._goal_handle.cancel_goal_async()
                    return NavResult.ABORTED
                if time.monotonic() > deadline:
                    # 超时必须主动取消目标，否则机器人会继续往那边走
                    self._goal_handle.cancel_goal_async()
                    self.get_logger().warn("导航超时，已取消目标：%s" % label)
                    return NavResult.TIMEOUT

            status = result_future.result().status
            if status == self.STATUS_SUCCEEDED:
                return NavResult.SUCCEEDED
            if status == self.STATUS_CANCELED:
                return NavResult.ABORTED
            if status == self.STATUS_ABORTED:
                # Nav2 的 ABORTED 在现场多数是"被障碍挡住/规划失败"
                return NavResult.BLOCKED
            return NavResult.ABORTED

        def cancel(self) -> None:
            self._cancel_requested = True
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()

        def pose(self) -> Pose2D:
            return self._current_pose

        def close(self) -> None:
            try:
                self._client.destroy()
            except Exception:            # noqa: BLE001
                pass
            self.destroy_node()

else:

    class Ros2NavBackend(object):        # type: ignore[no-redef]
        """没有 ROS2 时的占位实现：给出明确错误，而不是 ImportError 堆栈。"""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "未检测到 ROS2（rclpy 不可用）。\n"
                "  · 若只想跑任务逻辑与单元测试：用 patrol_core.SimNavBackend\n"
                "  · 若要在真机上跑：请先 source ROS2 环境（如 /opt/ros/humble/setup.bash）")
