# -*- coding: utf-8 -*-
"""
patrol_ros — ROS2 节点层（把 patrol_core 接到真实机器人上）

内容：
  · nav_backend.Ros2NavBackend   调 Nav2 的 NavigateToPose Action
  · patrol_node.PatrolNode       巡检/导览主节点（参数/话题/服务）
  · sensor_monitor.py            传感器监测节点（激光/IMU/温度 -> 检测项）
  · interfaces/                  自定义 msg / srv / action 定义
  · launch/patrol.launch.py      一键启动（含 route 文件参数）

⚠️ **重要**：本目录的代码**没有在装有 ROS2 的机器上运行验证过** ——
   开发环境没有 ROS2。它们按 rclpy / Nav2 的接口规范编写，
   上机前请先在仿真（Gazebo + Nav2）里跑一遍。

   之所以敢这么分，是因为**业务逻辑不在这里**：
   任务编排、状态机、异常判定、报告生成全在 patrol_core，
   由 tests/ 的 46 项单测覆盖，不需要 ROS2 就能验证。
   这一层只做"ROS2 世界 <-> core 世界"的翻译。
"""

from .patrol_node import route_from_dict, detector_from_route
from .nav_backend import yaw_to_quaternion, quaternion_to_yaw

__version__ = "1.0.0"

__all__ = ["route_from_dict", "detector_from_route",
           "yaw_to_quaternion", "quaternion_to_yaw"]
