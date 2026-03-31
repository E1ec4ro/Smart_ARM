#!/usr/bin/env python3
"""
Исполнитель целей для нейросети: подписка на geometry_msgs/PoseStamped → план MoveIt → /arm_controller/command.

Рекомендуемый контур:
  камера / детектор (в т.ч. нейросеть) → PoseStamped (world или base_link) → этот узел → траектория.

Так вы отделяете «где цель» (NN) от «как доехать без столкновений» (MoveIt).
"""
from __future__ import annotations

import copy
import math
import threading
import time

import moveit_commander
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
from trajectory_msgs.msg import JointTrajectory


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def param_bool(name: str, default: bool) -> bool:
    v = rospy.get_param(name, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return bool(v)


class NnGoalMoveIt:
    def __init__(self) -> None:
        rospy.init_node("nn_goal_moveit", anonymous=False)

        self.world_frame = rospy.get_param("~world_frame", "world")
        self.planning_frame = rospy.get_param("~planning_frame", "base_link")
        self.world_to_base_link_xyz = rospy.get_param("~world_to_base_link_xyz", [0.0, 0.0, 0.20])
        self.group_name = rospy.get_param("~group_name", "manipulator")
        self.arm_command_topic = rospy.get_param("~arm_command_topic", "/arm_controller/command")
        self.joint_names = rospy.get_param(
            "~joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.goal_pose_topic = rospy.get_param("~goal_pose_topic", "/nn_robot_goal")
        self.execute_on_receive = param_bool("~execute_on_receive", True)
        self.min_interval_sec = float(rospy.get_param("~min_interval_sec", 0.5))
        self.trajectory_time_scale = float(rospy.get_param("~trajectory_time_scale", 2.0))

        self.ee_orientation_mode = str(rospy.get_param("~ee_orientation_mode", "none")).strip().lower()
        self._quat_down = None
        if self.ee_orientation_mode == "down":
            rpy = rospy.get_param("~ee_down_rpy", [0.0, math.pi, 0.0])
            self._quat_down = quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))

        self._lock = threading.Lock()
        self._last_goal: PoseStamped | None = None
        self._last_exec_time = 0.0

        self.pub_arm = rospy.Publisher(self.arm_command_topic, JointTrajectory, queue_size=1)

        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        try:
            self.group.set_pose_reference_frame(self.planning_frame)
        except Exception:
            pass
        ee_link = rospy.get_param("~ee_link", "")
        if isinstance(ee_link, str) and ee_link:
            try:
                self.group.set_end_effector_link(ee_link)
            except Exception:
                rospy.logwarn("ee_link=%s not set", ee_link)

        self.group.set_planning_time(float(rospy.get_param("~planning_time", 10.0)))
        self.group.set_num_planning_attempts(int(rospy.get_param("~planning_attempts", 5)))
        self.group.allow_replanning(param_bool("~allow_replanning", True))
        self.group.set_max_velocity_scaling_factor(float(rospy.get_param("~vel_scale", 0.2)))
        self.group.set_max_acceleration_scaling_factor(float(rospy.get_param("~acc_scale", 0.2)))
        self.group.set_goal_position_tolerance(float(rospy.get_param("~pos_tol", 0.02)))
        self.group.set_goal_orientation_tolerance(float(rospy.get_param("~ori_tol", 0.1)))

        rospy.Subscriber(self.goal_pose_topic, PoseStamped, self._on_goal, queue_size=1)
        rospy.Service("~move_to_latest_goal", Trigger, self._srv_move)

        rospy.loginfo(
            "nn_goal_moveit: goal_topic=%s execute_on_receive=%s frame: use %s or %s",
            self.goal_pose_topic,
            self.execute_on_receive,
            self.world_frame,
            self.planning_frame,
        )

    def _world_to_base(self, ps: PoseStamped) -> PoseStamped:
        dx, dy, dz = (
            float(self.world_to_base_link_xyz[0]),
            float(self.world_to_base_link_xyz[1]),
            float(self.world_to_base_link_xyz[2]),
        )
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.planning_frame
        out.pose.position.x = float(ps.pose.position.x - dx)
        out.pose.position.y = float(ps.pose.position.y - dy)
        out.pose.position.z = float(ps.pose.position.z - dz)
        out.pose.orientation.w = 1.0
        return out

    def _goal_in_planning_frame(self, msg: PoseStamped) -> tuple[float, float, float, bool]:
        """Возвращает (x,y,z, use_pose) в base_link."""
        fid = (msg.header.frame_id or "").strip()
        if fid == self.planning_frame or fid == "":
            p = msg.pose.position
            return float(p.x), float(p.y), float(p.z), self.ee_orientation_mode == "down"
        if fid == self.world_frame:
            b = self._world_to_base(msg)
            p = b.pose.position
            return float(p.x), float(p.y), float(p.z), self.ee_orientation_mode == "down"
        rospy.logwarn("Неизвестный frame_id=%s, ожидаю %s или %s", fid, self.world_frame, self.planning_frame)
        p = msg.pose.position
        return float(p.x), float(p.y), float(p.z), self.ee_orientation_mode == "down"

    def _plan(self, x: float, y: float, z: float, use_pose: bool) -> object | None:
        self.group.clear_pose_targets()
        try:
            self.group.set_start_state_to_current_state()
        except Exception:
            pass
        if use_pose and self._quat_down is not None:
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = z
            p.orientation = self._quat_down
            self.group.set_pose_target(p, self.group.get_end_effector_link())
        else:
            self.group.set_position_target([x, y, z], self.group.get_end_effector_link())
        return self.group.plan()

    def _execute_plan(self, plan) -> None:
        try:
            jt = plan.joint_trajectory
        except Exception:
            jt = plan[1].joint_trajectory if isinstance(plan, (list, tuple)) and len(plan) > 1 else None
        if jt is None or len(jt.points) == 0:
            raise RuntimeError("Empty trajectory")
        jt_scaled = self._time_scale(jt)
        self.pub_arm.publish(jt_scaled)
        dur = float(jt_scaled.points[-1].time_from_start.to_sec())
        rospy.sleep(max(dur, 0.3))

    def _time_scale(self, traj: JointTrajectory) -> JointTrajectory:
        scale = float(self.trajectory_time_scale)
        if scale <= 1.0:
            return traj
        out = copy.deepcopy(traj)
        for pt in out.points:
            t = float(pt.time_from_start.to_sec()) * scale
            pt.time_from_start = rospy.Duration(t)
            if pt.velocities:
                pt.velocities = [float(v) / scale for v in pt.velocities]
            if pt.accelerations:
                pt.accelerations = [float(a) / (scale * scale) for a in pt.accelerations]
        return out

    def _run_to_pose(self, msg: PoseStamped) -> None:
        x, y, z, want_pose = self._goal_in_planning_frame(msg)
        rospy.loginfo("nn_goal: target base_link (%.3f, %.3f, %.3f) pose=%s", x, y, z, want_pose)
        plan = self._plan(x, y, z, want_pose)
        try:
            jt = plan.joint_trajectory
        except Exception:
            jt = plan[1].joint_trajectory if isinstance(plan, (list, tuple)) and len(plan) > 1 else None
        if jt is None or len(jt.points) == 0:
            if want_pose and self._quat_down is not None:
                rospy.logwarn("План с ориентацией не найден, пробуем только позицию")
                plan = self._plan(x, y, z, False)
                try:
                    jt = plan.joint_trajectory
                except Exception:
                    jt = None
            if jt is None or len(jt.points) == 0:
                rospy.logwarn("План пустой, цель недостижима или move_group недоступен")
                return
        self._execute_plan(plan)

    def _on_goal(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_goal = msg
        if not self.execute_on_receive:
            return
        now = time.time()
        if now - self._last_exec_time < self.min_interval_sec:
            return
        with self._lock:
            g = self._last_goal
        if g is None:
            return
        self._last_exec_time = now
        try:
            self._run_to_pose(g)
        except Exception as e:
            rospy.logerr("nn_goal execution failed: %s", e)

    def _srv_move(self, _req: Trigger) -> TriggerResponse:
        with self._lock:
            g = self._last_goal
        if g is None:
            return TriggerResponse(success=False, message="No goal received yet")
        try:
            self._run_to_pose(g)
            return TriggerResponse(success=True, message="ok")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    try:
        NnGoalMoveIt().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
