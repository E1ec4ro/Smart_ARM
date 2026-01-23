#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Optional

import rospy
import moveit_commander

from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_srvs.srv import Trigger
from moveit_msgs.msg import RobotState, Constraints, JointConstraint


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


@dataclass
class WorkspaceConfig:
    cube_size: float
    approach_height: float
    lift_height: float
    place_clearance: float


class PickPlaceMoveIt:
    def __init__(self) -> None:
        rospy.init_node("pick_place_moveit", anonymous=False)

        self.group_name = rospy.get_param("~group_name", "manipulator")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.planning_frame = rospy.get_param("~planning_frame", "base_link")
        self.world_to_base_link_xyz = rospy.get_param("~world_to_base_link_xyz", [0.0, 0.0, 0.20])
        if isinstance(self.world_to_base_link_xyz, str):
            s = self.world_to_base_link_xyz.strip()
            s = s[1:-1] if s.startswith("[") and s.endswith("]") else s
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 3:
                self.world_to_base_link_xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            else:
                rospy.logwarn(f"Invalid world_to_base_link_xyz string: {self.world_to_base_link_xyz}, using default")
                self.world_to_base_link_xyz = [0.0, 0.0, 0.20]
        self.go_home_on_start = bool(rospy.get_param("~go_home_on_start", True))
        self.home_duration = float(rospy.get_param("~home_duration", 3.0))
        self.home_joint_positions = rospy.get_param(
            "~home_joint_positions", [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]
        )
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

        self.cube_pose_topic = rospy.get_param("~cube_pose_topic", "/detected_cube_pose_world")
        self.goal_pose_topic = rospy.get_param("~goal_pose_topic", "/detected_goal_pose_world")

        self.arm_command_topic = rospy.get_param("~arm_command_topic", "/arm_controller/command")
        self.joint_states_topic = rospy.get_param("~joint_states_topic", "/joint_states")

        self.gripper_open_srv = rospy.get_param("~gripper_open_srv", "/gripper_control/open")
        self.gripper_close_srv = rospy.get_param("~gripper_close_srv", "/gripper_control/close")

        self.attach_srv = rospy.get_param("~attach_srv", "/gazebo_attach/attach")
        self.detach_srv = rospy.get_param("~detach_srv", "/gazebo_attach/detach")

        self.cfg = WorkspaceConfig(
            cube_size=float(rospy.get_param("~cube_size", rospy.get_param("~target_cube/size", 0.08))),
            approach_height=float(rospy.get_param("~approach_height", 0.15)),
            lift_height=float(rospy.get_param("~lift_height", 0.20)),
            place_clearance=float(rospy.get_param("~place_clearance", 0.01)),
        )
        self.grasp_height_offset = float(rospy.get_param("~grasp_height_offset", 0.0))

        self.lock_wrist = bool(rospy.get_param("~lock_wrist", True))
        self.wrist_2_center = float(rospy.get_param("~wrist_2_center", 1.57))
        self.wrist_3_center = float(rospy.get_param("~wrist_3_center", 0.0))
        self.wrist_tol = float(rospy.get_param("~wrist_tol", 0.35))

        self._latest_cube = None
        self._latest_goal = None
        self._last_joint_state = None

        self.pub_arm = rospy.Publisher(self.arm_command_topic, JointTrajectory, queue_size=1)

        rospy.Subscriber(self.cube_pose_topic, PoseStamped, self._on_cube_pose, queue_size=1)
        rospy.Subscriber(self.goal_pose_topic, PoseStamped, self._on_goal_pose, queue_size=1)
        rospy.Subscriber(self.joint_states_topic, JointState, self._on_joint_state, queue_size=5)

        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
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
                rospy.logwarn(f"Failed to set ee_link={ee_link}; using default: {self.group.get_end_effector_link()}")

        self.group.set_planning_time(float(rospy.get_param("~planning_time", 5.0)))
        self.group.set_num_planning_attempts(int(rospy.get_param("~planning_attempts", 5)))
        self.group.allow_replanning(True)

        self.group.set_max_velocity_scaling_factor(float(rospy.get_param("~vel_scale", 0.4)))
        self.group.set_max_acceleration_scaling_factor(float(rospy.get_param("~acc_scale", 0.4)))
        self.group.set_goal_position_tolerance(float(rospy.get_param("~pos_tol", 0.01)))
        self.group.set_goal_orientation_tolerance(float(rospy.get_param("~ori_tol", 0.05)))

        self._apply_path_constraints()

        rospy.loginfo(
            f"Pick&Place ready. group={self.group_name}, planning_frame={self.group.get_planning_frame()}, ee={self.group.get_end_effector_link()}"
        )

    def _apply_path_constraints(self) -> None:
        try:
            if not self.lock_wrist:
                self.group.clear_path_constraints()
                return
            c = Constraints()
            jc2 = JointConstraint()
            jc2.joint_name = "wrist_2_joint"
            jc2.position = self.wrist_2_center
            jc2.tolerance_above = self.wrist_tol
            jc2.tolerance_below = self.wrist_tol
            jc2.weight = 1.0

            jc3 = JointConstraint()
            jc3.joint_name = "wrist_3_joint"
            jc3.position = self.wrist_3_center
            jc3.tolerance_above = self.wrist_tol
            jc3.tolerance_below = self.wrist_tol
            jc3.weight = 1.0

            c.joint_constraints = [jc2, jc3]
            self.group.set_path_constraints(c)
        except Exception as e:
            rospy.logwarn(f"Failed to apply wrist constraints: {e}")

    def _on_cube_pose(self, msg: PoseStamped) -> None:
        self._latest_cube = msg

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        self._latest_goal = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self._last_joint_state = msg

    def _wait_for_joint_states(self, timeout_s: float = 20.0) -> None:
        t0 = time.time()
        last_stamp = None
        while not rospy.is_shutdown():
            js = self._last_joint_state
            if js is not None:
                st = js.header.stamp
                if last_stamp is None:
                    last_stamp = st
                elif st > last_stamp:
                    return
                if st.to_sec() == 0.0 and all(j in js.name for j in self.joint_names):
                    return
            if time.time() - t0 > timeout_s:
                rospy.logwarn("Timeout waiting for fresh /joint_states; continuing anyway.")
                return
            rospy.sleep(0.1)

    def _send_home(self) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in self.home_joint_positions]
        pt.time_from_start = rospy.Duration(self.home_duration)
        msg.points = [pt]
        self.pub_arm.publish(msg)
        rospy.sleep(max(self.home_duration, 0.5))

    def _set_moveit_start_state_home(self) -> None:
        rs = RobotState()
        rs.joint_state.name = list(self.joint_names)
        rs.joint_state.position = [float(x) for x in self.home_joint_positions]
        try:
            self.group.set_start_state(rs)
        except Exception:
            try:
                self.group.set_start_state_to_current_state()
            except Exception:
                pass

    def _wait_for_pose(self, which: str, timeout_s: float = 30.0) -> PoseStamped:
        t0 = time.time()
        while not rospy.is_shutdown():
            if which == "cube" and self._latest_cube is not None:
                return self._latest_cube
            if which == "goal" and self._latest_goal is not None:
                return self._latest_goal
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Timeout waiting for {which} pose")
            rospy.sleep(0.1)
        raise RuntimeError("ROS shutdown")

    def _call_trigger(self, service_name: str, timeout_s: float = 10.0) -> None:
        rospy.wait_for_service(service_name, timeout=timeout_s)
        srv = rospy.ServiceProxy(service_name, Trigger)
        resp = srv()
        if not resp.success:
            raise RuntimeError(f"Service {service_name} failed: {resp.message}")

    def _publish_trajectory(self, traj: JointTrajectory) -> None:
        self.pub_arm.publish(traj)

    def _world_to_base(self, ps: PoseStamped) -> PoseStamped:
        dx, dy, dz = float(self.world_to_base_link_xyz[0]), float(self.world_to_base_link_xyz[1]), float(self.world_to_base_link_xyz[2])
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.planning_frame
        out.pose.position.x = float(ps.pose.position.x - dx)
        out.pose.position.y = float(ps.pose.position.y - dy)
        out.pose.position.z = float(ps.pose.position.z - dz)
        out.pose.orientation.w = 1.0
        return out

    def _plan_to_position_base(self, x: float, y: float, z: float):
        self.group.clear_pose_targets()
        try:
            self.group.set_start_state_to_current_state()
        except Exception:
            self._set_moveit_start_state_home()

        self.group.set_position_target([float(x), float(y), float(z)], self.group.get_end_effector_link())
        retries = int(rospy.get_param("~plan_retries", 3))
        last_plan = None
        for attempt in range(max(retries, 1)):
            last_plan = self.group.plan()
            try:
                jt = last_plan.joint_trajectory
            except Exception:
                jt = last_plan[1].joint_trajectory if isinstance(last_plan, (list, tuple)) and len(last_plan) > 1 else None
            if jt is not None and len(jt.points) > 0:
                return last_plan
            rospy.sleep(0.2)

            if attempt == 0:
                self._set_moveit_start_state_home()
        return last_plan

    def _execute_plan(self, plan) -> None:
        try:
            jt = plan.joint_trajectory
        except Exception:
            jt = plan[1].joint_trajectory if isinstance(plan, (list, tuple)) and len(plan) > 1 else None
        if jt is None or len(jt.points) == 0:
            raise RuntimeError("Empty trajectory from MoveIt plan()")

        self._publish_trajectory(jt)
        dur = float(jt.points[-1].time_from_start.to_sec())
        rospy.sleep(max(dur, 0.5))

    def _make_pose(self, x: float, y: float, z: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.w = 1.0
        return ps

    def run_once(self) -> None:
        self._wait_for_joint_states()
        if self.go_home_on_start:
            rospy.loginfo("Moving to home joint pose.")
            self._send_home()
            self._wait_for_joint_states()

        rospy.loginfo("Waiting for perception poses...")
        cube = self._wait_for_pose("cube")
        goal = self._wait_for_pose("goal")
        cube_b = self._world_to_base(cube)
        goal_b = self._world_to_base(goal)
        rospy.loginfo(
            "Targets in base_link: cube(%.3f, %.3f, %.3f) goal(%.3f, %.3f, %.3f)",
            cube_b.pose.position.x,
            cube_b.pose.position.y,
            cube_b.pose.position.z,
            goal_b.pose.position.x,
            goal_b.pose.position.y,
            goal_b.pose.position.z,
        )

        rospy.loginfo("Opening gripper.")
        self._call_trigger(self.gripper_open_srv)

        pre = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.cfg.approach_height,
        )
        rospy.loginfo("Planning to pre-grasp.")
        pre_b = self._world_to_base(pre)
        self._execute_plan(self._plan_to_position_base(pre_b.pose.position.x, pre_b.pose.position.y, pre_b.pose.position.z))

        grasp = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.grasp_height_offset,
        )
        rospy.loginfo("Planning to grasp.")
        grasp_b = self._world_to_base(grasp)
        self._execute_plan(self._plan_to_position_base(grasp_b.pose.position.x, grasp_b.pose.position.y, grasp_b.pose.position.z))

        rospy.loginfo("Closing gripper.")
        self._call_trigger(self.gripper_close_srv)

        rospy.loginfo("Attaching cube to gripper.")
        self._call_trigger(self.attach_srv)

        lift = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.cfg.lift_height,
        )
        rospy.loginfo("Planning to lift.")
        lift_b = self._world_to_base(lift)
        self._execute_plan(self._plan_to_position_base(lift_b.pose.position.x, lift_b.pose.position.y, lift_b.pose.position.z))

        pre_place = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + self.cfg.approach_height,
        )
        rospy.loginfo("Planning to pre-place.")
        pre_place_b = self._world_to_base(pre_place)
        self._execute_plan(
            self._plan_to_position_base(pre_place_b.pose.position.x, pre_place_b.pose.position.y, pre_place_b.pose.position.z)
        )

        place = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + (self.cfg.cube_size * 0.5) + self.cfg.place_clearance,
        )
        rospy.loginfo("Planning to place.")
        place_b = self._world_to_base(place)
        self._execute_plan(self._plan_to_position_base(place_b.pose.position.x, place_b.pose.position.y, place_b.pose.position.z))

        rospy.loginfo("Detaching cube.")
        self._call_trigger(self.detach_srv)

        rospy.loginfo("Opening gripper.")
        self._call_trigger(self.gripper_open_srv)

        retreat = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + self.cfg.approach_height,
        )
        rospy.loginfo("Planning retreat.")
        retreat_b = self._world_to_base(retreat)
        self._execute_plan(
            self._plan_to_position_base(retreat_b.pose.position.x, retreat_b.pose.position.y, retreat_b.pose.position.z)
        )

        rospy.loginfo("Pick&place completed.")

    def spin(self) -> None:
        rospy.sleep(1.0)
        run_once = bool(rospy.get_param("~run_once", True))
        if run_once:
            self.run_once()
            rospy.spin()
        else:
            rate = rospy.Rate(0.1)
            while not rospy.is_shutdown():
                try:
                    self.run_once()
                except Exception as e:
                    rospy.logerr(f"pick&place failed: {e}")
                rate.sleep()


def main() -> None:
    try:
        PickPlaceMoveIt().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

