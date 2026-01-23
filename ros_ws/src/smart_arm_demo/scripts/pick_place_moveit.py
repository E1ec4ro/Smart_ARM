#!/usr/bin/env python3
import copy
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Iterator

import rospy
import moveit_commander

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
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


class StepRunner:
    def __init__(self) -> None:
        self._i = 0
        self.current_step = ""

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        self._i += 1
        self.current_step = f"{self._i:02d} {name}"
        t0 = time.time()
        banner = f"========== [STEP {self.current_step}] START =========="
        rospy.loginfo(banner)
        print(banner, flush=True)
        try:
            yield
            dt = time.time() - t0
            msg = f"========== [STEP {self.current_step}] OK (dt={dt:.2f}s) =========="
            rospy.loginfo(msg)
            print(msg, flush=True)
        except Exception as e:
            dt = time.time() - t0
            msg = f"========== [STEP {self.current_step}] FAIL (dt={dt:.2f}s): {e} =========="
            rospy.logerr(msg)
            print(msg, flush=True)
            raise


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

        self.trajectory_time_scale = float(rospy.get_param("~trajectory_time_scale", 2.0))
        self.post_action_pause_s = float(rospy.get_param("~post_action_pause_s", 5.0))
        self.attach_max_distance_m = float(rospy.get_param("~attach_max_distance_m", 0.12))

        self.enable_scene_obstacles = bool(rospy.get_param("~enable_scene_obstacles", True))
        self.scene_table_center_world = rospy.get_param("~scene_table_center_world", [-0.030288, 0.895696, 0.40])
        self.scene_table_size = rospy.get_param("~scene_table_size", [1.20, 0.80, 0.05])
        self.scene_pedestal_center_world = rospy.get_param("~scene_pedestal_center_world", [0.0, 0.0, 0.10])
        self.scene_pedestal_size = rospy.get_param("~scene_pedestal_size", [1.0, 1.0, 0.20])
        self.scene_keepout_center_base = rospy.get_param("~scene_keepout_center_base", [0.0, 0.0, -0.20])
        self.scene_keepout_size = rospy.get_param("~scene_keepout_size", [2.0, 2.0, 0.40])

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

        self.ee_orientation_mode = str(rospy.get_param("~ee_orientation_mode", "home")).strip().lower()
        self._ee_orientation_ref = None

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

        self.group.set_max_velocity_scaling_factor(float(rospy.get_param("~vel_scale", 0.2)))
        self.group.set_max_acceleration_scaling_factor(float(rospy.get_param("~acc_scale", 0.2)))
        self.group.set_goal_position_tolerance(float(rospy.get_param("~pos_tol", 0.01)))
        self.group.set_goal_orientation_tolerance(float(rospy.get_param("~ori_tol", 0.05)))

        self._apply_path_constraints()

        rospy.loginfo(
            f"Pick&Place ready. group={self.group_name}, planning_frame={self.group.get_planning_frame()}, ee={self.group.get_end_effector_link()}"
        )

    def _setup_planning_scene(self) -> None:
        if not self.enable_scene_obstacles:
            return
        try:
            for name in ["scene_table", "scene_pedestal", "scene_keepout"]:
                try:
                    self.scene.remove_world_object(name)
                except Exception:
                    pass

            tx, ty, tz = float(self.scene_table_center_world[0]), float(self.scene_table_center_world[1]), float(self.scene_table_center_world[2])
            sx, sy, sz = float(self.scene_table_size[0]), float(self.scene_table_size[1]), float(self.scene_table_size[2])
            table_ps = PoseStamped()
            table_ps.header.frame_id = self.planning_frame
            table_ps.header.stamp = rospy.Time.now()
            table_ps.pose.position.x = tx - float(self.world_to_base_link_xyz[0])
            table_ps.pose.position.y = ty - float(self.world_to_base_link_xyz[1])
            table_ps.pose.position.z = tz - float(self.world_to_base_link_xyz[2])
            table_ps.pose.orientation.w = 1.0
            self.scene.add_box("scene_table", table_ps, size=(sx, sy, sz))

            px, py, pz = float(self.scene_pedestal_center_world[0]), float(self.scene_pedestal_center_world[1]), float(self.scene_pedestal_center_world[2])
            psx, psy, psz = float(self.scene_pedestal_size[0]), float(self.scene_pedestal_size[1]), float(self.scene_pedestal_size[2])
            ped_ps = PoseStamped()
            ped_ps.header.frame_id = self.planning_frame
            ped_ps.header.stamp = rospy.Time.now()
            ped_ps.pose.position.x = px - float(self.world_to_base_link_xyz[0])
            ped_ps.pose.position.y = py - float(self.world_to_base_link_xyz[1])
            ped_ps.pose.position.z = pz - float(self.world_to_base_link_xyz[2])
            ped_ps.pose.orientation.w = 1.0
            self.scene.add_box("scene_pedestal", ped_ps, size=(psx, psy, psz))

            kx, ky, kz = float(self.scene_keepout_center_base[0]), float(self.scene_keepout_center_base[1]), float(self.scene_keepout_center_base[2])
            ksx, ksy, ksz = float(self.scene_keepout_size[0]), float(self.scene_keepout_size[1]), float(self.scene_keepout_size[2])
            keep_ps = PoseStamped()
            keep_ps.header.frame_id = self.planning_frame
            keep_ps.header.stamp = rospy.Time.now()
            keep_ps.pose.position.x = kx
            keep_ps.pose.position.y = ky
            keep_ps.pose.position.z = kz
            keep_ps.pose.orientation.w = 1.0
            self.scene.add_box("scene_keepout", keep_ps, size=(ksx, ksy, ksz))

            rospy.sleep(0.8)
        except Exception as e:
            rospy.logwarn(f"Failed to set up planning scene obstacles: {e}")

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

        use_pose_target = bool(self._ee_orientation_ref is not None and self.ee_orientation_mode != "none")
        if use_pose_target:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(z)
            p.orientation = self._ee_orientation_ref
            self.group.set_pose_target(p, self.group.get_end_effector_link())
        else:
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
                if use_pose_target:
                    try:
                        self.group.clear_pose_targets()
                        self.group.set_position_target([float(x), float(y), float(z)], self.group.get_end_effector_link())
                        use_pose_target = False
                    except Exception:
                        pass
        return last_plan

    def _move_via_safe_z(self, x: float, y: float, z: float, safe_z: float) -> None:
        safe_z = float(max(safe_z, z))
        self._execute_plan(self._plan_to_position_base(x, y, safe_z))
        if abs(float(z) - safe_z) > 1e-3:
            self._execute_plan(self._plan_to_position_base(x, y, z))

    def _execute_plan(self, plan) -> None:
        try:
            jt = plan.joint_trajectory
        except Exception:
            jt = plan[1].joint_trajectory if isinstance(plan, (list, tuple)) and len(plan) > 1 else None
        if jt is None or len(jt.points) == 0:
            raise RuntimeError("Empty trajectory from MoveIt plan()")

        jt_scaled = self._time_scale_trajectory(jt)
        self._publish_trajectory(jt_scaled)
        dur = float(jt_scaled.points[-1].time_from_start.to_sec())
        rospy.sleep(max(dur, 0.5))
        self._pause("after_motion")

    def _pause(self, reason: str) -> None:
        s = float(self.post_action_pause_s)
        if s <= 0.0:
            return
        rospy.sleep(s)

    def _time_scale_trajectory(self, traj: JointTrajectory) -> JointTrajectory:
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
        steps = StepRunner()

        with steps.step("wait_joint_states"):
            self._wait_for_joint_states()

        if self.go_home_on_start:
            with steps.step("go_home"):
                self._send_home()
                self._pause("after_go_home")
            with steps.step("wait_joint_states_after_home"):
                self._wait_for_joint_states()

        with steps.step("setup_planning_scene"):
            self._setup_planning_scene()

        if self.ee_orientation_mode == "home":
            with steps.step("capture_ee_orientation_home"):
                try:
                    self._ee_orientation_ref = self.group.get_current_pose().pose.orientation
                    rospy.loginfo(
                        "EE orientation ref (home): x=%.3f y=%.3f z=%.3f w=%.3f",
                        self._ee_orientation_ref.x,
                        self._ee_orientation_ref.y,
                        self._ee_orientation_ref.z,
                        self._ee_orientation_ref.w,
                    )
                except Exception as e:
                    rospy.logwarn(f"Failed to capture EE orientation; falling back to position-only: {e}")
                    self._ee_orientation_ref = None

        with steps.step("wait_perception_poses"):
            cube = self._wait_for_pose("cube")
            goal = self._wait_for_pose("goal")

        with steps.step("log_targets"):
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

        with steps.step("gripper_open"):
            self._call_trigger(self.gripper_open_srv)
            self._pause("after_gripper_open")

        pre = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.cfg.approach_height,
        )
        pre_b = self._world_to_base(pre)
        with steps.step("move_pre_grasp"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(pre_b.pose.position.x),
                float(pre_b.pose.position.y),
                float(pre_b.pose.position.z),
                safe_z=max(cur_z, float(pre_b.pose.position.z)),
            )

        grasp = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.grasp_height_offset,
        )
        grasp_b = self._world_to_base(grasp)
        with steps.step("move_grasp"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(grasp_b.pose.position.x),
                float(grasp_b.pose.position.y),
                float(grasp_b.pose.position.z),
                safe_z=max(cur_z, float(pre_b.pose.position.z)),
            )

        with steps.step("gripper_close"):
            self._call_trigger(self.gripper_close_srv)
            self._pause("after_gripper_close")

        with steps.step("attach_cube"):
            ee = self.group.get_current_pose().pose.position
            cube_now = self._latest_cube if self._latest_cube is not None else cube
            cube_now_b = self._world_to_base(cube_now).pose.position
            d = math.sqrt(
                (float(ee.x) - float(cube_now_b.x)) ** 2
                + (float(ee.y) - float(cube_now_b.y)) ** 2
                + (float(ee.z) - float(cube_now_b.z)) ** 2
            )
            if d > float(self.attach_max_distance_m):
                raise RuntimeError(
                    f"Refusing to attach: EE too far from cube (dist={d:.3f}m, max={float(self.attach_max_distance_m):.3f}m)"
                )
            self._call_trigger(self.attach_srv)
            self._pause("after_attach")

        lift = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.cfg.lift_height,
        )
        lift_b = self._world_to_base(lift)
        with steps.step("move_lift"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(lift_b.pose.position.x),
                float(lift_b.pose.position.y),
                float(lift_b.pose.position.z),
                safe_z=max(cur_z, float(lift_b.pose.position.z)),
            )

        pre_place = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + self.cfg.approach_height,
        )
        pre_place_b = self._world_to_base(pre_place)
        with steps.step("move_pre_place"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(pre_place_b.pose.position.x),
                float(pre_place_b.pose.position.y),
                float(pre_place_b.pose.position.z),
                safe_z=max(cur_z, float(pre_place_b.pose.position.z)),
            )

        place = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + (self.cfg.cube_size * 0.5) + self.cfg.place_clearance,
        )
        place_b = self._world_to_base(place)
        with steps.step("move_place"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(place_b.pose.position.x),
                float(place_b.pose.position.y),
                float(place_b.pose.position.z),
                safe_z=max(cur_z, float(pre_place_b.pose.position.z)),
            )

        with steps.step("detach_cube"):
            self._call_trigger(self.detach_srv)
            self._pause("after_detach")

        with steps.step("gripper_open_after_place"):
            self._call_trigger(self.gripper_open_srv)
            self._pause("after_gripper_open_place")

        retreat = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + self.cfg.approach_height,
        )
        retreat_b = self._world_to_base(retreat)
        with steps.step("move_retreat"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(retreat_b.pose.position.x),
                float(retreat_b.pose.position.y),
                float(retreat_b.pose.position.z),
                safe_z=max(cur_z, float(retreat_b.pose.position.z)),
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

