#!/usr/bin/env python3
import copy
import math
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple

import rospy
import moveit_commander

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from grasp_workspace import (
    AttachLimits,
    load_grasp_workspace_config,
    param_bool,
    validate_virtual_attach,
)

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_srvs.srv import Trigger, TriggerResponse
from moveit_msgs.msg import RobotState, Constraints, JointConstraint


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """RPY в радианах (base_link) → quaternion для set_pose_target."""
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
        self.go_home_on_start = param_bool("~go_home_on_start", True)
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
        self.gripper_close_before_attach = param_bool("~gripper_close_before_attach", True)

        self.attach_srv = rospy.get_param("~attach_srv", "/gazebo_attach/attach")
        self.detach_srv = rospy.get_param("~detach_srv", "/gazebo_attach/detach")

        self.trajectory_time_scale = float(rospy.get_param("~trajectory_time_scale", 2.0))
        self.post_action_pause_s = float(rospy.get_param("~post_action_pause_s", 5.0))
        self.attach_max_distance_m = float(rospy.get_param("~attach_max_distance_m", 0.12))
        # Gazebo /attach телепортирует куб к захвату; отказ, если EE слишком высоко над центром куба (base_link).
        self.attach_max_dz_above_cube_m = float(rospy.get_param("~attach_max_dz_above_cube_m", 0.14))
        self.attach_limits = AttachLimits(
            max_distance_m=self.attach_max_distance_m,
            max_dz_above_cube_center_m=self.attach_max_dz_above_cube_m,
        )

        self.enable_scene_obstacles = param_bool("~enable_scene_obstacles", True)
        self.scene_table_center_world = rospy.get_param("~scene_table_center_world", [-0.030288, 0.895696, 0.40])
        self.scene_table_size = rospy.get_param("~scene_table_size", [1.20, 0.80, 0.05])
        self.scene_pedestal_center_world = rospy.get_param("~scene_pedestal_center_world", [0.0, 0.0, 0.10])
        self.scene_pedestal_size = rospy.get_param("~scene_pedestal_size", [1.0, 1.0, 0.20])
        self.scene_keepout_center_base = rospy.get_param("~scene_keepout_center_base", [0.0, 0.0, -0.20])
        self.scene_keepout_size = rospy.get_param("~scene_keepout_size", [2.0, 2.0, 0.40])

        self.cfg = load_grasp_workspace_config()
        self.grasp_height_offset = self.cfg.grasp_height_offset

        self.lock_wrist = param_bool("~lock_wrist", True)
        self.wrist_2_center = float(rospy.get_param("~wrist_2_center", 1.57))
        self.wrist_3_center = float(rospy.get_param("~wrist_3_center", 0.0))
        self.wrist_tol = float(rospy.get_param("~wrist_tol", 0.35))

        self.ee_orientation_mode = str(rospy.get_param("~ee_orientation_mode", "home")).strip().lower()
        self._ee_orientation_ref = None
        self._quat_down = None

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
        self.group.allow_replanning(param_bool("~allow_replanning", False))

        self.group.set_max_velocity_scaling_factor(float(rospy.get_param("~vel_scale", 0.2)))
        self.group.set_max_acceleration_scaling_factor(float(rospy.get_param("~acc_scale", 0.2)))
        self.group.set_goal_position_tolerance(float(rospy.get_param("~pos_tol", 0.01)))
        self.group.set_goal_orientation_tolerance(float(rospy.get_param("~ori_tol", 0.05)))

        planner_id = str(rospy.get_param("~planner_id", "")).strip()
        if planner_id:
            try:
                self.group.set_planner_id(planner_id)
                rospy.loginfo("MoveGroup planner_id=%s", planner_id)
            except Exception as e:
                rospy.logwarn("set_planner_id(%s) failed: %s — используется планировщик по умолчанию", planner_id, e)

        # down: кватернион «схват вниз»; применяется не ко всему пути — см. down_orientation_for_grasp_only.
        if self.ee_orientation_mode == "down":
            rpy = rospy.get_param("~ee_down_rpy", [0.0, math.pi, 0.0])
            self._quat_down = quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            self._ee_orientation_ref = self._quat_down
            rospy.loginfo(
                "EE orientation mode=down rpy=[%.4f, %.4f, %.4f]; дальние ходы — position-only, down только на снижении",
                float(rpy[0]),
                float(rpy[1]),
                float(rpy[2]),
            )

        self._apply_path_constraints()

        self._run_lock = threading.Lock()
        self._place_descent_active = False
        self._pick_carry_active = False
        self._grasp_descent_active = False
        self._post_attach_lift_active = False
        self._move_pre_place_active = False
        self._cube_attached_in_cycle = False
        self._place_skip_vertical_polyline_once = False

        rospy.Service("~run_pick_place", Trigger, self._cb_run_pick_place)

        rospy.loginfo(
            "Pick&Place ready. group=%s, planning_frame=%s, ee=%s, gripper_close_before_attach=%s. "
            "Повтор цикла: rosservice call .../pick_place_moveit/run_pick_place",
            self.group_name,
            self.group.get_planning_frame(),
            self.group.get_end_effector_link(),
            self.gripper_close_before_attach,
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

            pad = float(rospy.get_param("~scene_collision_padding_m", 0.0))
            tx, ty, tz = float(self.scene_table_center_world[0]), float(self.scene_table_center_world[1]), float(self.scene_table_center_world[2])
            sx, sy, sz = float(self.scene_table_size[0]), float(self.scene_table_size[1]), float(self.scene_table_size[2])
            sx, sy, sz = sx + 2.0 * pad, sy + 2.0 * pad, sz + 2.0 * pad
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
            psx, psy, psz = psx + 2.0 * pad, psy + 2.0 * pad, psz + 2.0 * pad
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
            rospy.loginfo(
                "Planning scene obstacles: table top z≈%.3f m in %s, min horizontal EE z≈%.3f m",
                self._table_top_z_in_planning_frame(),
                self.planning_frame,
                self._min_safe_z_horizontal(),
            )
        except Exception as e:
            rospy.logwarn(f"Failed to set up planning scene obstacles: {e}")

    def _apply_path_constraints(self) -> None:
        """
        Ограничения пути для OMPL/cartesian:
        - lock_wrist: фиксация запястий около центра;
        - hold_wrist_near_current_carry: опционально на переносе; при hold_wrist_exempt_move_pre_place на move_pre_place
          отключено (длинный перенос к площадке + коридор запястий даёт пустой план).
        - limit_joint_deviation: коридор вокруг текущих shoulder_pan/lift/elbow/wrist_1 — меньше лишних разворотов OMPL.
          place_phase_only=true: коридор только при снижении укладки (pick из дома без узкого коридора).
          place_phase_only=false: на всём цикле — *global* допуски (широкие, IK из дома проходит); при укладке — узкие *_rad.
          grasp_descent: при limit_joint_deviation_exempt_grasp_descent коридор отключается на pre→grasp (иначе часто не хватает
          сгиба плеча/локтя до grasp z, EE остаётся «над» кубом и attach отклоняется).
        """
        try:
            hold_carry = (
                param_bool("~hold_wrist_near_current_carry", True)
                and self._pick_carry_active
                and not self._place_descent_active
                and not (
                    self._move_pre_place_active
                    and param_bool("~hold_wrist_exempt_move_pre_place", True)
                )
            )
            carry_overrides_lock = hold_carry and param_bool("~carry_overrides_lock_wrist", True)
            use_lock_wrist = bool(self.lock_wrist) and not carry_overrides_lock

            limit_dev = param_bool("~limit_joint_deviation", True)
            place_only = param_bool("~limit_joint_deviation_place_phase_only", True)
            if limit_dev and place_only and not self._place_descent_active:
                limit_dev = False
            if (
                limit_dev
                and self._grasp_descent_active
                and param_bool("~limit_joint_deviation_exempt_grasp_descent", True)
            ):
                limit_dev = False
            if (
                limit_dev
                and self._post_attach_lift_active
                and param_bool("~limit_joint_deviation_exempt_post_attach_lift", True)
            ):
                limit_dev = False
            if (
                limit_dev
                and self._move_pre_place_active
                and param_bool("~limit_joint_deviation_exempt_move_pre_place", True)
            ):
                limit_dev = False
            if not use_lock_wrist and not limit_dev and not hold_carry:
                self.group.clear_path_constraints()
                return
            c = Constraints()
            c.joint_constraints = []
            if use_lock_wrist:
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
                c.joint_constraints.extend([jc2, jc3])
            if hold_carry:
                try:
                    qh = list(self.group.get_current_joint_values())
                    nh = list(self.group.get_active_joints())
                except Exception:
                    qh, nh = [], []
                if len(qh) == len(nh) and nh:
                    tw1 = float(rospy.get_param("~carry_wrist_1_tol_rad", 0.42))
                    tw2 = float(rospy.get_param("~carry_wrist_2_tol_rad", 0.48))
                    tw3 = float(rospy.get_param("~carry_wrist_3_tol_rad", 0.52))
                    wt = float(rospy.get_param("~carry_wrist_constraint_weight", 1.0))
                    for jn, tol in (
                        ("wrist_1_joint", tw1),
                        ("wrist_2_joint", tw2),
                        ("wrist_3_joint", tw3),
                    ):
                        if jn in nh:
                            jc = JointConstraint()
                            jc.joint_name = jn
                            jc.position = float(qh[nh.index(jn)])
                            jc.tolerance_above = tol
                            jc.tolerance_below = tol
                            jc.weight = wt
                            c.joint_constraints.append(jc)
            if limit_dev:
                try:
                    q = list(self.group.get_current_joint_values())
                    names = list(self.group.get_active_joints())
                except Exception:
                    q, names = [], []
                if len(q) >= 3 and len(names) == len(q) and sum(abs(float(x)) for x in q) > 1e-3:
                    if self._place_descent_active:
                        pan_tol = float(rospy.get_param("~limit_shoulder_pan_rad", 0.52))
                        lift_tol = float(rospy.get_param("~limit_shoulder_lift_rad", 0.72))
                        elbow_tol = float(rospy.get_param("~limit_elbow_rad", 0.72))
                        w1_tol = float(rospy.get_param("~limit_wrist_1_rad", 0.85))
                    else:
                        pan_tol = float(rospy.get_param("~limit_shoulder_pan_global_rad", 1.55))
                        lift_tol = float(rospy.get_param("~limit_shoulder_lift_global_rad", 1.25))
                        elbow_tol = float(rospy.get_param("~limit_elbow_global_rad", 1.25))
                        w1_tol = float(rospy.get_param("~limit_wrist_1_global_rad", 1.4))
                    arm_pairs = (
                        ("shoulder_pan_joint", pan_tol),
                        ("shoulder_lift_joint", lift_tol),
                        ("elbow_joint", elbow_tol),
                        ("wrist_1_joint", w1_tol),
                    )
                    for jn, tol in arm_pairs:
                        if hold_carry and jn == "wrist_1_joint":
                            continue
                        if jn in names:
                            jc = JointConstraint()
                            jc.joint_name = jn
                            jc.position = float(q[names.index(jn)])
                            jc.tolerance_above = tol
                            jc.tolerance_below = tol
                            jc.weight = float(rospy.get_param("~limit_joint_constraint_weight", 1.0))
                            c.joint_constraints.append(jc)
            if not c.joint_constraints:
                self.group.clear_path_constraints()
                return
            self.group.set_path_constraints(c)
        except Exception as e:
            rospy.logwarn(f"Failed to apply path constraints: {e}")

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

    def _cb_run_pick_place(self, _req) -> TriggerResponse:
        """Повторить полный цикл pick&place (ожидание поз, pick, place, при необходимости домой)."""
        with self._run_lock:
            try:
                self.run_once()
                return TriggerResponse(True, "pick&place completed")
            except Exception as e:
                rospy.logerr("run_pick_place: %s", e)
                rospy.logerr("%s", traceback.format_exc())
                return TriggerResponse(False, str(e))

    def _publish_trajectory(self, traj: JointTrajectory) -> None:
        self.pub_arm.publish(traj)

    def _table_top_z_in_planning_frame(self) -> float:
        """Верх столешницы в planning_frame (base_link), из scene_table_* и world_to_base_link_xyz."""
        tc = self.scene_table_center_world
        sz = self.scene_table_size
        top_z_world = float(tc[2]) + float(sz[2]) * 0.5
        return top_z_world - float(self.world_to_base_link_xyz[2])

    def _min_safe_z_horizontal(self) -> float:
        """Минимальный z ЗХ для горизонтального этапа (x,y,safe_z), чтобы не планировать сквозь стол."""
        if not param_bool("~enforce_min_safe_z_above_table", True):
            return -1e9
        margin = float(rospy.get_param("~safe_z_clearance_above_table_m", 0.10))
        extra = float(rospy.get_param("~min_safe_z_extra_m", 0.0))
        return self._table_top_z_in_planning_frame() + margin + extra

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

    def _clamp_pose_base_z(self, ps: PoseStamped, label: str) -> PoseStamped:
        """
        Ограничить z в base_link для фазы place: слишком высокая цель = пустой план MoveIt, робот «замирает».
        """
        out = copy.deepcopy(ps)
        z = float(out.pose.position.z)
        zmax = float(rospy.get_param("~place_max_z_base", 0.92))
        zmin = float(rospy.get_param("~place_min_z_base", 0.0))
        if z > zmax:
            rospy.logwarn(
                "%s: z=%.3f в %s > place_max_z_base=%.3f — снижаем (досягаемость/рациональный подход)",
                label,
                z,
                self.planning_frame,
                zmax,
            )
            out.pose.position.z = zmax
        if zmin > 1e-6 and z < zmin:
            rospy.logwarn("%s: z=%.3f < place_min_z_base=%.3f — поднимаем", label, z, zmin)
            out.pose.position.z = zmin
        return out

    def _ensure_place_below_pre(self, pre_b: PoseStamped, place_b: PoseStamped) -> PoseStamped:
        """
        Если оба z обрезаны place_max_z_base, pre и place могут совпасть — тогда _move_via_safe_z не даёт
        второго сегмента вниз и робот «висит» над целью. Принудительно: place_z < pre_z.
        """
        out = copy.deepcopy(place_b)
        pre_z = float(pre_b.pose.position.z)
        pl_z = float(out.pose.position.z)
        min_below = float(rospy.get_param("~place_min_below_pre_m", 0.06))
        eps = 1e-4
        if pl_z < pre_z - eps:
            return out
        new_z = pre_z - min_below
        zfloor = float(rospy.get_param("~place_min_z_base", 0.0))
        if zfloor > 1e-6:
            new_z = max(new_z, zfloor)
        else:
            new_z = max(new_z, self._table_top_z_in_planning_frame() + 0.02)
        rospy.logwarn(
            "place z=%.4f не ниже pre z=%.4f (часто общий place_max_z_base) — place_z → %.4f для снижения к укладке",
            pl_z,
            pre_z,
            new_z,
        )
        out.pose.position.z = new_z
        return out

    def _effective_place_target_z(self, z_place: float, pre_z: float) -> float:
        """
        Сжимает цель z укладки: завышенная vision/clamp (place_max_z_base) даёт z_final выше EE —
        тогда cz <= z_final+eps срабатывает сразу и снижение не выполняется вообще.
        Потолок: верх столешницы в planning_frame + place_ee_max_above_table_m; пол: table + зазор.
        """
        table_top = self._table_top_z_in_planning_frame()
        margin = float(rospy.get_param("~place_ee_max_above_table_m", 0.42))
        min_below = float(rospy.get_param("~place_min_below_pre_m", 0.06))
        z_floor = float(rospy.get_param("~place_min_z_base", 0.0))
        min_above = float(rospy.get_param("~place_min_height_above_table_m", 0.028))
        z_floor_eff = max(z_floor, table_top + min_above)
        z_cap = table_top + margin
        z = float(z_place)
        z = min(z, pre_z - min_below, z_cap)
        z = max(z, z_floor_eff)
        z = min(z, pre_z - min_below)
        if abs(z - z_place) > 1e-4:
            rospy.logwarn(
                "place target z: %.4f → %.4f (table_top=%.4f z_cap=%.4f pre_z=%.4f)",
                z_place,
                z,
                table_top,
                z_cap,
                pre_z,
            )
        return z

    def _thin_joint_trajectory(self, jt: JointTrajectory) -> JointTrajectory:
        """Декартов путь может дать сотни точек → долгое ожидание. OMPL+TOPT обычно <100 точек — не трогаем."""
        if not param_bool("~thin_long_trajectories", True):
            return jt
        max_pts = int(rospy.get_param("~max_trajectory_points", 120))
        if len(jt.points) <= max_pts:
            return jt
        out = copy.deepcopy(jt)
        n = len(jt.points)
        if max_pts < 2:
            out.points = [jt.points[-1]]
        else:
            indices = [int(round(i * (n - 1) / (max_pts - 1))) for i in range(max_pts)]
            out.points = [jt.points[i] for i in indices]
        rospy.loginfo("joint trajectory thinned: %d -> %d points", n, len(out.points))
        return out

    def _stamp_uniform_joint_times(self, jt: JointTrajectory) -> JointTrajectory:
        """Cartesian-путь иногда приходит с нулевыми time_from_start — задаём равномерный шаг."""
        if not jt.points:
            return jt
        try:
            last_t = float(jt.points[-1].time_from_start.to_sec())
        except Exception:
            last_t = 0.0
        if last_t > 1e-4:
            return jt
        n = len(jt.points)
        dt = float(rospy.get_param("~cartesian_point_dt_sec", 0.04))
        if n > int(rospy.get_param("~trajectory_points_coarse_threshold", 64)):
            dt = float(rospy.get_param("~cartesian_point_dt_coarse_sec", 0.08))
        out = copy.deepcopy(jt)
        for i, pt in enumerate(out.points):
            pt.time_from_start = rospy.Duration(float((i + 1) * dt))
        return out

    def _make_cartesian_pose(self, x: float, y: float, z: float, ori: str, ee_ref: Pose) -> Pose:
        """Ориентация ЗХ для декартова пути (ori как в _plan_segment / safe_z)."""
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = float(z)
        if ori == "none":
            p.orientation = copy.deepcopy(ee_ref.orientation)
        elif ori == "down" and self._quat_down is not None:
            p.orientation = copy.deepcopy(self._quat_down)
        elif ori == "home" or (
            ori == "auto" and self.ee_orientation_mode == "home" and self._ee_orientation_ref is not None
        ):
            p.orientation = copy.deepcopy(self._ee_orientation_ref)
        else:
            p.orientation = copy.deepcopy(ee_ref.orientation)
        return p

    def _try_cartesian_straight(self, x: float, y: float, z: float, ori: str) -> tuple:
        """
        Прямая в декартовом пространстве (позиция ЗХ линейно от текущей к цели).
        Возвращает (plan или None, успех).
        """
        try:
            self.group.clear_pose_targets()
            try:
                self.group.set_start_state_to_current_state()
            except Exception:
                self._set_moveit_start_state_home()
            self._apply_path_constraints()
            cp = self.group.get_current_pose()
            end = self._make_cartesian_pose(x, y, z, ori, cp.pose)

            eef_step = float(rospy.get_param("~cartesian_eef_step_m", 0.005))
            jump = float(rospy.get_param("~cartesian_jump_threshold", 2.0))
            avoid = param_bool("~cartesian_avoid_collisions", True)
            if self._place_descent_active and param_bool("~place_cartesian_allow_collisions", False):
                avoid = False
            waypoints = [end]
            plan, fraction = self.group.compute_cartesian_path(waypoints, eef_step, jump, avoid_collisions=avoid)
            if self._place_descent_active:
                min_frac = float(rospy.get_param("~place_cartesian_min_fraction", 0.62))
            else:
                min_frac = float(rospy.get_param("~cartesian_min_fraction", 0.82))
            if fraction + 1e-6 < min_frac:
                rospy.logwarn("cartesian: fraction=%.3f < %.3f", fraction, min_frac)
                return None, False
            try:
                jt = plan.joint_trajectory
            except Exception:
                jt = None
            if jt is None or len(jt.points) == 0:
                return None, False
            rospy.loginfo("cartesian path OK: fraction=%.3f points=%d", fraction, len(jt.points))
            return plan, True
        except Exception as e:
            rospy.logwarn("compute_cartesian_path failed: %s", e)
            return None, False

    def _try_cartesian_polyline(self, segments: List[Tuple[float, float, float, str]]) -> tuple:
        """
        Одна декартова траектория через несколько waypoints (ломаная в ЗХ — короче, чем два отдельных плана).
        segments: [(x,y,z, ori), ...] от текущего положения через каждую точку по порядку.
        """
        if not segments:
            return None, False
        try:
            self.group.clear_pose_targets()
            try:
                self.group.set_start_state_to_current_state()
            except Exception:
                self._set_moveit_start_state_home()
            self._apply_path_constraints()
            cp = self.group.get_current_pose()
            ee_ref = cp.pose
            waypoints = [self._make_cartesian_pose(x, y, z, ori, ee_ref) for x, y, z, ori in segments]
            eef_step = float(rospy.get_param("~cartesian_eef_step_m", 0.005))
            jump = float(rospy.get_param("~cartesian_jump_threshold", 2.0))
            avoid = param_bool("~cartesian_avoid_collisions", True)
            if self._place_descent_active and param_bool("~place_cartesian_allow_collisions", False):
                avoid = False
            plan, fraction = self.group.compute_cartesian_path(waypoints, eef_step, jump, avoid_collisions=avoid)
            _min_cart = float(rospy.get_param("~cartesian_min_fraction", 0.75))
            if self._place_descent_active:
                min_frac = float(rospy.get_param("~place_cartesian_polyline_min_fraction", 0.52))
            else:
                min_frac = float(rospy.get_param("~cartesian_polyline_min_fraction", _min_cart))
            if fraction + 1e-6 < min_frac:
                rospy.logwarn("cartesian polyline: fraction=%.3f < %.3f waypoints=%d", fraction, min_frac, len(waypoints))
                return None, False
            try:
                jt = plan.joint_trajectory
            except Exception:
                jt = None
            if jt is None or len(jt.points) == 0:
                return None, False
            rospy.loginfo("cartesian polyline OK: fraction=%.3f points=%d waypoints=%d", fraction, len(jt.points), len(waypoints))
            return plan, True
        except Exception as e:
            rospy.logwarn("compute_cartesian_path (polyline) failed: %s", e)
            return None, False

    def _plan_segment(self, x: float, y: float, z: float, ori: str = "auto", method_override: Optional[str] = None):
        """
        План одного участка: по умолчанию декартова прямая + при неудаче OMPL.
        ~planning_method: cartesian | cartesian_prefer | ompl | cartesian_only
        method_override: если задан, перекрывает ~planning_method (для move_lift — ompl).
        """
        if method_override is not None and str(method_override).strip():
            method = str(method_override).strip().lower()
        else:
            method = str(rospy.get_param("~planning_method", "cartesian")).strip().lower()
        if method == "ompl":
            return self._plan_to_position_base(x, y, z, ori=ori)
        if method == "cartesian_only":
            plan, ok = self._try_cartesian_straight(x, y, z, ori)
            if ok and plan is not None:
                return plan
            raise RuntimeError("cartesian_only: декартов путь не построен (fraction/IK)")
        if method in ("cartesian", "linear", "cartesian_prefer"):
            plan, ok = self._try_cartesian_straight(x, y, z, ori)
            if ok and plan is not None:
                return plan
            rospy.logwarn("переход на OMPL для точки (%.3f, %.3f, %.3f) ori=%s", x, y, z, ori)
        return self._plan_to_position_base(x, y, z, ori=ori)

    def _plan_to_position_base(self, x: float, y: float, z: float, ori: str = "auto"):
        """
        ori:
          auto — для mode=down по умолчанию только позиция (достижимость); home — pose если есть ref;
          none — только позиция; down — pose с _quat_down (узкий вертикальный участок).
        """
        retries = int(rospy.get_param("~plan_retries", 2))
        last_plan = None
        if ori == "none":
            use_pose_effective = False
        elif ori == "down":
            use_pose_effective = self._quat_down is not None
        elif ori == "auto":
            if self.ee_orientation_mode == "down":
                use_pose_effective = False
            elif self.ee_orientation_mode == "none":
                use_pose_effective = False
            elif self.ee_orientation_mode == "home":
                use_pose_effective = self._ee_orientation_ref is not None
            else:
                use_pose_effective = False
        else:
            use_pose_effective = False

        for attempt in range(max(retries, 1)):
            self.group.clear_pose_targets()
            if attempt > 0:
                self.group.clear_path_constraints()
            try:
                self.group.set_start_state_to_current_state()
            except Exception:
                self._set_moveit_start_state_home()
            self._apply_path_constraints()

            if use_pose_effective:
                p = Pose()
                p.position.x = float(x)
                p.position.y = float(y)
                p.position.z = float(z)
                p.orientation = self._quat_down if ori == "down" else self._ee_orientation_ref
                self.group.set_pose_target(p, self.group.get_end_effector_link())
            else:
                self.group.set_position_target([float(x), float(y), float(z)], self.group.get_end_effector_link())

            last_plan = self.group.plan()
            try:
                jt = last_plan.joint_trajectory
            except Exception:
                jt = last_plan[1].joint_trajectory if isinstance(last_plan, (list, tuple)) and len(last_plan) > 1 else None
            if jt is not None and len(jt.points) > 0:
                return last_plan
            rospy.sleep(0.2)
            if attempt == 0 and use_pose_effective and ori == "auto" and self.ee_orientation_mode == "home":
                use_pose_effective = False

        if ori == "down" and self._quat_down is not None:
            try:
                jt = last_plan.joint_trajectory
            except Exception:
                jt = last_plan[1].joint_trajectory if isinstance(last_plan, (list, tuple)) and len(last_plan) > 1 else None
            if jt is None or len(jt.points) == 0:
                rospy.logwarn("План с ориентацией down не найден — пробуем только позицию для этой точки")
                return self._plan_to_position_base(x, y, z, "none")
        return last_plan

    def _move_via_safe_z(
        self,
        x: float,
        y: float,
        z: float,
        safe_z: float,
        planning_method_override: Optional[str] = None,
    ) -> None:
        """Два этапа: (x,y,safe_z) затем (x,y,z). При mode=down: дальние точки — position-only, down только на снижении (z < safe_z)."""
        safe_z = float(max(safe_z, z))
        fz = float(z)
        fs = float(safe_z)
        min_fs = self._min_safe_z_horizontal()
        if fs < min_fs:
            rospy.loginfo(
                "safe_z %.3f → %.3f (min над столом в %s; иначе OMPL может «протолкнуть» сквозь стол в Gazebo)",
                fs,
                min_fs,
                self.planning_frame,
            )
            fs = min_fs
        grasp_only = param_bool("~down_orientation_for_grasp_only", True)
        if self.ee_orientation_mode == "down" and grasp_only:
            ori1 = "none"
            ori2 = "none" if abs(fz - fs) <= 1e-3 else ("down" if fz < fs - 1e-6 else "none")
        else:
            ori1 = "auto"
            ori2 = "auto"
        if abs(fz - fs) <= 1e-3:
            self._execute_plan(self._plan_segment(x, y, fs, ori=ori1, method_override=planning_method_override))
            return
        method_eff = (planning_method_override or "").strip().lower()
        if not method_eff:
            method_eff = str(rospy.get_param("~planning_method", "cartesian")).strip().lower()
        if param_bool("~cartesian_safe_z_single_plan", True) and method_eff in ("cartesian", "linear", "cartesian_prefer"):
            # Снижение к кубу: (none→down) заставляет слеpить ориентацию по всему отрезку — крутятся запястья.
            # Сначала пробуем две точки с постоянным down (вертикаль при уже выставленном наклоне схвата).
            if (
                param_bool("~pick_vertical_constant_down_first", True)
                and grasp_only
                and self.ee_orientation_mode == "down"
                and self._quat_down is not None
                and fz < fs - 1e-6
                and ori2 == "down"
            ):
                plan_d, ok_d = self._try_cartesian_polyline([(x, y, fs, "down"), (x, y, fz, "down")])
                if ok_d and plan_d is not None:
                    rospy.loginfo(
                        "pick descent: полилиния safe_z→цель с постоянной ориентацией down (меньше вращения ЗХ)"
                    )
                    self._execute_plan(plan_d)
                    return
                rospy.loginfo("constant-down polyline не собралась — резерв (none→down или два плана)")
            plan, ok = self._try_cartesian_polyline([(x, y, fs, ori1), (x, y, fz, ori2)])
            if ok and plan is not None:
                self._execute_plan(plan)
                return
            rospy.loginfo("декартова полилиния (safe_z→цель) не собралась — два отдельных сегмента")
        self._execute_plan(self._plan_segment(x, y, fs, ori=ori1, method_override=planning_method_override))
        self._execute_plan(self._plan_segment(x, y, fz, ori=ori2, method_override=planning_method_override))

    def _try_place_descent_vertical_polyline(self, z_goal: float) -> bool:
        """
        Только Z при текущих X,Y (где реально висит схват после переноса). Мелкие waypoints — выше fraction.
        """
        if not param_bool("~place_prefer_vertical_polyline_first", True):
            return False
        try:
            self.group.clear_pose_targets()
            try:
                self.group.set_start_state_to_current_state()
            except Exception:
                pass
            cp = self.group.get_current_pose()
            x = float(cp.pose.position.x)
            y = float(cp.pose.position.y)
            z0 = float(cp.pose.position.z)
        except Exception as e:
            rospy.logwarn("vertical polyline: pose: %s", e)
            return False
        if z_goal >= z0 - 1e-6:
            return True
        # Одна прямая в ЗХ (текущая поза → цель) — минимальный путь; при неудаче fraction — полилиния
        if param_bool("~place_vertical_single_straight", True):
            plan, ok = self._try_cartesian_straight(x, y, z_goal, "none")
            if ok and plan is not None:
                self._execute_plan(plan)
                rospy.loginfo("place vertical: одна прямая в ЗХ z %.4f → %.4f", z0, z_goal)
                return True
            rospy.logwarn("place vertical: прямая не собралась (fraction), пробуем полилинию")
        sub = float(rospy.get_param("~place_vertical_cartesian_substep_m", 0.018))
        n = max(int(math.ceil((z0 - z_goal) / sub)), 1)
        n = min(n, int(rospy.get_param("~place_vertical_cartesian_max_segments", 32)))
        segments: List[Tuple[float, float, float, str]] = []
        for i in range(1, n + 1):
            t = float(i) / float(n)
            z = z0 + (z_goal - z0) * t
            segments.append((x, y, z, "none"))
        plan, ok = self._try_cartesian_polyline(segments)
        if not ok or plan is None:
            rospy.logwarn("vertical polyline: не собрана (z %.4f → %.4f)", z0, z_goal)
            return False
        self._execute_plan(plan)
        rospy.loginfo("place vertical polyline OK: z %.4f → %.4f (%d сегм.)", z0, z_goal, n)
        return True

    def _move_via_safe_z_place_with_fallback(
        self,
        x: float,
        y: float,
        z: float,
        safe_z: float,
        step_name: str,
        planning_method_override: Optional[str] = None,
    ) -> None:
        """Place phase: OMPL often fails if goal z is too high; retry with lower target."""
        try:
            self._move_via_safe_z(x, y, z, safe_z, planning_method_override=planning_method_override)
        except RuntimeError as e:
            if param_bool("~place_pre_place_single_attempt", True):
                rospy.logerr("%s: без повторной попытки (~place_pre_place_single_attempt)", step_name)
                raise
            dz = float(rospy.get_param("~place_fallback_delta_z", 0.10))
            rospy.logwarn("%s failed (%s); retry with z -= %.3f m", step_name, str(e), dz)
            z2 = float(z) - dz
            safe_z2 = float(max(safe_z - dz, z2))
            self._move_via_safe_z(x, y, z2, safe_z2, planning_method_override=planning_method_override)

    def _plan_execute_place_step(self, gx: float, gy: float, zn: float, planning_method_override: Optional[str] = None) -> None:
        """
        Один шаг к (gx,gy,zn). Сначала только позиция (none) — IK чаще находит сгиб локтя/плеча;
        затем жёсткий down; затем OMPL. Жёсткий down с первого шага часто даёт «зависание» над целью.
        Дополнительно: ослабленные допуски, joint-space план, прямой шаг на контроллер (симуляция).

        При ~place_single_plan_attempt=true и фазе снижения: одна вертикаль + один план (без перебора 4×ориентация/метод).
        """
        if self._place_descent_active and param_bool("~place_single_plan_attempt", True):
            skip_v = False
            if getattr(self, "_place_skip_vertical_polyline_once", False):
                skip_v = True
                self._place_skip_vertical_polyline_once = False
            if param_bool("~place_prefer_vertical_polyline_first", True) and not skip_v:
                try:
                    if self._try_place_descent_vertical_polyline(zn):
                        rospy.loginfo("place step OK (vertical polyline): z=%.4f", zn)
                        return
                except Exception as e:
                    rospy.logwarn("vertical polyline place step: %s", e)
            mo = planning_method_override if (planning_method_override and str(planning_method_override).strip()) else str(
                rospy.get_param("~planning_method_place_descent", "cartesian_prefer")
            ).strip().lower()
            plan = self._plan_segment(gx, gy, zn, ori="none", method_override=mo)
            self._execute_plan(plan)
            rospy.loginfo("place step OK (single plan): z=%.4f method=%s", zn, mo)
            return

        skip_v = False
        if getattr(self, "_place_skip_vertical_polyline_once", False):
            skip_v = True
            self._place_skip_vertical_polyline_once = False
        if (
            self._place_descent_active
            and param_bool("~place_prefer_vertical_polyline_first", True)
            and not skip_v
        ):
            try:
                if self._try_place_descent_vertical_polyline(zn):
                    rospy.loginfo("place step OK (vertical polyline): z=%.4f", zn)
                    return
            except Exception as e:
                rospy.logwarn("vertical polyline place step: %s", e)
        md = str(rospy.get_param("~planning_method_place_descent", "cartesian_prefer")).strip().lower()
        mo_def = planning_method_override if (planning_method_override and str(planning_method_override).strip()) else md
        attempts: List[Tuple[str, Optional[str]]] = [
            ("none", mo_def),
            ("down", mo_def),
            ("none", "ompl"),
            ("down", "ompl"),
        ]
        last_err: Optional[Exception] = None
        for ori, method in attempts:
            try:
                plan = self._plan_segment(gx, gy, zn, ori=ori, method_override=method)
                self._execute_plan(plan)
                rospy.loginfo("place step OK: z=%.4f ori=%s method=%s", zn, ori, method)
                return
            except Exception as e:
                last_err = e
                rospy.logwarn("place step z=%.4f ori=%s method=%s: %s", zn, ori, method, e)
        if param_bool("~place_relaxed_tol_fallback", True):
            old_pos = self.group.get_goal_position_tolerance()
            old_ori = self.group.get_goal_orientation_tolerance()
            try:
                self.group.set_goal_position_tolerance(float(rospy.get_param("~place_relaxed_pos_tol", 0.028)))
                self.group.set_goal_orientation_tolerance(float(rospy.get_param("~place_relaxed_ori_tol", 0.15)))
                for ori, method in attempts:
                    try:
                        plan = self._plan_segment(gx, gy, zn, ori=ori, method_override=method)
                        self._execute_plan(plan)
                        rospy.loginfo("place step OK (relaxed tol): z=%.4f ori=%s method=%s", zn, ori, method)
                        return
                    except Exception as e:
                        last_err = e
                        rospy.logwarn("place relaxed z=%.4f ori=%s method=%s: %s", zn, ori, method, e)
            finally:
                self.group.set_goal_position_tolerance(old_pos)
                self.group.set_goal_orientation_tolerance(old_ori)
        try:
            self._try_joint_space_place_fallback()
            rospy.loginfo("place step OK (joint-space fallback), target z=%.4f", zn)
            return
        except Exception as e:
            last_err = e
            rospy.logwarn("joint-space fallback failed: %s", e)
        try:
            if self._direct_joint_nudge_downward_once():
                rospy.loginfo("place step OK (direct joint nudge), target z=%.4f", zn)
                return
        except Exception as e:
            last_err = e
            rospy.logwarn("direct joint nudge failed: %s", e)
        if last_err is not None:
            raise last_err
        raise RuntimeError("place step: empty failure state")

    def _try_joint_space_place_fallback(self) -> None:
        """Малые шаги в joint-space через OMPL, когда pose IK/картезиан не дают траектории вниз."""
        if not param_bool("~place_joint_fallback_enable", True):
            raise RuntimeError("place_joint_fallback_enable=false")
        max_n = max(int(rospy.get_param("~place_joint_fallback_nudges", 10)), 1)
        d_sl = float(rospy.get_param("~place_joint_fallback_delta_shoulder_lift", -0.07))
        d_el = float(rospy.get_param("~place_joint_fallback_delta_elbow", 0.09))
        if param_bool("~place_joint_fallback_invert_sign", False):
            d_sl, d_el = -d_sl, -d_el
        names = list(self.group.get_active_joints())
        for k in range(max_n):
            scale = 1.0 + 0.22 * float(k)
            self.group.clear_pose_targets()
            self.group.clear_path_constraints()
            try:
                self.group.set_start_state_to_current_state()
            except Exception:
                pass
            q = list(self.group.get_current_joint_values())
            for jn, dj in (("shoulder_lift_joint", d_sl), ("elbow_joint", d_el)):
                if jn in names:
                    q[names.index(jn)] += dj * scale
            self.group.set_joint_value_target(q)
            plan = self.group.plan()
            try:
                jt = plan.joint_trajectory
            except Exception:
                jt = plan[1].joint_trajectory if isinstance(plan, (list, tuple)) and len(plan) > 1 else None
            if jt is not None and len(jt.points) > 0:
                self._execute_plan(plan)
                self._apply_path_constraints()
                return
            rospy.sleep(float(rospy.get_param("~place_joint_fallback_sleep_s", 0.03)))
        raise RuntimeError("joint-space fallback: no plan after nudges")

    def _current_joint_positions_for_controller(self) -> List[float]:
        act = list(self.group.get_active_joints())
        q = list(self.group.get_current_joint_values())
        name_to_q = dict(zip(act, q))
        out: List[float] = []
        for jn in self.joint_names:
            if jn not in name_to_q:
                raise RuntimeError(f"joint {jn} not in active joints {act}")
            out.append(float(name_to_q[jn]))
        return out

    def _publish_trajectory_direct(self, positions: List[float], duration: float) -> None:
        if len(positions) != len(self.joint_names):
            raise RuntimeError("joint positions length != joint_names")
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions]
        pt.time_from_start = rospy.Duration(duration)
        msg.points = [pt]
        self.pub_arm.publish(msg)
        pad = float(rospy.get_param("~direct_joint_traj_wait_pad_s", 0.06))
        rospy.sleep(max(duration + pad, 0.22))
        self._pause("after_direct_joint")

    def _direct_joint_nudge_downward_once(self) -> bool:
        """Последний резерв: короткая траектория на arm_controller без collision-check MoveIt (только сим)."""
        if not param_bool("~place_direct_joint_nudge_enable", True):
            return False
        z0 = float(self.group.get_current_pose().pose.position.z)
        d_sl0 = float(rospy.get_param("~place_direct_joint_nudge_shoulder_lift", -0.06))
        d_el0 = float(rospy.get_param("~place_direct_joint_nudge_elbow", 0.085))
        if param_bool("~place_direct_joint_nudge_invert", False):
            d_sl0, d_el0 = -d_sl0, -d_el0
        try_both = param_bool("~place_direct_joint_nudge_try_both", True)
        for flip in (False, True):
            if flip and not try_both:
                break
            d_sl, d_el = (-d_sl0, -d_el0) if flip else (d_sl0, d_el0)
            pos = self._current_joint_positions_for_controller()
            names = list(self.joint_names)
            for jn, dj in (("shoulder_lift_joint", d_sl), ("elbow_joint", d_el)):
                if jn in names:
                    pos[names.index(jn)] += dj
            dur = float(rospy.get_param("~place_direct_joint_nudge_duration_s", 0.52))
            self._publish_trajectory_direct(pos, dur)
            self._apply_path_constraints()
            z1 = float(self.group.get_current_pose().pose.position.z)
            rospy.loginfo("direct joint nudge EE z: %.4f -> %.4f (flip=%s)", z0, z1, flip)
            if z1 < z0 - 5.0e-4:
                return True
            z0 = z1
        return False

    def _descend_to_place_height(
        self,
        gx: float,
        gy: float,
        z_final: float,
        pre_z: float,
        planning_method_override: Optional[str] = None,
    ) -> None:
        """
        Снижение над (gx,gy) до z_final малыми шагами по Z; на каждом шаге — несколько попыток IK (none/down/OMPL).
        pre_z — высота pre_place в base_link; нужна для _effective_place_target_z (анти «ложный успех»).
        """
        self._pick_carry_active = False
        z_final = self._effective_place_target_z(z_final, pre_z)
        self._place_descent_active = True
        self._place_skip_vertical_polyline_once = False
        if param_bool("~place_smooth_descent", True):
            dz = float(rospy.get_param("~place_descent_step_m", 0.032))
        else:
            dz = float(rospy.get_param("~place_descent_step_coarse_m", 0.055))
        eps = float(rospy.get_param("~place_descent_eps_m", 0.006))
        max_iter = int(rospy.get_param("~place_descent_max_steps", 48))
        try:
            # Режим «одна попытка укладки»: вертикаль до z_final, при необходимости один запасной план — без циклов IK
            if param_bool("~place_one_shot_placement", True):
                # Допуск по высоте шире eps: иначе контроллер чуть не дотягивает → лишний запасной план («вторая попытка»)
                slack = float(rospy.get_param("~place_one_shot_success_slack_m", 0.025))
                vert_ok = self._try_place_descent_vertical_polyline(z_final)
                cz = float(self.group.get_current_pose().pose.position.z)
                if vert_ok and cz <= z_final + max(eps, slack):
                    rospy.loginfo(
                        "place one-shot: вертикаль OK z=%.4f (цель %.4f, допуск %.3f м)",
                        cz,
                        z_final,
                        max(eps, slack),
                    )
                    return
                if vert_ok:
                    rospy.logwarn(
                        "place one-shot: вертикаль выполнена, но z=%.4f выше цели с допуском (цель %.4f) — запасной план",
                        cz,
                        z_final,
                    )
                if param_bool("~place_one_shot_backup_plan", True):
                    cp = self.group.get_current_pose().pose.position
                    if param_bool("~place_descent_keep_current_xy", True):
                        gx_e, gy_e = float(cp.x), float(cp.y)
                    else:
                        gx_e, gy_e = gx, gy
                    mo = (
                        planning_method_override
                        if (planning_method_override and str(planning_method_override).strip())
                        else str(rospy.get_param("~planning_method_place_descent", "cartesian_prefer")).strip().lower()
                    )
                    plan = self._plan_segment(gx_e, gy_e, z_final, ori="none", method_override=mo)
                    self._execute_plan(plan)
                    cz = float(self.group.get_current_pose().pose.position.z)
                    rospy.loginfo("place one-shot: запасной план z=%.4f (цель %.4f)", cz, z_final)
                    if param_bool("~place_one_shot_strict_height", False) and cz > z_final + max(eps, slack):
                        raise RuntimeError(
                            f"place one-shot: EE z={cz:.4f} выше цели {z_final:.4f} с допуском (включите пошаговый режим или ослабьте strict)"
                        )
                    return
                raise RuntimeError("place one-shot: вертикаль не достигла цели, запасной план отключён (~place_one_shot_backup_plan)")

            # Одна вертикальная траектория до z_final — меньше «попыток» и изгибов, чем много микрошагов
            if param_bool("~place_simple_descent", True):
                if self._try_place_descent_vertical_polyline(z_final):
                    cz = float(self.group.get_current_pose().pose.position.z)
                    if cz <= z_final + eps:
                        rospy.loginfo(
                            "place simple descent: одна вертикаль до z=%.4f (цель %.4f)",
                            cz,
                            z_final,
                        )
                        return
                    # Частично дошли — не дублировать полную вертикаль на первом микрошаге
                    self._place_skip_vertical_polyline_once = True
                rospy.loginfo("place: пошаговый резерв (простое снижение не завершило цель полностью)")
            for i in range(max_iter):
                cp = self.group.get_current_pose().pose.position
                cz = float(cp.z)
                if cz <= z_final + eps:
                    rospy.loginfo("place descent: достигнуто z=%.4f (цель %.4f)", cz, z_final)
                    return
                next_z = max(z_final, cz - dz)
                rospy.loginfo("place descent: шаг %d z %.4f → %.4f (цель %.4f)", i + 1, cz, next_z, z_final)
                if param_bool("~place_descent_keep_current_xy", True):
                    gx_eff = float(cp.x)
                    gy_eff = float(cp.y)
                    if abs(gx_eff - gx) > 0.012 or abs(gy_eff - gy) > 0.012:
                        rospy.logwarn(
                            "place descent: XY из топика (%.4f,%.4f) ≠ текущие ЗХ (%.4f,%.4f) — снижение только по Z",
                            gx,
                            gy,
                            gx_eff,
                            gy_eff,
                        )
                else:
                    gx_eff, gy_eff = gx, gy
                self._plan_execute_place_step(gx_eff, gy_eff, next_z, planning_method_override=planning_method_override)
            raise RuntimeError(
                f"place descent: за {max_iter} шагов не достигнуто z<={z_final + eps:.4f} (текущее z проверьте визуально)"
            )
        finally:
            self._place_descent_active = False
            self._place_skip_vertical_polyline_once = False

    def _execute_plan(self, plan) -> None:
        try:
            jt = plan.joint_trajectory
        except Exception:
            jt = plan[1].joint_trajectory if isinstance(plan, (list, tuple)) and len(plan) > 1 else None
        if jt is None or len(jt.points) == 0:
            raise RuntimeError("Empty trajectory from MoveIt plan()")

        jt = self._thin_joint_trajectory(jt)
        jt = self._stamp_uniform_joint_times(jt)
        jt_scaled = self._time_scale_trajectory(jt)
        self._publish_trajectory(jt_scaled)
        dur = float(jt_scaled.points[-1].time_from_start.to_sec())
        cap = float(rospy.get_param("~max_motion_wait_s", 45.0))
        if dur > cap:
            rospy.logwarn("длительность траектории %.1f s > max_motion_wait_s=%.1f — ограничиваем ожидание", dur, cap)
        dur = min(dur, cap)
        rospy.loginfo("ожидание движения: %.1f s (~%d точек)", dur, len(jt_scaled.points))
        floor = float(rospy.get_param("~motion_wait_floor_s", 0.3))
        if self._place_descent_active:
            floor = float(rospy.get_param("~motion_wait_floor_place_descent_s", 0.06))
        rospy.sleep(max(dur, floor))
        self._pause("after_motion")

    def _pause(self, reason: str) -> None:
        if self._place_descent_active:
            s = float(rospy.get_param("~post_action_pause_descent_s", 0.08))
        else:
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

    def _repair_grasp_for_attach(self, cube_fallback: PoseStamped) -> None:
        """
        Если после move_grasp EE всё ещё выше/дальше допусков attach — короткие шаги вниз по Z
        (симулятор/дробление декартова пути часто недотягивают до grasp_height_offset).
        """
        if not param_bool("~grasp_attach_repair_enable", True):
            return
        n = max(int(rospy.get_param("~grasp_attach_repair_max_attempts", 5)), 1)
        step = float(rospy.get_param("~grasp_attach_repair_step_m", 0.018))
        method = str(rospy.get_param("~grasp_attach_repair_planning_method", "cartesian_prefer")).strip().lower()
        min_z_cfg = float(rospy.get_param("~grasp_attach_repair_min_ee_z_base", -1.0))
        if min_z_cfg < 0.0:
            min_z = self._min_safe_z_horizontal() - float(
                rospy.get_param("~grasp_attach_repair_below_safe_z_margin_m", 0.15)
            )
        else:
            min_z = min_z_cfg
        for i in range(n):
            cube_now = self._latest_cube if self._latest_cube is not None else cube_fallback
            cb = self._world_to_base(cube_now).pose.position
            cxyz = (float(cb.x), float(cb.y), float(cb.z))
            try:
                ee = self.group.get_current_pose().pose.position
                validate_virtual_attach((float(ee.x), float(ee.y), float(ee.z)), cxyz, self.attach_limits)
                if i > 0:
                    rospy.loginfo("grasp_attach_repair: геометрия OK после %d микро-шаг(ов)", i)
                return
            except RuntimeError as e:
                rospy.logwarn("grasp_attach_repair: проверка %d/%d — %s", i + 1, n, e)
            try:
                ee = self.group.get_current_pose().pose.position
                z_new = float(ee.z) - step
                if z_new < min_z:
                    rospy.logwarn(
                        "grasp_attach_repair: z_new=%.3f < min_ee_z_base=%.3f — остановка ремонта",
                        z_new,
                        min_z,
                    )
                    break
                self._pick_carry_active = False
                self._grasp_descent_active = True
                self._apply_path_constraints()
                try:
                    plan = self._plan_segment(float(ee.x), float(ee.y), z_new, ori="none", method_override=method)
                    self._execute_plan(plan)
                finally:
                    self._grasp_descent_active = False
                    self._apply_path_constraints()
            except Exception as ex:
                rospy.logwarn("grasp_attach_repair: план/исполнение: %s", ex)
                break
        cube_now = self._latest_cube if self._latest_cube is not None else cube_fallback
        cb = self._world_to_base(cube_now).pose.position
        cxyz = (float(cb.x), float(cb.y), float(cb.z))
        ee = self.group.get_current_pose().pose.position
        validate_virtual_attach((float(ee.x), float(ee.y), float(ee.z)), cxyz, self.attach_limits)

    def _run_pick_phase(self, steps: StepRunner, cube: PoseStamped) -> None:
        with steps.step("gripper_open"):
            self._call_trigger(self.gripper_open_srv)
            self._pause("after_gripper_open")

        pre = self._make_pose(
            cube.pose.position.x,
            cube.pose.position.y,
            cube.pose.position.z + self.cfg.approach_height,
        )
        pre_b = self._world_to_base(pre)
        try:
            with steps.step("move_pre_grasp"):
                # Без _pick_carry_active: иначе коридор запястий от «дома» делает цель pre_grasp недостижимой для OMPL.
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
                # Без carry; без limit_joint_deviation на время шага — иначе часто не хватает сгиба до grasp z.
                self._pick_carry_active = False
                self._grasp_descent_active = True
                self._apply_path_constraints()
                try:
                    cur_z = float(self.group.get_current_pose().pose.position.z)
                    self._move_via_safe_z(
                        float(grasp_b.pose.position.x),
                        float(grasp_b.pose.position.y),
                        float(grasp_b.pose.position.z),
                        safe_z=max(cur_z, float(pre_b.pose.position.z)),
                    )
                finally:
                    self._grasp_descent_active = False
                    self._apply_path_constraints()

            # carry выключен до конца move_lift — иначе подъём с кубом часто без плана (вися над кубом).
            self._pick_carry_active = False
            self._apply_path_constraints()
            # Опционально сомкнуть губки; иначе только /attach (куб «прилипает» к схвату без сжатия в симуляции).
            if self.gripper_close_before_attach:
                with steps.step("gripper_close_before_attach"):
                    self._call_trigger(self.gripper_close_srv)
                    self._pause("after_gripper_close")
                rospy.sleep(float(rospy.get_param("~gripper_settle_after_close_s", 0.6)))
            else:
                rospy.loginfo("gripper_close_before_attach=false: смыкание пропущено, короткая пауза перед /attach")
                rospy.sleep(float(rospy.get_param("~gripper_settle_before_attach_s", 0.2)))

            with steps.step("attach_cube"):
                cube_now = self._latest_cube if self._latest_cube is not None else cube
                self._repair_grasp_for_attach(cube_now)
                ee = self.group.get_current_pose().pose.position
                cube_now_b = self._world_to_base(cube_now).pose.position
                dz, d = validate_virtual_attach(
                    (float(ee.x), float(ee.y), float(ee.z)),
                    (float(cube_now_b.x), float(cube_now_b.y), float(cube_now_b.z)),
                    self.attach_limits,
                )
                rospy.loginfo(
                    "Calling /attach: dz=%.3f m, dist=%.3f m — куб следует за gripper_base_link (сжатие=%s).",
                    dz,
                    d,
                    self.gripper_close_before_attach,
                )
                self._call_trigger(self.attach_srv)
                self._cube_attached_in_cycle = True
                self._pause("after_attach")

            # Lift vertically from current EE: после /attach куб «прыгает» к схвату — цель по старому pose куба неверна.
            with steps.step("move_lift"):
                self._pick_carry_active = False
                self._post_attach_lift_active = True
                self._apply_path_constraints()
                try:
                    ee = self.group.get_current_pose().pose.position
                    cur_z = float(ee.z)
                    dz_lift = float(self.cfg.lift_height) - float(self.grasp_height_offset)
                    z_tgt = float(ee.z) + dz_lift
                    _pm = str(rospy.get_param("~planning_method", "cartesian_prefer")).strip().lower()
                    _lift_m = str(rospy.get_param("~planning_method_lift", _pm)).strip().lower()
                    self._move_via_safe_z(
                        float(ee.x),
                        float(ee.y),
                        z_tgt,
                        safe_z=max(cur_z, z_tgt),
                        planning_method_override=_lift_m,
                    )
                finally:
                    self._post_attach_lift_active = False
                    self._apply_path_constraints()
        finally:
            self._pick_carry_active = False
            self._grasp_descent_active = False
            self._post_attach_lift_active = False
            self._move_pre_place_active = False
            self._apply_path_constraints()

    def _run_place_phase(self, steps: StepRunner, goal: PoseStamped) -> None:
        # Свежая цель после pick (долгий подъём): иначе устаревший goal → нереализуемая точка.
        try:
            goal = self._wait_for_pose("goal", timeout_s=10.0)
        except Exception as e:
            rospy.logwarn("не удалось обновить goal перед place (%s), используем переданный pose", e)

        pre_place = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + self.cfg.place_approach_height,
        )
        pre_place_b = self._clamp_pose_base_z(self._world_to_base(pre_place), "pre_place")
        with steps.step("move_pre_place"):
            self._pick_carry_active = True
            self._move_pre_place_active = True
            self._apply_path_constraints()
            try:
                cur_z = float(self.group.get_current_pose().pose.position.z)
                _mpp = str(rospy.get_param("~planning_method_pre_place", "")).strip().lower()
                _mo_pp = _mpp if _mpp else None
                self._move_via_safe_z_place_with_fallback(
                    float(pre_place_b.pose.position.x),
                    float(pre_place_b.pose.position.y),
                    float(pre_place_b.pose.position.z),
                    safe_z=max(cur_z, float(pre_place_b.pose.position.z)),
                    step_name="move_pre_place",
                    planning_method_override=_mo_pp,
                )
            finally:
                self._move_pre_place_active = False
                self._pick_carry_active = False
                self._apply_path_constraints()

        base_clear = float(self.cfg.cube_size * 0.5) + self.cfg.place_clearance
        dz_fb = float(rospy.get_param("~place_z_fallback_delta_m", 0.06))
        n_try = max(int(rospy.get_param("~place_z_fallback_attempts", 4)), 1)
        with steps.step("move_place"):
            last_err = None
            for attempt in range(n_try):
                place = self._make_pose(
                    goal.pose.position.x,
                    goal.pose.position.y,
                    goal.pose.position.z + base_clear - float(attempt) * dz_fb,
                )
                place_b = self._clamp_pose_base_z(self._world_to_base(place), "move_place")
                place_b = self._ensure_place_below_pre(pre_place_b, place_b)
                try:
                    # Всегда пошаговое снижение: один большой _move_via_safe_z с down часто не даёт согнуть локоть (IK).
                    self._descend_to_place_height(
                        float(place_b.pose.position.x),
                        float(place_b.pose.position.y),
                        float(place_b.pose.position.z),
                        float(pre_place_b.pose.position.z),
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    rospy.logwarn(
                        "move_place попытка %d/%d не удалась (%s); z_world −= %.3f м",
                        attempt + 1,
                        n_try,
                        e,
                        dz_fb,
                    )
            if last_err is not None:
                raise last_err

        with steps.step("detach_cube"):
            self._call_trigger(self.detach_srv)
            self._cube_attached_in_cycle = False
            self._pause("after_detach")

        with steps.step("gripper_open_after_place"):
            self._call_trigger(self.gripper_open_srv)
            self._pause("after_gripper_open_place")

        retreat = self._make_pose(
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z + self.cfg.place_approach_height,
        )
        retreat_b = self._clamp_pose_base_z(self._world_to_base(retreat), "retreat")
        with steps.step("move_retreat"):
            cur_z = float(self.group.get_current_pose().pose.position.z)
            self._move_via_safe_z(
                float(retreat_b.pose.position.x),
                float(retreat_b.pose.position.y),
                float(retreat_b.pose.position.z),
                safe_z=max(cur_z, float(retreat_b.pose.position.z)),
            )

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

        self._cube_attached_in_cycle = False
        self._run_pick_phase(steps, cube)
        try:
            self._run_place_phase(steps, goal)
        except Exception as e:
            rospy.logerr("фаза place завершилась с ошибкой: %s", e)
            if param_bool("~go_home_on_place_failure", True):
                if self._cube_attached_in_cycle and param_bool("~detach_before_home_on_place_failure", True):
                    with steps.step("detach_before_emergency_home"):
                        try:
                            self._call_trigger(self.detach_srv)
                            self._cube_attached_in_cycle = False
                            self._pause("after_detach_emergency")
                            rospy.logwarn(
                                "Куб отсоединён перед аварийным home — иначе Gazebo тащит его вместе с манипулятором"
                            )
                        except Exception as e_detach:
                            rospy.logwarn("detach перед аварийным home не удался: %s", e_detach)
                with steps.step("go_home_after_place_error"):
                    try:
                        self._send_home()
                        self._pause("after_go_home_error")
                    except Exception as e2:
                        rospy.logwarn("go_home после ошибки place: %s", e2)
            raise
        if param_bool("~go_home_after_place", True):
            with steps.step("go_home_after_place"):
                self._send_home()
                self._pause("after_go_home_end")

        rospy.loginfo("Pick&place завершён. Ожидание: rosservice call .../run_pick_place или spin.")

    def spin(self) -> None:
        rospy.sleep(1.0)
        run_once = param_bool("~run_once", True)
        if run_once:
            try:
                with self._run_lock:
                    self.run_once()
            except Exception as e:
                rospy.logerr("pick&place остановлен с ошибкой: %s", e)
                rospy.logerr("%s", traceback.format_exc())
            rospy.loginfo("Узел активен. Повтор: rosservice call <ns>/pick_place_moveit/run_pick_place")
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

