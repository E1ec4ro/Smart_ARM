#!/usr/bin/env python3
"""
Виртуальный attach куба к фланцу в Gazebo.

Важно: поза из ModelStates — это опорная точка *модели*, а не центр cube_link.
При ненулевом <pose> ссылки в SDF set_model_state «ломает» положение куба.
Используем target_cube::cube_link из /gazebo/link_states и /gazebo/set_link_state.
"""
import copy
import os
import sys
import threading

import rospy
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import LinkState, LinkStates, ModelState, ModelStates
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest, SetModelState
from std_srvs.srv import Trigger, TriggerResponse

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from grasp_workspace import min_cube_center_z_above_table, param_bool


def _rotate_vec_by_quat(qx: float, qy: float, qz: float, qw: float, vx: float, vy: float, vz: float):
    """Rotate vector (vx,vy,vz) by unit quaternion (x,y,z,w)."""
    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    return (
        float(vx + 2.0 * (qw * uvx + uuvx)),
        float(vy + 2.0 * (qw * uvy + uuvy)),
        float(vz + 2.0 * (qw * uvz + uuvz)),
    )


class GazeboAttachNode:
    def __init__(self) -> None:
        rospy.init_node("gazebo_attach", anonymous=False)

        self.model_name = rospy.get_param("~model_name", "target_cube")
        self.link_name = rospy.get_param("~link_name", "ur5::gripper_base_link")
        self.cube_link_name = rospy.get_param("~cube_link_name", "target_cube::cube_link")
        _off = rospy.get_param("~offset_xyz", [0.0, 0.0, 0.02])
        if isinstance(_off, str):
            s = _off.strip()
            s = s[1:-1] if s.startswith("[") and s.endswith("]") else s
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 3:
                _off = [float(parts[0]), float(parts[1]), float(parts[2])]
            else:
                rospy.logwarn(f"Invalid offset_xyz string: {_off}, using default")
                _off = [0.0, 0.0, 0.02]
        self._offset_xyz_default = [float(_off[0]), float(_off[1]), float(_off[2])]
        self._active_offset = list(self._offset_xyz_default)
        self._dynamic_offset_on_attach = param_bool("~dynamic_offset_on_attach", True)
        # True: ориентация куба = world identity (стабильнее); False: как у фланца (куб крутится с запястьем)
        self._cube_orientation_identity = param_bool("~cube_orientation_identity", True)
        self._use_set_link_state = param_bool("~use_set_link_state", True)

        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))
        _tbl = rospy.get_param("~table", {})
        if not isinstance(_tbl, dict):
            _tbl = {}
        _tc = rospy.get_param("~target_cube", {})
        if not isinstance(_tc, dict):
            _tc = {}
        table_center_z = float(_tbl.get("center_z", rospy.get_param("~table_center_z", 0.40)))
        table_size_z = float(_tbl.get("size_z", rospy.get_param("~table_size_z", 0.05)))
        cube_size = float(_tc.get("size", rospy.get_param("~cube_size", 0.08)))
        table_top_z = table_center_z + table_size_z * 0.5
        min_cube_center_z = min_cube_center_z_above_table(table_center_z, table_size_z, cube_size)
        self.min_hold_z = float(rospy.get_param("~min_hold_z", min_cube_center_z))
        self.release_min_z = float(rospy.get_param("~release_min_z", min_cube_center_z))
        rospy.loginfo(
            "table_top_z=%.4f min_cube_center_z=%.4f release_min_z=%.4f",
            table_top_z,
            min_cube_center_z,
            self.release_min_z,
        )

        self._attached = False
        self._lock = threading.Lock()
        self._link_pose = None
        self._cube_link_pose = None
        self._model_states = None
        self._last_cube_link_pose = None

        rospy.Subscriber("/gazebo/link_states", LinkStates, self._on_link_states, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_model_states, queue_size=1)

        rospy.wait_for_service("/gazebo/set_model_state")
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._set_link_state = None
        if self._use_set_link_state:
            try:
                rospy.wait_for_service("/gazebo/set_link_state", timeout=15.0)
                self._set_link_state = rospy.ServiceProxy("/gazebo/set_link_state", SetLinkState)
            except rospy.ROSException:
                rospy.logwarn("/gazebo/set_link_state недоступен — fallback на set_model_state (менее точно)")
                self._use_set_link_state = False

        rospy.Service("~attach", Trigger, self._srv_attach)
        rospy.Service("~detach", Trigger, self._srv_detach)

        self._compose_in_world = param_bool("~compose_pose_in_world", True)

        rospy.loginfo(
            "gazebo_attach: model=%s gripper_link=%s cube_link=%s use_set_link_state=%s "
            "dynamic_offset=%s cube_orientation_identity=%s",
            self.model_name,
            self.link_name,
            self.cube_link_name,
            self._use_set_link_state,
            self._dynamic_offset_on_attach,
            self._cube_orientation_identity,
        )

    def _on_link_states(self, msg: LinkStates) -> None:
        try:
            idx = msg.name.index(self.link_name)
            self._link_pose = msg.pose[idx]
        except ValueError:
            pass
        try:
            idx = msg.name.index(self.cube_link_name)
            self._cube_link_pose = msg.pose[idx]
        except ValueError:
            pass

    def _on_model_states(self, msg: ModelStates) -> None:
        self._model_states = msg

    def _get_model_pose(self):
        ms = self._model_states
        if ms is None:
            return None
        try:
            idx = ms.name.index(self.model_name)
        except ValueError:
            return None
        return ms.pose[idx]

    def _compute_dynamic_offset(self):
        """Смещение центра куба (cube_link) от фланца в локальных осях gripper_base_link."""
        lp = self._link_pose
        cp = self._cube_link_pose
        if lp is None or cp is None:
            return None
        dwx = float(cp.position.x) - float(lp.position.x)
        dwy = float(cp.position.y) - float(lp.position.y)
        dwz = float(cp.position.z) - float(lp.position.z)
        qx = float(lp.orientation.x)
        qy = float(lp.orientation.y)
        qz = float(lp.orientation.z)
        qw = float(lp.orientation.w)
        lx, ly, lz = _rotate_vec_by_quat(-qx, -qy, -qz, qw, dwx, dwy, dwz)
        return [lx, ly, lz]

    def _cube_orientation_for_attach(self, lp: Pose) -> Pose:
        out = Pose()
        if self._cube_orientation_identity:
            out.orientation.w = 1.0
            out.orientation.x = 0.0
            out.orientation.y = 0.0
            out.orientation.z = 0.0
        else:
            out.orientation = lp.orientation
        return out

    def _build_cube_link_pose(self) -> Pose:
        """Мировая поза cube_link: фланец + R(q)*offset (offset в осях фланца)."""
        lp = self._link_pose
        ox, oy, oz = float(self._active_offset[0]), float(self._active_offset[1]), float(self._active_offset[2])
        qx = float(lp.orientation.x)
        qy = float(lp.orientation.y)
        qz = float(lp.orientation.z)
        qw = float(lp.orientation.w)
        rx, ry, rz = _rotate_vec_by_quat(qx, qy, qz, qw, ox, oy, oz)
        p = Pose()
        p.position.x = float(lp.position.x) + rx
        p.position.y = float(lp.position.y) + ry
        p.position.z = float(lp.position.z) + rz
        p.orientation = self._cube_orientation_for_attach(lp).orientation
        return p

    def _apply_cube_link_pose(self, pose: Pose) -> bool:
        """Задать позу cube_link в world."""
        if self._set_link_state is None:
            return False
        ls = LinkState()
        ls.link_name = self.cube_link_name
        ls.pose = pose
        ls.twist = Twist()
        ls.reference_frame = "world"
        try:
            req = SetLinkStateRequest()
            req.link_state = ls
            resp = self._set_link_state(req)
            if not resp.success:
                rospy.logwarn_throttle(2.0, "set_link_state failed: %s", resp.status_message)
                return False
            return True
        except Exception as e:
            rospy.logwarn("set_link_state exception: %s", e)
            return False

    def _fallback_set_model_state(self, pose: Pose) -> bool:
        """Запасной путь: позиция модели ≈ центр куба (неточно при offset link в SDF)."""
        st = ModelState()
        st.model_name = self.model_name
        st.reference_frame = "world"
        st.pose = pose
        st.twist.linear.x = 0.0
        st.twist.linear.y = 0.0
        st.twist.linear.z = 0.0
        st.twist.angular.x = 0.0
        st.twist.angular.y = 0.0
        st.twist.angular.z = 0.0
        try:
            resp = self._set_state(st)
            return bool(resp.success)
        except Exception as e:
            rospy.logwarn("set_model_state fallback: %s", e)
            return False

    def _publish_attached_pose(self) -> None:
        pose = self._build_cube_link_pose()
        ok = False
        if self._use_set_link_state and self._compose_in_world:
            ok = self._apply_cube_link_pose(pose)
        if not ok:
            ok = self._fallback_set_model_state(pose)
        if ok:
            self._last_cube_link_pose = pose

    def _srv_attach(self, _req) -> TriggerResponse:
        with self._lock:
            if self._dynamic_offset_on_attach:
                dyn = self._compute_dynamic_offset()
                if dyn is not None:
                    self._active_offset = dyn
                    rospy.loginfo(
                        "gazebo_attach: динамическое смещение (cube_link − фланец, локально) = [%.4f, %.4f, %.4f]",
                        dyn[0],
                        dyn[1],
                        dyn[2],
                    )
                else:
                    self._active_offset = list(self._offset_xyz_default)
                    rospy.logwarn("gazebo_attach: нет pose фланца/cube_link — offset из YAML")
            else:
                self._active_offset = list(self._offset_xyz_default)
            self._attached = True
        self._publish_attached_pose()
        return TriggerResponse(success=True, message="attached=true")

    def _srv_detach(self, _req) -> TriggerResponse:
        with self._lock:
            self._attached = False
            self._active_offset = list(self._offset_xyz_default)
            pose = self._last_cube_link_pose
        if pose is not None:
            p = copy.deepcopy(pose)
            p.position.z = max(float(p.position.z), self.release_min_z, self.min_hold_z)
            if self._use_set_link_state and self._set_link_state is not None:
                ls = LinkState()
                ls.link_name = self.cube_link_name
                ls.pose = p
                ls.twist = Twist()
                ls.reference_frame = "world"
                try:
                    req = SetLinkStateRequest()
                    req.link_state = ls
                    self._set_link_state(req)
                except Exception as e:
                    rospy.logwarn("set_link_state on detach: %s", e)
            else:
                self._fallback_set_model_state(p)
        return TriggerResponse(success=True, message="attached=false")

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self._lock:
                attached = self._attached
            if attached and self._link_pose is not None:
                self._publish_attached_pose()
            rate.sleep()


def main() -> None:
    try:
        GazeboAttachNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
