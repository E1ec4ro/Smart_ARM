#!/usr/bin/env python3
import threading

import rospy
from gazebo_msgs.msg import LinkStates, ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Trigger, TriggerResponse


class GazeboAttachNode:
    def __init__(self) -> None:
        rospy.init_node("gazebo_attach", anonymous=False)

        self.model_name = rospy.get_param("~model_name", "target_cube")
        self.link_name = rospy.get_param("~link_name", "ur5::gripper_base_link")
        self.offset_xyz = rospy.get_param("~offset_xyz", [0.0, 0.0, 0.02])
        if isinstance(self.offset_xyz, str):
            s = self.offset_xyz.strip()
            s = s[1:-1] if s.startswith("[") and s.endswith("]") else s
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 3:
                self.offset_xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            else:
                rospy.logwarn(f"Invalid offset_xyz string: {self.offset_xyz}, using default")
                self.offset_xyz = [0.0, 0.0, 0.02]

        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))

        self._attached = False
        self._lock = threading.Lock()
        self._link_pose = None

        rospy.Subscriber("/gazebo/link_states", LinkStates, self._on_link_states, queue_size=1)

        rospy.wait_for_service("/gazebo/set_model_state")
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.Service("~attach", Trigger, self._srv_attach)
        rospy.Service("~detach", Trigger, self._srv_detach)

        rospy.loginfo(f"gazebo_attach ready. model={self.model_name} link={self.link_name}")

    def _srv_attach(self, _req) -> TriggerResponse:
        with self._lock:
            self._attached = True
        return TriggerResponse(success=True, message="attached=true")

    def _srv_detach(self, _req) -> TriggerResponse:
        with self._lock:
            self._attached = False
        return TriggerResponse(success=True, message="attached=false")

    def _on_link_states(self, msg: LinkStates) -> None:
        try:
            idx = msg.name.index(self.link_name)
        except ValueError:
            return
        self._link_pose = msg.pose[idx]

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self._lock:
                attached = self._attached
            if attached and self._link_pose is not None:
                lp = self._link_pose
                ox, oy, oz = float(self.offset_xyz[0]), float(self.offset_xyz[1]), float(self.offset_xyz[2])
                st = ModelState()
                st.model_name = self.model_name
                st.reference_frame = "world"
                st.pose.position.x = lp.position.x + ox
                st.pose.position.y = lp.position.y + oy
                st.pose.position.z = lp.position.z + oz
                st.pose.orientation = lp.orientation
                try:
                    self._set_state(st)
                except Exception as e:
                    rospy.logwarn(f"set_model_state failed: {e}")
            rate.sleep()


def main() -> None:
    try:
        GazeboAttachNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

