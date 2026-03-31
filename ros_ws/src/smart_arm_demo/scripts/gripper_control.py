#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerResponse


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


class GripperControl:
    def __init__(self) -> None:
        rospy.init_node("gripper_control", anonymous=True)

        self.left_topic = rospy.get_param("~left_topic", "/finger_left_controller/command")
        self.right_topic = rospy.get_param("~right_topic", "/finger_right_controller/command")

        self.open_position = float(rospy.get_param("~open_position", 0.0))
        # URDF finger limits 0..0.055 m; lower = gentler grasp (attach carries load in sim, not squeeze)
        self.close_position = float(rospy.get_param("~close_position", 0.02))

        self.pub_left = rospy.Publisher(self.left_topic, Float64, queue_size=1)
        self.pub_right = rospy.Publisher(self.right_topic, Float64, queue_size=1)

        rospy.Service("~open", Trigger, self.handle_open)
        rospy.Service("~close", Trigger, self.handle_close)

        rospy.loginfo("Gripper control ready. Services: ~open, ~close")

    def publish(self, pos: float) -> None:
        self.pub_left.publish(Float64(pos))
        self.pub_right.publish(Float64(pos))

    def handle_open(self, _req) -> TriggerResponse:
        self.publish(self.open_position)
        return TriggerResponse(success=True, message=f"gripper open -> {self.open_position}")

    def handle_close(self, _req) -> TriggerResponse:
        self.publish(self.close_position)
        return TriggerResponse(success=True, message=f"gripper close -> {self.close_position}")

    def spin(self) -> None:
        start_open = param_bool("~start_open", True)
        if start_open:
            rospy.sleep(0.5)
            self.publish(self.open_position)
        rospy.spin()


def main() -> None:
    try:
        GripperControl().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

