#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerResponse


class GripperControl:
    def __init__(self) -> None:
        rospy.init_node("gripper_control", anonymous=True)

        self.left_topic = rospy.get_param("~left_topic", "/finger_left_controller/command")
        self.right_topic = rospy.get_param("~right_topic", "/finger_right_controller/command")

        self.open_position = float(rospy.get_param("~open_position", 0.0))
        self.close_position = float(rospy.get_param("~close_position", 0.03))

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
        start_open = rospy.get_param("~start_open", True)
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

