#!/usr/bin/env python3
import math
import time

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main() -> None:
    rospy.init_node("move_ur5_demo", anonymous=False)
    pub = rospy.Publisher("/arm_controller/command", JointTrajectory, queue_size=1)

    base = [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]

    t0 = time.time()
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        if time.time() - t0 > 60:
            rospy.logwarn("Нет подписчика на /arm_controller/command уже 60с. Продолжаю попытки...")
            t0 = time.time()
        rospy.sleep(0.2)

    rospy.loginfo("Старт демо-движения UR5: синус на shoulder_pan_joint.")

    rate = rospy.Rate(10)
    start = rospy.get_time()

    while not rospy.is_shutdown():
        t = rospy.get_time() - start

        q = list(base)
        q[0] = 0.8 * math.sin(0.5 * t)
        q[3] = -1.57 + 0.4 * math.sin(0.25 * t)

        msg = JointTrajectory()
        msg.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.time_from_start = rospy.Duration(0.5)
        msg.points = [pt]

        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()

