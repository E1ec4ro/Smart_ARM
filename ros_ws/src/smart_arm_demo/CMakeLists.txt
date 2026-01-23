cmake_minimum_required(VERSION 3.0.2)
project(smart_arm_demo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  trajectory_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/move_ur5_demo.py
  scripts/neural_network_camera.py
  scripts/gripper_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

