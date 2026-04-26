cmake_minimum_required(VERSION 2.8.3)
project(detection_pipeline_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package()

# Install Python executables
install(PROGRAMS
  detection_pipeline_ros/detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Install model files
install(DIRECTORY
  model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
