cmake_minimum_required(VERSION 2.8.3)
project(action_controller)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Boost REQUIRED thread)

include_directories(SYSTEM "msg_gen/cpp/include")

find_package(catkin REQUIRED COMPONENTS
  # moveit_core
  pluginlib
  actionlib
  roscpp
  # trajectory_msgs
  actionlib_msgs
  std_msgs
  message_generation
  geometry_msgs
  sensor_msgs
  # tum_ardrone
  rospy
  dynamic_reconfigure
)

include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

add_message_files(
  FILES
)

add_service_files(
  FILES
  # DenseCaption.srv
  MeteorScore.srv
)

add_action_files(
  DIRECTORY action
  FILES
  DenseCaption.action
  DenseImageQuery.action
  Localize.action
  LocalizeProbs.action
  LocalizeQuery.action
  ExtractFeatures.action
  DenseRefexpLoad.action
  DenseRefexpQuery.action
  DenseRefexpLoadQuery.action
  RelevancyClustering.action
  BoxesRefexpQuery.action
  DenseRefexpLoadBBoxes.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  # trajectory_msgs
  actionlib_msgs
  action_controller
  # tum_ardrone
  dynamic_reconfigure
  sensor_msgs
)

generate_dynamic_reconfigure_options(
  cfg/RefexpMicoDemo.cfg
)

catkin_package(
  LIBRARIES
  INCLUDE_DIRS
  DEPENDS
  CATKIN_DEPENDS message_runtime
  # moveit_core
  geometry_msgs
  sensor_msgs
  # tum_ardrone
  dynamic_reconfigure
  actionlib_msgs
)

include_directories(include)

# # add_library(action_controller src/actionController.cpp)
# add_executable(action_controller src/actionController.cpp)
# target_link_libraries(action_controller ${catkin_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


# ### Controller Priority Mux: ---------------------------

# add_executable(controller_priority_mux src/controller_priority_mux.cpp)

# ## Add cmake target dependencies of the executable
# ## same as for the library above
# add_dependencies(controller_priority_mux ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ## Specify libraries to link a library or executable target against
# target_link_libraries(controller_priority_mux
#   ${catkin_LIBRARIES}
# )


# ### Drone State Publisher: ---------------------------

# add_executable(drone_state_publisher src/drone_state_publisher.cpp)

# ## Add cmake target dependencies of the executable
# ## same as for the library above
# add_dependencies(drone_state_publisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ## Specify libraries to link a library or executable target against
# target_link_libraries(drone_state_publisher
#   ${catkin_LIBRARIES}
# )


# ### EKF State Publisher: ---------------------------

# add_executable(ekf_state_publisher src/ekf_state_publisher.cpp)

# ## Add cmake target dependencies of the executable
# ## same as for the library above
# add_dependencies(ekf_state_publisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ## Specify libraries to link a library or executable target against
# target_link_libraries(ekf_state_publisher
#   ${catkin_LIBRARIES}
# )



#install(TARGETS action_controller LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})


