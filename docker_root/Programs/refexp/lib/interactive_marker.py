#!/usr/bin/env python

import rospy
import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster

from random import random
from math import sin

import camera_params

server = None
menu_handler = MenuHandler()
br = None
counter = 0
robot_view_pose = camera_params.ROBOT_VIEW_POSE
user_view_pose = camera_params.USER_VIEW_POSE

ROBOT_FRAME = "robot_view"
USER_FRAME = "user_view"
WORLD_FRAME = "odom_combined"

ROBOT_FRUSTUM_COLOR = [0, 191, 255, 0.8]
USER_FRUSTUM_COLOR = [255, 102, 102, 0.8]

def frameCallback( msg ):
    time = rospy.Time.now()

    if robot_view_pose == None:
        br.sendTransform( (0, 0, 0), (0, 0, 0, 1.0), time, ROBOT_FRAME, WORLD_FRAME )
    else:
        br.sendTransform( (robot_view_pose.position.x, robot_view_pose.position.y, robot_view_pose.position.z), (robot_view_pose.orientation.x, robot_view_pose.orientation.y, robot_view_pose.orientation.z, robot_view_pose.orientation.w), time, ROBOT_FRAME, WORLD_FRAME)

    if user_view_pose == None:
        br.sendTransform( (0, 0, 0), (0, 0, 0, 1.0), time, USER_FRAME, WORLD_FRAME )
    else:
        br.sendTransform( (user_view_pose.position.x, user_view_pose.position.y, user_view_pose.position.z), (user_view_pose.orientation.x, user_view_pose.orientation.y, user_view_pose.orientation.z, user_view_pose.orientation.w), time, USER_FRAME, WORLD_FRAME)



        
def processFeedback( feedback ):
    s = "Feedback from marker '" + feedback.marker_name
    s += "' / control '" + feedback.control_name + "'"

    mp = ""
    if feedback.mouse_point_valid:
        mp = " at " + str(feedback.mouse_point.x)
        mp += ", " + str(feedback.mouse_point.y)
        mp += ", " + str(feedback.mouse_point.z)
        mp += " in frame " + feedback.header.frame_id

    # if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
    #     rospy.loginfo( s + ": button click" + mp + "." )
    # elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
    #     rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
    if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        rospy.loginfo( s + ": pose changed")

        if feedback.marker_name == "robot_view":
            global robot_view_pose
            robot_view_pose = feedback.pose
        elif feedback.marker_name == "user_view":
            global user_view_pose
            user_view_pose = feedback.pose
        else:
            rospy.logerr("Invalid marker name")
            return

        print feedback.pose

# TODO
#          << "\nposition = "
#          << feedback.pose.position.x
#          << ", " << feedback.pose.position.y
#          << ", " << feedback.pose.position.z
#          << "\norientation = "
#          << feedback.pose.orientation.w
#          << ", " << feedback.pose.orientation.x
#          << ", " << feedback.pose.orientation.y
#          << ", " << feedback.pose.orientation.z
#          << "\nframe: " << feedback.header.frame_id
#          << " time: " << feedback.header.stamp.sec << "sec, "
#          << feedback.header.stamp.nsec << " nsec" )
    # elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
    #     rospy.loginfo( s + ": mouse down" + mp + "." )
    # elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
    #     rospy.loginfo( s + ": mouse up" + mp + "." )
    server.applyChanges()

def alignMarker( feedback ):
    pose = feedback.pose

    pose.position.x = round(pose.position.x-0.5)+0.5
    pose.position.y = round(pose.position.y-0.5)+0.5

    rospy.loginfo( feedback.marker_name + ": aligning position = " + str(feedback.pose.position.x) + "," + str(feedback.pose.position.y) + "," + str(feedback.pose.position.z) + " to " +
                                                                     str(pose.position.x) + "," + str(pose.position.y) + "," + str(pose.position.z) )

    server.setPose( feedback.marker_name, pose )
    server.applyChanges()

def rand( min_, max_ ):
    return min_ + random()*(max_-min_)

def makeFrustum( msg, frustum_depth=1.0, color=[255, 255, 255, 1.0] ):
    marker = Marker()

    marker.type = Marker.LINE_LIST
    marker.scale.x = 0.01

    z = frustum_depth
    fx = camera_params.CAMERA_Fx
    fy = camera_params.CAMERA_Fy
    w = camera_params.IMAGE_CROP_X_END - camera_params.IMAGE_CROP_X
    h = camera_params.IMAGE_CROP_Y_END - camera_params.IMAGE_CROP_Y

    x_dist_half = (z * w) / (2. * fx)
    y_dist_half = (z * h) / (2. * fy)
    
    origin = Point()
    origin.x = 0.
    origin.y = 0.
    origin.z = 0.

    top_left = Point()
    top_left.x = - x_dist_half
    top_left.y = - y_dist_half
    top_left.z = z

    top_right = Point()
    top_right.x =   x_dist_half
    top_right.y = - y_dist_half
    top_right.z = z

    bottom_left = Point()
    bottom_left.x = - x_dist_half
    bottom_left.y =   y_dist_half
    bottom_left.z = z

    bottom_right = Point()
    bottom_right.x =  x_dist_half
    bottom_right.y =  y_dist_half
    bottom_right.z = z

    marker.points.append(origin)
    marker.points.append(top_left)
    marker.points.append(top_left)
    marker.points.append(top_right)

    marker.points.append(origin)
    marker.points.append(top_right)
    marker.points.append(top_right)
    marker.points.append(bottom_right)
    
    marker.points.append(origin)
    marker.points.append(bottom_right)
    marker.points.append(bottom_right)
    marker.points.append(bottom_left)

    marker.points.append(origin)
    marker.points.append(bottom_left)
    marker.points.append(bottom_left)
    marker.points.append(top_left)

    marker.color.r = color[0]/255.
    marker.color.g = color[1]/255.
    marker.color.b = color[2]/255.
    marker.color.a = color[3]

    return marker

def makeFrustumControl( msg, frustum_depth, color ):
    control =  InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append( makeFrustum(msg, frustum_depth, color) )
    msg.controls.append( control )
    return control

def saveMarker( int_marker ):
  server.insert(int_marker, processFeedback)


#####################################################################
# Marker Creation

def make6DofMarker( fixed, interaction_mode, pose, name, desc, show_6dof, frustum_depth, color):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "odom_combined"
    int_marker.pose = pose
    int_marker.scale = 1

    int_marker.name = name
    int_marker.description = desc

    # insert a box
    makeFrustumControl(int_marker, frustum_depth, color)
    int_marker.controls[0].interaction_mode = interaction_mode

    if fixed:
        int_marker.name += "_fixed"
        int_marker.description += "\n(fixed orientation)"

    if interaction_mode != InteractiveMarkerControl.NONE:
        control_modes_dict = { 
                          InteractiveMarkerControl.MOVE_3D : "MOVE_3D",
                          InteractiveMarkerControl.ROTATE_3D : "ROTATE_3D",
                          InteractiveMarkerControl.MOVE_ROTATE_3D : "MOVE_ROTATE_3D" }
        int_marker.name += "_" + control_modes_dict[interaction_mode]
        int_marker.description = "3D Control"
        if show_6dof: 
          int_marker.description += " + 6-DOF controls"
        int_marker.description += "\n" + control_modes_dict[interaction_mode]
    
    if show_6dof: 
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

    server.insert(int_marker, processFeedback)
    menu_handler.apply( server, int_marker.name )


if __name__=="__main__":
    rospy.init_node("basic_controls")

    br = TransformBroadcaster()
    
    # create a timer to update the published transforms
    rospy.Timer(rospy.Duration(1./50), frameCallback)

    server = InteractiveMarkerServer("basic_controls")

    menu_handler.insert( "First Entry", callback=processFeedback )
    menu_handler.insert( "Second Entry", callback=processFeedback )
    sub_menu_handle = menu_handler.insert( "Submenu" )
    menu_handler.insert( "First Entry", parent=sub_menu_handle, callback=processFeedback )
    menu_handler.insert( "Second Entry", parent=sub_menu_handle, callback=processFeedback )

    make6DofMarker( False, InteractiveMarkerControl.NONE, pose=robot_view_pose, name="robot_view", desc="Robot View", show_6dof=True, frustum_depth=camera_params.ROBOT_VIEW_FRUSTUM_DEPTH, color=ROBOT_FRUSTUM_COLOR) 
        
    make6DofMarker( False, InteractiveMarkerControl.NONE, pose=user_view_pose, name="user_view", desc="User View", show_6dof=True, frustum_depth=camera_params.USER_VIEW_FRUSTUM_DEPTH, color=USER_FRUSTUM_COLOR)    
    
    server.applyChanges()

    rospy.spin()
