import rospy
import numpy as np

import camera_params
import utils

import unittest
import random

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
#from tf import transformations
from numpy.linalg import inv

import cv2
import tf
from cv_bridge import CvBridge, CvBridgeError
from tf import TransformListener

#############################################
# Parameters
#############################################

CAMERA_MATRIX = np.array([[camera_params.CAMERA_Fx, 0, camera_params.CAMERA_Cx],
                          [0, camera_params.CAMERA_Fy,  camera_params.CAMERA_Cy],
                          [0,           0,          1.0]])
CAMERA_MATRIX_INV = inv(CAMERA_MATRIX)

CAMERA_VIEW_FRAME = '/kinect2_rgb_optical_frame'
ROBOT_VIEW_FRAME = '/robot_view'
USER_VIEW_FRAME = '/user_view'

USER_SYNONYMS = ['mine', 'my', 'our', 'me']
ROBOT_SYNONYMS = ['your', 'yourself', 'yours', "you're"]

############################################

class Perspective(object):
    
    camera = 0
    robot = 1
    user = 2
    

tl = TransformListener()

def transformPoint(target_frame, point):

    while not rospy.is_shutdown():
        try:
            time = tl.getLatestCommonTime(target_frame, point.header.frame_id)
            point.header.stamp = time
            point_transformed = tl.transformPoint(target_frame, point)
            break
        except:
            print "Waiting for centroid point transforation during perspective correction"
            continue
            
    return point_transformed
    
def transform_bbox_perspective(img, bbox, centroid, world2target_4x4, max_width=1920, max_height=1080):
    
    # correct camera matrix for cropping
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    camera_matrix = CAMERA_MATRIX # copy default first
    camera_matrix[0,2] = img_w/2. # !! inaccurate !!
    camera_matrix[1,2] = img_h/2.

    # unpack bbox
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    # transform centroid to target coordinates
    centroid = transformPoint(CAMERA_VIEW_FRAME, centroid)
    centroid_world_frame_4x1 = np.array([[centroid.point.x, centroid.point.y, centroid.point.z, 1.0]]).T
    centroid_target_frame_4x1 = np.dot(world2target_4x4, centroid_world_frame_4x1)

    # # check if the point is behind the camera
    if centroid_target_frame_4x1[2,0] < 0:
        print ("Pespective Correction: centroid is behind the camera!")
        return None

    # # compute scale factor of bbox for re-renderin(Zithod)
    # wh_camera_frame_3x1 = np.dot(CAMERA_MATRIX_INV, np.array([[w/2. + camera_params.CAMERA_Cx, h/2. + camera_params.CAMERA_Cy, 1.]]).T)
    # wh_camera_frame_3x1_homora_frame_3x1 / wh_camera_frame_3x1[2,0]
    # scale_factor_w = centroid_target_frame_4x1[2,0] * wh_camera_frame_3x1_homo[0,0] 
    # scale_factor_h = centroid_target_frame_4x1[2,0] * wh_camera_frame_3x1_homo[1,0] 
    # scale_factor = centroid_target_frame_4x1[2,0] * (wh_camera_frame_3x1_homo[0,0] + wh_camera_frame_3x1_homo[1,0]) / 2.
    # print "Scale Factor: %f, %f" % (scale_factor_w, scale_factor_h)
    
    # reproject bbox
    reprojection_3x1 = np.dot(camera_matrix, centroid_target_frame_4x1[:3])
    u = reprojection_3x1[0,0] / reprojection_3x1[2,0]
    v = reprojection_3x1[1,0] / reprojection_3x1[2,0]

    # # wh_cam_4x1 = np.array([[scale_factor_w, scale_factor_h, centroid_target_frame_4x1[2,0], 1.]]).T
    # wh_cam_4x1 = np.array([[scale_factor_w, scale_factor_h, centroid_target_frame_4x1[2,0], 1.]]).T
    # wh_screen_3x1 = np.dot(CAMERA_MATRIX, wh_cam_4x1[:3])
    # half_w = wh_screen_3x1[0,0] / wh_screen_3x1[2, - rams.CAMERA_Cx
    # half_h = wh_screen_3x1[1creen_3x1[2,0] - camera_params.CAMERA_Cy

    # # convert to new x, y, w, h
    # return [int(u-half_w), int(v-half_h), int(2.*half_w), int(2.*half_h)]
    
    scale_factor = (centroid_world_frame_4x1[2,0] / centroid_target_frame_4x1[2,0])
    new_w = w * scale_factor
    new_h = h * scale_factor
    
    # clamp to edges
    # if u+new_w/2. > img_w:
    #     new_w = img_w - u-new_w/2.
    # if v+new_h/2. > img_h:
    #     new_h = img_h - v-new_h/2.

    return [int(u-new_w/2.), int(v-new_h/2.), int(new_w), int(new_h)]


def correct_response(query_str, syns, possesive_str):
    
    corrected_str = str(query_str)
    for syn in syns:
        corrected_str = corrected_str.replace(syn, possesive_str)
    return corrected_str

def resolve_perspective(query_str, cam2robot_4x4, cam2user_4x4):

    # default perspective
    perspective = Perspective.camera
    world2target_4x4 = np.identity(4)
    corrected_resp = str(query_str)

    if any(syn in query_str for syn in ROBOT_SYNONYMS):
        perspective = Perspective.robot
        world2target_4x4 = cam2robot_4x4
        corrected_resp = correct_response(query_str, ROBOT_SYNONYMS, "my")
    elif any(syn in query_str for syn in USER_SYNONYMS):
        perspective = Perspective.user
        world2target_4x4 = cam2user_4x4
        corrected_resp = correct_response(query_str, USER_SYNONYMS, "your")

    return perspective, world2target_4x4, corrected_resp


def get_corrected_response(query_str):
    
    cam2robot_4x4, cam2user_4x4 = get_transformations()
    persp, world2target_4x4, corr_resp = resolve_perspective(query_str, cam2robot_4x4, cam2user_4x4)
    
    return corr_resp


def render_reprojection(img, orig_bboxes, reproj_bboxes, colored_edges=False):

    orig_bboxes = utils.flip_xywh_bboxes(orig_bboxes)

    orig_img = img.copy()
    rendered_img = img.copy()
    
    colors = []
    # put black holes inside background:
    for orig_bbox in orig_bboxes:
        x1, y1, x2, y2 = orig_bbox[0], orig_bbox[1], orig_bbox[0]+orig_bbox[2], orig_bbox[1]+orig_bbox[3]

        # choose random color
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        colors.append(color)

        # clear orig bbox
        rendered_img[y1:y2,x1:x2,:] = 255
        if colored_edges: 
            cv2.rectangle(rendered_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=7)

    # copy over to new place
    for (idx, new_bbox) in enumerate(reproj_bboxes):
        orig_bbox = orig_bboxes[idx]
        x1, y1, x2, y2 = orig_bbox[0], orig_bbox[1], orig_bbox[0]+orig_bbox[2], orig_bbox[1]+orig_bbox[3]
        # print x1, y1, x2, y2
        new_w, new_h = new_bbox[2], new_bbox[3]

        orig_obj = orig_img[y1:y2,x1:x2,:]
        scaled_obj = cv2.resize(orig_obj, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
        new_x1, new_y1, new_x2, new_y2 = new_bbox[0], new_bbox[1], new_bbox[0]+new_bbox[2], new_bbox[1]+new_bbox[3]
        rendered_img[new_y1:new_y2,new_x1:new_x2,:] = scaled_obj.copy()
        color = colors[idx]
        if colored_edges: 
            cv2.rectangle(rendered_img, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), color, thickness=7)

        
    # cv2.imshow("result", rendered_img)
    # cv2.waitKey(0)
    
    return rendered_img


def get_transformations():
    
    cam2robot_4x4 = lookupTransformation(target_frame=ROBOT_VIEW_FRAME, source_frame=CAMERA_VIEW_FRAME)
    cam2user_4x4 = lookupTransformation(target_frame=USER_VIEW_FRAME, source_frame=CAMERA_VIEW_FRAME)

    return cam2robot_4x4, cam2user_4x4

def lookupTransformation(target_frame, source_frame):

    tl.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            tl.waitForTransform(target_frame, source_frame, now, rospy.Duration(4.0))
            (trans,rot) = tl.lookupTransform(target_frame, source_frame, now)
            return  np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
        except:
            print "Waiting.."


def correct_perspective(img, bboxes, centroids, query_str, colored_edges=False):
    
    cam2robot_4x4, cam2user_4x4 = get_transformations()
    persp, world2target_4x4, corr_resp = resolve_perspective(query_str, cam2robot_4x4, cam2user_4x4)
    
    print "\tPerspective: %d" % (persp) 
    
    if persp == Perspective.camera:
        return bboxes, img, persp

    reprojected_bboxes = [transform_bbox_perspective(img, bbox, centroids[idx], world2target_4x4)  for (idx, bbox) in enumerate(bboxes)]

    # check if reprojection was successful, otherwise return the original bboxes
    try:                        
        reproj_viz = render_reprojection(img, bboxes, reprojected_bboxes, colored_edges=colored_edges)
    except:
        rospy.logwarn("Perspective reprojection failed; returned original bboxes and img. Check your robot/camera/user transformations")
        return bboxes, img, persp

    return reprojected_bboxes, reproj_viz, persp


############################################
# Unit Tests
############################################

class TestPerspectives(unittest.TestCase):

    def test_perspective_correction(self):

        rospy.init_node("Testing")
        im_path = '/home/rls/Downloads/frame0000.jpg'
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)

        # x, y, w, h
        bboxes = np.array([[654, 591, 140, 159], 
                           [468, 581, 120, 148],
                           [898, 608, 120, 148]])

        # X, Y, Z in left_in
        obj_point1 = PointStamped()
        obj_point1.header.stamp = rospy.Time.now()
        obj_point1.header.frame_id = 'left_in'
        obj_point1.point.x = 0.36974678763
        obj_point1.point.y = 0.576438259091
        obj_point1.point.z = 0.0585531324144
        obj_point2 = PointStamped()
        obj_point2.header = obj_point1.header
        obj_point2.point.x = 0.36974678763 
        obj_point2.point.y = 0.576438259091 + 0.17
        obj_point2.point.z = 0.0585531324144
        obj_point3 = PointStamped()
        obj_point3.header = obj_point1.header
        obj_point3.point.x = 0.36974678763
        obj_point3.point.y = 0.576438259091 - 0.17
        obj_point3.point.z = 0.0585531324144
 
        centroids = np.array([obj_point1, obj_point2, obj_point3])
        query_str = "the red cup on your left"

        corrected_bboxes, viz = correct_perspective(img, bboxes, centroids, query_str)
        
        self.assertTrue(True)

#unittest.main()
