#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import xml.etree.ElementTree as ET
import actionlib
import action_controller.msg
import copy
import numpy as np
import os

def get_bboxes_from_xml(xml_path):
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()

    bboxes = []
    for obj in root.findall('object'):
        name = obj.get('name')
        bbox_tmp = obj.get('bndbox')
        bbox = [bbox_tmp.get('xmin'), bbox_tmp.get('ymin'), bbox_tmp.get('xmax'), bbox_tmp.get('ymax')]
        bboxes.append(bbox)
    
    rospy.loginfo("bboxes found: {}".format(bboxes))
    return bboxes

def ground_img_with_bbox(img, bboxes):

    # load image, extract and store feature vectors for each bounding box
    img_msg = CvBridge().cv2_to_imgmsg(img, "rgb8")
    goal = action_controller.msg.DenseRefexpLoadBBoxesActionGoal(input = img_msg, boxes = bboxes)
    load_client.send_goal(goal)
    load_client.wait_for_result()  
    load_result = load_client.get_result()

    rospy.loginfo("captions: {}".format(load_result.captions))
    rospy.loginfo("scores: {}".format(load_result.scores))

if __name__ == '__main__':
    try:
        rospy.init_node('Grounding', anonymous=True)

        # wait for action servers to show up
        # if you are stuck here, that means the servers are not ready
        # or your network connection is broken
        load_client = actionlib.SimpleActionClient('dense_refexp_load_bboxes', action_controller.msg.DenseRefexpLoadBBoxesAction)
        rospy.loginfo("1. Waiting for dense_refexp_load_bboxes action server ...")
        load_client.wait_for_server()

        query_client = actionlib.SimpleActionClient('dense_refexp_query', action_controller.msg.DenseRefexpQueryAction)
        rospy.loginfo("2. Waiting for dense_refexp_query action server ...")
        query_client.wait_for_server()    
        
        rospy.loginfo("Ingress server found! Ready.")

        while not rospy.is_shutdown():
            try:
                # get image
                img_path = raw_input("Enter path to image: ")
                img = cv2.imread(path,cv2.IMREAD_COLOR)

                xml_path = os.path.splitext(img_path)[0] + '.xml'
                bboxes = get_bboxes_from_xml(xml_path)
            except Exception as e:
                rospy.logerr(e)
                
            ground_img_with_bbox(img, bboxes)

    except rospy.ROSInterruptException:
        pass