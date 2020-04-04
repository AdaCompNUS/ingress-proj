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

import ingress_srv.ingress_srv as ingress_srv

def get_bboxes_from_xml(xml_path):
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()

    bboxes = []
    for obj in root.findall('object'):
        bbox_tmp = obj.find('bndbox')
        bbox = []
        bbox.append(float(bbox_tmp.find('xmin').text))
        bbox.append(float(bbox_tmp.find('ymin').text))
        bbox.append(float(bbox_tmp.find('xmax').text) - float(bbox_tmp.find('xmin').text))
        bbox.append(float(bbox_tmp.find('ymax').text) - float(bbox_tmp.find('ymin').text))
        bboxes.append(bbox)

    rospy.loginfo("bboxes found: {}".format(bboxes))
    return bboxes


def ground_img_with_bbox(load_client, img, bboxes):

    # load image, extract and store feature vectors for each bounding box
    img_msg = CvBridge().cv2_to_imgmsg(img, "rgb8")
    goal = action_controller.msg.DenseRefexpLoadBBoxesGoal()
    goal.input = img_msg
    goal.boxes = bboxes
    load_client.send_goal(goal)
    load_client.wait_for_result()
    load_result = load_client.get_result()

    rospy.loginfo("ground_img_with_bbox, result received")
    rospy.loginfo("captions: {}".format(load_result.captions))
    rospy.loginfo("scores: {}".format(load_result.scores))



if __name__ == '__main__':
    try:
        rospy.init_node('Grounding', anonymous=True)

        ingress_service = ingress_srv.Ingress()

        while not rospy.is_shutdown():
            try:
                # get image
                img_path = raw_input("Enter path to image: ")
                img_path = "images/" + img_path
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                xml_path = os.path.splitext(img_path)[0] + '.xml'
                bboxes = get_bboxes_from_xml(xml_path)
            except Exception as e:
                rospy.logerr(e)

            expr = raw_input("Enter user expression: ")

            bboxes, top_idx, context_idxs, captions = ingress_service.ground_img_with_bbox(img, bboxes, expr)
            sem_captions, sem_probs, rel_captions, rel_probs = captions
            rospy.loginfo("Top bbox is {}".format(bboxes[top_idx]))
            rospy.loginfo("Top bbox self-referential caption: {}".format(sem_captions[top_idx]))
            rospy.loginfo("Top bbox relational caption: {}".format(rel_captions[top_idx]))

    except rospy.ROSInterruptException:
        pass