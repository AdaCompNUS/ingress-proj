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
        bbox_tmp = obj.find('bndbox')
        bboxes.append(float(bbox_tmp.find('xmin').text))
        bboxes.append(float(bbox_tmp.find('ymin').text))
        bboxes.append(float(bbox_tmp.find('xmax').text) - float(bbox_tmp.find('xmin').text))
        bboxes.append(float(bbox_tmp.find('ymax').text) - float(bbox_tmp.find('ymin').text))

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

    # ------------------------------------------------
    # visualization
    draw_img = img.copy()
    for idx in range(0, len(bboxes), 4):
        x1 = int(bboxes[idx])
        y1 = int(bboxes[idx + 1])
        x2 = int(x1 + bboxes[idx + 2])
        y2 = int(y1 + bboxes[idx + 3])

        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # add captions
        font = cv2.FONT_HERSHEY_DUPLEX
        # print("x1 {}, y1 {}".format(x1, y1))
        if y1 - 15 > 5:
            cv2.putText(draw_img, load_result.captions[int(idx // 4)],
                        (x1 + 6, y1 - 15), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(draw_img, load_result.captions[int(idx // 4)],
                        (x1 + 6, y1 + 5), font, 1, (255, 255, 255), 2)

    cv2.imwrite('./grounding_result.png', draw_img)


if __name__ == '__main__':
    try:
        rospy.init_node('Grounding', anonymous=True)

        # wait for action servers to show up
        # if you are stuck here, that means the servers are not ready
        # or your network connection is broken
        load_client = actionlib.SimpleActionClient(
            'dense_refexp_load_bboxes', action_controller.msg.DenseRefexpLoadBBoxesAction)
        rospy.loginfo("1. Waiting for dense_refexp_load_bboxes action server ...")
        load_client.wait_for_server()

        query_client = actionlib.SimpleActionClient(
            'dense_refexp_query', action_controller.msg.DenseRefexpQueryAction)
        rospy.loginfo("2. Waiting for dense_refexp_query action server ...")
        query_client.wait_for_server()

        rospy.loginfo("Ingress server found! Ready.")

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

            ground_img_with_bbox(load_client, img, bboxes)

    except rospy.ROSInterruptException:
        pass
