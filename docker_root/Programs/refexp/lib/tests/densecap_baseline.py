#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg
import copy
import os

import numpy as np

from argparse import ArgumentParser

# folder_path = '/home/mohit/Programs/densecap/webcam'
# save_path = '/home/mohit/Programs/referring-expressions/lib/tests'

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="folder_path", help="input folder with images")
parser.add_argument("-o", "--output", dest="save_path", help="output to save images")
parser.add_argument("--step", dest="step", action='store_true')

parser.set_defaults(step=False)

args = parser.parse_args()

folder_path = args.folder_path
save_path = args.save_path


def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)
    
    load_client = actionlib.SimpleActionClient('dense_refexp_load', action_controller.msg.DenseRefexpLoadAction)
    print ("Waiting for dense_refexp_load")
    load_client.wait_for_server()
    print ("Done")

    query_client = actionlib.SimpleActionClient('localize_query', action_controller.msg.LocalizeQueryAction)
    query_client.wait_for_server()    

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 1280, 720)

    metadata_path = folder_path + '/metadata.txt'
    with open(metadata_path, 'r') as m_file:

        for line in m_file:
            image_name, query = line.split("|||||")
             
            print "Processing " + image_name
            print "Query " + query

            path = folder_path + '/' + image_name

            # print path
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            msg_frame = CvBridge().cv2_to_imgmsg(img, "rgb8")
            goal_loc = action_controller.msg.LocalizeGoal(1, msg_frame)
            load_client.send_goal(goal_loc)
            load_client.wait_for_result()    
            loc_result = load_client.get_result()

            # unpack result
            boxes = np.reshape(loc_result.boxes, (-1, 4))     
            beam_length = len(list(boxes))      

            goal_query = action_controller.msg.LocalizeQueryGoal(query, beam_length, 6.0)
            query_client.send_goal(goal_query)
            query_client.wait_for_result()
            query_result = query_client.get_result()

            q_boxes = list(np.reshape(query_result.boxes, (-1, 4)))

            # visualize
            draw_img = img.copy()
            for idx in range(3):

                x1 = int(q_boxes[idx][0])
                y1 = int(q_boxes[idx][1])
                x2 = int(q_boxes[idx][0]+q_boxes[idx][2])
                y2 = int(q_boxes[idx][1]+q_boxes[idx][3])

                if idx == 0:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,0,255), 12)
                elif idx == 1:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,255,0), 6)
                else:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255,0,0), 2)

            path = save_path + '/' + image_name.replace('.png', '') + "_densecap_result.png"
            cv2.imwrite(path, draw_img)

            if args.step:
                cv2.imshow('result', draw_img)
                k = cv2.waitKey(0)
                #_ = raw_input('Continue: ')


if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass
