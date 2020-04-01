#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg
import copy


import numpy as np


def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)

    # Single Tabel Test
    # path = '/home/mohitshridhar/Programs/densecap/webcam/table_medium.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/table_medium_2.png'
    path = '/home/mohitshridhar/Programs/densecap/webcam/table_mult_dups.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/mult_obj_conf.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/3_water_bottles.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/3_blocks.png'  


    # Query test
    client = actionlib.SimpleActionClient('dense_refexp_load_query', action_controller.msg.DenseRefexpLoadQueryAction)
    client.wait_for_server()    
    
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('result', img.shape[1], img.shape[0])

    while True:
        # query = "the water bottle next to the green glass"
        query = raw_input('Search Query: ').lower()
        msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
        incorrect_idxs = []

        goal = action_controller.msg.DenseRefexpLoadQueryGoal(msg_frame, query, incorrect_idxs)
        client.send_goal(goal)
        client.wait_for_result()
        query_result = client.get_result()

        boxes = np.reshape(query_result.boxes, (-1, 4)) 
        top_idx = query_result.top_box_idx
        context_boxes_idxs = list(query_result.context_boxes_idxs)
        context_boxes_idxs.append(top_idx)

        # visualize
        draw_img = img.copy()
        for (count, idx) in enumerate(context_boxes_idxs):

            x1 = int(boxes[idx][0])
            y1 = int(boxes[idx][1])
            x2 = int(boxes[idx][0]+boxes[idx][2])
            y2 = int(boxes[idx][1]+boxes[idx][3])

            if count == len(context_boxes_idxs)-1:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,0,255), 12)
            else:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,255,0), 2)

        #cv2.imshow('result', draw_img)
        cv2.imwrite('result.png', draw_img)
        k = cv2.waitKey(0)

    return True


if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass
