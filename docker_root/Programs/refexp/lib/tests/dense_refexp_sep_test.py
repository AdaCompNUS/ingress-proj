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
    
    client = actionlib.SimpleActionClient('dense_refexp_load', action_controller.msg.DenseRefexpLoadAction)
    print ("Waiting for dense_refexp_load")
    client.wait_for_server()
    print ("Done")

    # Single Tabel Test
    # path = '/home/mohitshridhar/Programs/densecap/webcam/table_medium.png'
    # path = '/home/mohit/Programs/densecap/webcam/table_medium_2.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/table_mult_dups.png'
    path = '/home/mohit/Programs/densecap/webcam/mult_obj_conf.png'  
    # path = '/home/mohit/Programs/densecap/webcam/apple_orange_tennis.jpg'  
    # path = '/home/mohit/Programs/densecap/webcam/fruits.jpg'
    # path = '/home/mohit/Programs/densecap/webcam/3_blue_cups_720p.png' 
    # path = '/home/mohit/Programs/densecap/webcam/3_color_cups.jpg' 
    # path = '/home/mohitshridhar/Programs/densecap/webcam/green_top_test.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/3_water_bottles.png'
    # path = '/home/mohitshridhar/Programs/densecap/webcam/3_blocks.png'  

    img = cv2.imread(path,cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "rgb8")
    goal = action_controller.msg.DenseRefexpLoadGoal(msg_frame)
    client.send_goal(goal)
    client.wait_for_result()  
    load_result = client.get_result()

    boxes = np.reshape(load_result.boxes, (-1, 4))      

    # Query test
    client = actionlib.SimpleActionClient('dense_refexp_query', action_controller.msg.DenseRefexpQueryAction)
    print ("Waiting for dense_refexp_query")
    client.wait_for_server()    

    incorrect_idxs = []
    
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('result', img.shape[1], img.shape[0])

    while True:
        # query = "the water bottle next to the green glass"
        query = raw_input('Search Query: ').lower()
        goal = action_controller.msg.DenseRefexpQueryGoal(query, incorrect_idxs)
        client.send_goal(goal)
        client.wait_for_result()
        query_result = client.get_result()

        top_idx = query_result.top_box_idx
        context_boxes_idxs = list(query_result.context_boxes_idxs)
        context_boxes_idxs.append(top_idx)
        is_ambiguous = query_result.is_ambiguous

        if is_ambiguous:
            print ("Ambiguous Grounding ....")
            print ("Did you mean: " + query_result.predicted_captions[0])

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

        # cv2.imshow('result', draw_img)
        cv2.imwrite('result.png', draw_img)
        #k = cv2.waitKey(0)

    return True


if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass
