#!/usr/bin/env python

import rospy
from roslib import message
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import actionlib
import copy
import numpy as np
import tf
import math
from argparse import ArgumentParser
from datetime import datetime

import action_controller.msg

'''Settings'''
VALID_MIN_CLUSTER_SIZE = 500 # points
VALID_MIN_RELEVANCY_SCORE = 0.05

DEBUG = False

def dbg_print(text):
    if DEBUG:
        print(text)

class Ingress():
    def __init__(self):
        # wait for grounding load server
        self._load_client = actionlib.SimpleActionClient('dense_refexp_load', action_controller.msg.DenseRefexpLoadAction)
        rospy.loginfo("DemoManager: 1. Waiting for dense_refexp_load action server ...")
        self._self_captions = []
        self._load_client.wait_for_server()

        # wait for relevancy clustering server
        self._relevancy_client = actionlib.SimpleActionClient('relevancy_clustering', action_controller.msg.RelevancyClusteringAction)
        rospy.loginfo("DemoManager: 2. Waiting for relevancy_clustering action server ...")
        self._relevancy_client.wait_for_server()    

        # wait for grounding query server
        self._query_client = actionlib.SimpleActionClient('boxes_refexp_query', action_controller.msg.BoxesRefexpQueryAction)
        rospy.loginfo("DemoManager: 3. Waiting for boxes_refexp_query action server ...")
        self._rel_captions = []
        self._query_client.wait_for_server()

        # publisher for Ingress grounding results
        self._grounding_result_pub = rospy.Publisher('ingress/dense_refexp_result', Image, queue_size=10, latch=True)

        # demo ready!
        rospy.loginfo("Ingress ready!")

    def _ground_load(self, image):
        '''
        load the image into the grounding model
        input: image
        output: bounding boxes
        '''

        goal = action_controller.msg.DenseRefexpLoadGoal(image)
        rospy.loginfo("_ground_load: sending goal")
        self._load_client.send_goal(goal)
        rospy.loginfo("_ground_load: waiting for result")
        self._load_client.wait_for_result()
        rospy.loginfo("_ground_load: getting result")
        load_result = self._load_client.get_result()

        boxes = np.reshape(load_result.boxes, (-1, 4)) 
        losses = np.array(load_result.scores)     
        self._ungrounded_captions = np.array(load_result.captions)

        return boxes, losses

    def _pomdp_preprocess(self, relevancy_result, query_result, context_boxes_idxs, query):
        '''
        helper function to organize the data for the POMDP disambiguation model
        '''

        all_orig_idx = relevancy_result.all_orig_idx

        semantic_softmax = np.array(relevancy_result.softmax_probs)
        semantic_softmax_orig_idxs = [-1.] * len(all_orig_idx) # this is disgusting.... puke! You should be ashamed of yourself Mohit.
        for count, idx in enumerate(all_orig_idx):
            semantic_softmax_orig_idxs[idx] = semantic_softmax[count]
            
        top_idxs = context_boxes_idxs
        spatial_captions = query_result.predicted_captions
        spatial_softmax = np.array(query_result.probs) / np.sum(np.array(query_result.probs))

        dbg_print(all_orig_idx)
        dbg_print(semantic_softmax_orig_idxs)
        dbg_print(semantic_softmax)
        dbg_print(spatial_softmax)
        dbg_print(top_idxs)

        sem_captions = []
        sem_probs = []
        rel_captions = []
        rel_probs = []
        for (count, idx) in enumerate(top_idxs):
            if idx in context_boxes_idxs:
                sem_captions.append(self._ungrounded_captions[idx])
                sem_probs.append(semantic_softmax_orig_idxs[idx])
                if count < len(spatial_softmax) and count < len(spatial_captions):
                    rel_captions.append(spatial_captions[count].replace(query, '').replace('.', '').strip())
                    rel_probs.append(spatial_softmax[count])
                else:
                    rospy.logwarn("Semantic and Spatial captions+prob lengths dont match!")
                
        return [sem_captions, sem_probs, rel_captions, rel_probs]


    def _ground_query(self, expr, boxes):
        '''
        query a specific expression with the grounding model
        input: expr
        output: best bounding box index, other box indexes sorted by grounding scores 
        '''

        query = expr
        rospy.loginfo("Grounding Input String: " + query)

        # send expression for relevancy clustering 
        incorrect_idxs = []
        
        # dialog storage
        self._relevancy_result = None
        self._context_boxes_idxs =  None
        
        # relevancy
        relevancy_goal = action_controller.msg.RelevancyClusteringGoal(query, incorrect_idxs)
        self._relevancy_client.send_goal(relevancy_goal)
        self._relevancy_client.wait_for_result()
        self._relevancy_result = self._relevancy_client.get_result()
        # selection_orig_idx = self._relevancy_client.get_result().selection_orig_idx
        selection_orig_idx = self._relevancy_result.selection_orig_idx
        rospy.loginfo("orig index {}".format(selection_orig_idx))

        # clean
        # num_points_arr = [list(self._segment_pc(boxes[idx]))[1] for idx in selection_orig_idx]
        # print num_points_arr
        # pruned_selection_orig_idx = [i for i, idx in enumerate(selection_orig_idx) if num_points_arr[i] > VALID_MIN_CLUSTER_SIZE]
        # selection_orig_idx = list(np.take(selection_orig_idx, pruned_selection_orig_idx))
        # self._relevancy_result.selection_orig_idx = list(selection_orig_idx)

        ## clean by threshold
        semantic_softmax = np.array(self._relevancy_result.softmax_probs)

        all_orig_idx = self._relevancy_result.all_orig_idx
        semantic_softmax_orig_idxs = [-1.] * len(all_orig_idx) # this is disgusting.... puke! You should be ashamed of yourself Mohit.
        for count, idx in enumerate(all_orig_idx):
            semantic_softmax_orig_idxs[idx] = semantic_softmax[count]

        pruned_selection_orig_idx = [i for i, idx in enumerate(selection_orig_idx) if semantic_softmax_orig_idxs[idx] > VALID_MIN_RELEVANCY_SCORE]
        selection_orig_idx = list(np.take(selection_orig_idx, pruned_selection_orig_idx))
        self._relevancy_result.selection_orig_idx = selection_orig_idx
        rospy.loginfo("pruned index {}".format(selection_orig_idx))

        if len(selection_orig_idx) == 0:
            rospy.logwarn("Ingress_srv: no object detected")
            return 0, [], None

        # ground the query
        selected_boxes = np.take(boxes, selection_orig_idx, axis=0)
        query_goal = action_controller.msg.BoxesRefexpQueryGoal(query, np.array(selected_boxes).ravel(), selection_orig_idx, incorrect_idxs)
        self._query_client.send_goal(query_goal)
        self._query_client.wait_for_result()
        query_result = self._query_client.get_result()

        # grounding results: indexes of most likely bounding boxes
        top_idx = query_result.top_box_idx
        context_boxes_idxs = [top_idx] + list(query_result.context_boxes_idxs)
        self._context_boxes_idxs = list(context_boxes_idxs)

        # preprocess data for POMDP model
        pomdp_init_data = self._pomdp_preprocess(self._relevancy_result, query_result, context_boxes_idxs, query)

        return top_idx, context_boxes_idxs, pomdp_init_data

    def _reground_query(self, expr, boxes):
        '''
        ground new expr but with same boxes (for dialog interaction)
        NOTE: must be called after _ground_query
        '''

        query = expr
        if self._relevancy_result == None:
            rospy.logerr("Grounding server wasn't initialized. Can't reground without initial grounding from _ground_query function")
            return None

        rospy.loginfo("Regrounding Input String: " + query)

        if query == '':
            return None, None, [None, [1.0]*len(self._relevancy_result.selection_orig_idx), None, [1.0]*len(self._relevancy_result.selection_orig_idx)]

        # send expression for relevancy clustering 
        incorrect_idxs=[]
        relevancy_goal = action_controller.msg.RelevancyClusteringGoal(query, incorrect_idxs)
        self._relevancy_client.send_goal(relevancy_goal)
        self._relevancy_client.wait_for_result()
        new_relevancy_result = self._relevancy_client.get_result()
        # selection_orig_idx = self._relevancy_client.get_result().selection_orig_idx   

        # ground the new query with old relevancy clustering
        selection_orig_idx = self._relevancy_result.selection_orig_idx
        selected_boxes = np.take(boxes, selection_orig_idx, axis=0)
        query_goal = action_controller.msg.BoxesRefexpQueryGoal(query, np.array(selected_boxes).ravel(), selection_orig_idx, incorrect_idxs)
        self._query_client.send_goal(query_goal)
        self._query_client.wait_for_result()
        query_result = self._query_client.get_result()

        # grounding results: indexes of mostly bounding boxes
        top_idx = query_result.top_box_idx
        context_boxes_idxs = [top_idx] + list(query_result.context_boxes_idxs)

        # preprocess data for POMDP model
        pomdp_init_data = self._pomdp_preprocess(new_relevancy_result, query_result, self._context_boxes_idxs, query)

        return top_idx, context_boxes_idxs, pomdp_init_data

    def _publish_grounding_result(self, boxes, context_boxes_idxs):
        '''
        debugging visualization of the bounding box outputs from the grounding model
        '''

        ## Input validity check
        if len(boxes) == 0 or len(context_boxes_idxs) == 0:
            return

        # Deepcopy is important here, otherwise self._img_msg will be contaminated.
        draw_img = copy.deepcopy(CvBridge().imgmsg_to_cv2(self._img_msg)) 
        for (count, idx) in enumerate(context_boxes_idxs):

            x1 = int(boxes[idx][0])
            y1 = int(boxes[idx][1])
            x2 = int(boxes[idx][0] + boxes[idx][2])
            y2 = int(boxes[idx][1] + boxes[idx][3])

            if count == 0: # top idx
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,255,0), 15)
            elif count < len(context_boxes_idxs):
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,0,255), 11)

        # cv2.imshow('image', draw_img)
        # cv2.waitKey(50)
        # cv2.destroyAllWindows()
        result_img = CvBridge().cv2_to_imgmsg(draw_img)
        self._grounding_result_pub.publish(result_img)
        rospy.loginfo("grounding result published")

    def ground(self, image, expr):
        '''
        run the full grounding pipeline
        @param image, ros/sensor_msgs/Image, image
        @param expr, string, user input to describe the target object
        @return. the most likely bounding boxes. 
        '''
        
        ## Input validity check
        if image is None:
            rospy.logerr("DemoMovo/ground: image is none, skip")
            return None

        self._img_msg = image
        boxes, losses = self._ground_load(image)
        if expr == '':
            rospy.loginfo("Ingress: empty query string received, returning ungrounded result")
            top_idx = 0
            context_idxs = [i for i in range(0, len(boxes)) if losses[i] > 0]
            pomdp_init_data = ([self._ungrounded_captions[i] for i in context_idxs], [losses[i] for i in context_idxs], [], None)
            return boxes, top_idx, context_idxs, pomdp_init_data

        top_idx, context_idxs, pomdp_init_data = self._ground_query(expr, boxes)
        # context_idxs.append(top_idx)

        self._publish_grounding_result(boxes, context_idxs) # visualization of RViz
        return boxes, top_idx, context_idxs, pomdp_init_data

if __name__ == '__main__':
    rospy.init_node('ingress_ros')

    try:
        ingress = Ingress()   
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ingress exit!!")
