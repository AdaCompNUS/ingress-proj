from meteor_score import Meteor
from process_dataset import GoogleRefExp, UNCRefExp
from experiment_settings import get_experiment_paths, get_experiment_config
from shared_utils import UNK_IDENTIFIER, image_feature_extractor, get_encoded_line
from language_model import LanguageModel, MILContextLanguageModel, gen_stats
from sklearn.cluster import KMeans
from matplotlib.pyplot import imshow, pause
import numpy as np
from collections import defaultdict, OrderedDict
import matplotlib.image as mpimg
import time
import copy
import ingress_msgs.srv
import ingress_msgs.msg
import actionlib
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
from sensor_msgs.msg import Image
import rospy
import sys
import random
import json
import argparse
import h5py
#import ipdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#from adjustText import adjust_text


# plt.ion()


DEBUG_PRINT = False
VISUALIZE = False

AMBUGUITY_THRESHOLD = 0.1
NAME_REPLACEMENT_METEOR_SCORE_THRESHOLD = 0.1


class ComprehensionExperiment:
    def __init__(self, language_model, dataset, image_ids=None):
        self.lang_model = language_model

    def extract_image_features(self, experiment_paths, proposal_source, output_h5_file):
        img_bbox_pairs = []
        image_infos = {}
        processed_pairs = set()
        for image_id in self.images:
            image = self.dataset.loadImgs(image_id)[0]
            image_infos[image_id] = image
            if proposal_source != 'gt':
                bboxes = [cand['bounding_box'] for cand in image['region_candidates']]
            else:
                anns = self.dataset.coco.imgToAnns[image_id]
                bboxes = [ann['bbox'] for ann in anns]
            if len(bboxes) == 0:
                continue
            img_wd = int(image['width'])
            img_ht = int(image['height'])
            bboxes += [[0, 0, img_wd-1, img_ht-1]]
            for bbox in bboxes:
                if str((image_id, bbox)) not in processed_pairs:
                    processed_pairs.add(str((image_id, bbox)))
                    img_bbox_pairs.append((image_id, bbox))

        image_feature_extractor.extract_features_for_bboxes(self.dataset.image_root, image_infos,
                                                            img_bbox_pairs, output_h5_file, feature_layer='fc7')

    def extract_single_densecap_features(self, img, visualize=False):

        # send image to dense localizer (TODO: make this more efficient)
        msg_frame = CvBridge().cv2_to_imgmsg(img, "rgb8")
        goal = ingress_msgs.msg.LocalizeGoal(1, msg_frame)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()
        loc_result = client.get_result()

        # unpack result
        boxes = np.reshape(loc_result.boxes, (-1, 4))
        fc7_whole = np.reshape(loc_result.fc7_img, (-1, 4096))
        fc7_feats = np.reshape(loc_result.fc7_vecs, (-1, 4096))
        obj_classes = np.zeros((len(boxes), 10))
        captions = loc_result.captions
        scores = loc_result.scores
        # word_probs = loc_result.word_probs

        # add whole image as context
        boxes = np.vstack([boxes, np.array([0, 0, img.shape[1]-1, img.shape[0]-1])])

        # plot localization results
        if visualize:
            plt.cla()
            plt.imshow(img)
            plt.title('localization')
            for (box_id, box) in enumerate(boxes):
                plt.gca().add_patch(plt.Rectangle(
                    (box[0], box[1]), box[2], box[3], fill=False, edgecolor='b', linewidth=2))
            plt.axis('off')
            pause(1.0)

        # extract CNN object features fron VGG16
        # fc7_feats, fc7_whole, obj_classes = image_feature_extractor.extract_features_from_img(path, boxes)
        boxes = np.delete(boxes, (len(boxes)-1), axis=0)

        return boxes, fc7_whole, fc7_feats, obj_classes, captions, scores

    def extract_box_densecap_features(self, img, bboxes, visualize=False):

        # ravel boxes
        r_boxes = []
        for bbox in copy.copy(bboxes):
            r_boxes.append(int(bbox[0]+bbox[2]/2+1))  # xc
            r_boxes.append(int(bbox[1]+bbox[3]/2+1))  # yc
            r_boxes.append(int(bbox[2]))  # w
            r_boxes.append(int(bbox[3]))  # h

        # send image to dense localizer
        extract_client = actionlib.SimpleActionClient(
            'extract_features', ingress_msgs.msg.ExtractFeaturesAction)
        extract_client.wait_for_server()
        msg_frame = CvBridge().cv2_to_imgmsg(img, "rgb8")
        goal = ingress_msgs.msg.ExtractFeaturesGoal(1, msg_frame, r_boxes)
        extract_client.send_goal(goal, done_cb=done_cb)
        extract_client.wait_for_result()
        loc_result = extract_client.get_result()

        # unpack result
        fc7_feats = np.reshape(loc_result.fc7_vecs, (-1, 4096))
        captions = loc_result.captions
        scores = loc_result.scores

        # plot localization results
        if visualize:
            plt.figure(4)
            plt.cla()
            plt.imshow(img)
            plt.title('localization')
            for (box_id, box) in enumerate(bboxes):
                plt.gca().add_patch(plt.Rectangle(
                    (box[0], box[1]), box[2], box[3], fill=False, edgecolor='b', linewidth=2))
            plt.axis('off')
            pause(0.1)

        return fc7_feats, captions, scores


class MILContextComprehension(ComprehensionExperiment):
    def __init__(self, language_model, dataset, image_ids=None, no_confidence_prune=False, disambiguate=False, min_cluster_size=False):
        ComprehensionExperiment.__init__(self, language_model, dataset, image_ids=image_ids)

        self._load_service = actionlib.SimpleActionServer(
            'dense_refexp_load', ingress_msgs.msg.DenseRefexpLoadAction, execute_cb=self.load_image, auto_start=False)
        self._load_bboxes_service = actionlib.SimpleActionServer(
            'dense_refexp_load_bboxes', ingress_msgs.msg.DenseRefexpLoadBBoxesAction, execute_cb=self.load_image_with_boxes, auto_start=False)
        self._query_service = actionlib.SimpleActionServer(
            'dense_refexp_query', ingress_msgs.msg.DenseRefexpQueryAction, execute_cb=self.refexp_query, auto_start=False)

        self._relevancy_clustering_service = actionlib.SimpleActionServer(
            'relevancy_clustering', ingress_msgs.msg.RelevancyClusteringAction, execute_cb=self.relevancy_clustering, auto_start=False)
        self._boxes_refexp_query_service = actionlib.SimpleActionServer(
            'boxes_refexp_query', ingress_msgs.msg.BoxesRefexpQueryAction, execute_cb=self.boxes_refexp_query, auto_start=False)

        self._meteor_service = rospy.Service(
            'meteor_score', ingress_msgs.srv.MeteorScore, self._calc_meteor_score)

        self._load_service.start()
        self._load_bboxes_service.start()
        self._query_service.start()

        self._relevancy_clustering_service.start()
        self._boxes_refexp_query_service.start()

        self._combined_semaphore = False
        self._meteor = Meteor()

        self._no_confidence_prune = no_confidence_prune
        self._disambiguate = disambiguate
        self._min_cluster_size = min_cluster_size

    def _calc_meteor_score(self, req):
        score = self._meteor.score(req.ref, req.tar)
        return ingress_msgs.srv.MeteorScoreResponse(score)

    def box2box_distance(self, boxA, boxB):
        centerA_x = boxA[0] + 0.5 * boxA[2]
        centerA_y = boxA[1] + 0.5 * boxA[3]
        centerB_x = boxB[0] + 0.5 * boxB[2]
        centerB_y = boxB[1] + 0.5 * boxB[3]

        return np.sqrt((centerB_x - centerA_x)**2 + (centerB_y - centerA_y)**2)

    def unpack_and_prune(self, query_result, incorrect_idxs, query):
        # unpack result
        q_boxes = np.reshape(query_result.boxes, (-1, 4))
        q_fc7_feats = np.reshape(query_result.fc7_vecs, (-1, 4096))
        q_captioning_losses = query_result.captioning_losses
        q_similarity_ranks = query_result.meteor_ranks
        q_orig_idx = query_result.orig_idx
        # q_similarity_score = query_result.meteor_scores
        # q_similarity_score = [self._doc2vec.cosine_sim(query, caption) for caption in self.o_captions]
        # q_similarity_score = [nltk.translate.bleu_score.sentence_bleu([query.split()], caption.split()) for caption in self.o_captions]
        for c_idx in range(len(self.o_captions)):
            print("unpack and prune 1", query, self.o_captions[q_orig_idx[c_idx]])
        q_similarity_score = [self._meteor.score(
            query, self.o_captions[q_orig_idx[c_idx]]) for c_idx in range(len(self.o_captions))]

        # softmax for densecap
        softmax_inputs = np.array(q_captioning_losses)
        dense_softmax = np.zeros_like(q_captioning_losses)
        shifted_inputs = softmax_inputs.max() - softmax_inputs
        exp_outputs = np.exp(shifted_inputs)
        exp_outputs_sum = exp_outputs.sum()
        if math.isnan(exp_outputs_sum):
            dense_softmax = exp_outputs * float('nan')
        assert exp_outputs_sum > 0
        if math.isinf(exp_outputs_sum):
            dense_softmax = np.zeros_like(exp_outputs)
        eps_sum = 1e-20
        dense_softmax = exp_outputs / max(exp_outputs_sum, eps_sum)

        # for idx, cap in enumerate(self.o_captions):
        #   print str(dense_softmax[idx])

        # debug: print METEOR scores
        print "\nQuery: %s" % (query)
        for c, cap in enumerate(self.o_captions):
            print "loss: %f, softmax: %f, meteor: %f - '%s'" % (
                q_captioning_losses[c], dense_softmax[c], q_similarity_score[c], self.o_captions[q_orig_idx[c]])

        # remove incorrect selections from previous query
        if len(incorrect_idxs) > 0:
            remove_idxs = []
            for o in range(len(q_orig_idx)):
                if q_orig_idx[o] in incorrect_idxs:
                    remove_idxs.append(o)

            q_boxes = np.delete(q_boxes, remove_idxs, axis=0)
            q_fc7_feats = np.delete(q_fc7_feats, remove_idxs, axis=0)
            q_captioning_losses = np.delete(q_captioning_losses, remove_idxs, axis=0)
            q_similarity_ranks = np.delete(q_similarity_ranks, remove_idxs, axis=0)
            q_orig_idx = np.delete(q_orig_idx, remove_idxs, axis=0)
            q_similarity_score = np.delete(q_similarity_score, remove_idxs, axis=0)

        return q_boxes, q_fc7_feats, q_captioning_losses, q_similarity_ranks, q_orig_idx, q_similarity_score, dense_softmax

    def k_means(self, q_captioning_losses, q_similarity_score, q_orig_idx, captions, query, slice_point=10, max_cluster_size=5, norm_meteor=True, rev_loss=True, visualize=False, save_path='./clustering.png'):

        # sort out top k results
        k = slice_point
        k = min(k, len(q_captioning_losses))
        all_combined = np.vstack((q_captioning_losses, q_similarity_score)).T
        all_euclidean_error = [np.linalg.norm(np.array([1., 1.])-i) for i in all_combined]
        all_top_k = np.array(all_euclidean_error).argsort()[:k]

        slice_length = k
        q_captioning_losses = np.take(q_captioning_losses, all_top_k, axis=0)
        q_orig_idx = np.take(q_orig_idx, all_top_k, axis=0)
        q_similarity_score = np.take(q_similarity_score, all_top_k, axis=0)

        # clusttering
        if rev_loss:
            cap_loss = [1.-(float(i)-min(q_captioning_losses))/(max(q_captioning_losses) -
                                                                min(q_captioning_losses)) for i in q_captioning_losses]
        else:
            cap_loss = [(float(i)-min(q_captioning_losses))/(max(q_captioning_losses) -
                                                             min(q_captioning_losses)) for i in q_captioning_losses]
            # cap_loss = [float(i) for i in q_captioning_losses]

        if norm_meteor:
            similarity_scores = [(float(i)-min(q_similarity_score)) /
                                 (max(q_similarity_score)-min(q_similarity_score)) for i in q_similarity_score]
        else:
            similarity_scores = [float(i) for i in q_similarity_score]

        # cap_loss = [1.-float(i) for i in q_captioning_losses]

        combined_score = np.vstack((cap_loss, similarity_scores)).T
        if len(combined_score) <= 0 or np.any(np.isnan(combined_score)) or np.any(np.isinf(combined_score)):
            print "Sample size too small for clustering"
            return None

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(combined_score)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        #max_k = max_cluster_size
        max_k = min(max_cluster_size, len(q_captioning_losses))
        min_k = min(self._min_cluster_size, len(q_captioning_losses))
        cluster_euclidean_error = [np.linalg.norm(np.array([1., 1.])-i) for i in combined_score]
        top_k = np.array(cluster_euclidean_error).argsort()[:max_k]
        bottom_k = np.array(cluster_euclidean_error).argsort()[:min_k]

        # if greater than max_k choices
        selected_cluster_label = labels[top_k[0]]
        for i in range(len(labels)):
            if labels[i] == selected_cluster_label:
                if i not in top_k:
                    labels[i] = 1 - selected_cluster_label

        # if fewer than min_k choices
        for i in range(len(labels)):
            if labels[i] != selected_cluster_label:
                if i in bottom_k:
                    labels[i] = selected_cluster_label

        # visualize clustering
        if False:  # visualize:
            colors = ["g.", "r."]
            fig, ax = plt.subplots()
            ax.scatter(cap_loss, similarity_scores)
            texts = []
            orig_cap = [None] * slice_length
            for (count, idx) in enumerate(q_orig_idx):
                orig_cap[count] = captions[idx]
            for i in range(slice_length):
                plt.plot(cap_loss[i], similarity_scores[i], colors[labels[i]], markersize=10)
                plt.annotate(orig_cap[i], (cap_loss[i], similarity_scores[i]))
            # for i, txt in enumerate( orig_cap ):
            #   texts.append(ax.text(cap_loss[i], similarity_scores[i], orig_cap[i]))
            # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

            plt.title(query)
            plt.savefig(save_path, pad_inches=0, bbox_inches='tight')
            # plt.show()

        return np.where(labels == selected_cluster_label)[0]

    def load_image(self, goal):

        self._prev_query_breakdowns = []
        self._query_breakdowns = []

        self.img = CvBridge().imgmsg_to_cv2(goal.input, desired_encoding="passthrough")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        extraction_start = time.time()
        self.o_boxes, self.o_fc7_whole, self.o_fc7_feats, self.o_obj_classes, self.o_captions, self.o_scores = self.extract_single_densecap_features(
            self.img, visualize=False)
        extraction_end = time.time()
        print "Extraction Time: %f" % (extraction_end - extraction_start)

        result = ingress_msgs.msg.DenseRefexpLoadResult()
        result.captions = self.o_captions
        result.scores = self.o_scores
        result.boxes = self.o_boxes.ravel()
        # result.word_probs = self.o_word_probs

        if not self._combined_semaphore:
            self._load_service.set_succeeded(result)
        return result

    def load_image_with_boxes(self, goal):

        print("load_image_with_boxes start")

        self._prev_query_breakdowns = []
        self._query_breakdowns = []

        self.img = CvBridge().imgmsg_to_cv2(goal.input, desired_encoding="passthrough")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.o_boxes = np.reshape(goal.boxes, (-1, 4))
        bbox_obj_names = goal.bbox_obj_names

        # add whole image context bbox
        bboxes = list(self.o_boxes)
        img_ht = int(self.img.shape[0])
        img_wd = int(self.img.shape[1])
        bboxes.append([0, 0, img_wd-1, img_ht-1])

        extraction_start = time.time()
        dense_fc7_feats, self.o_captions, self.o_scores = self.extract_box_densecap_features(
            self.img, bboxes, visualize=False)
        extraction_end = time.time()
        print "Extraction Time: %f" % (extraction_end - extraction_start)

        if len(bbox_obj_names) > 0:
            print("Replacing object names!!!")
            self._replace_object_names(self.o_captions, bbox_obj_names)

        self.o_fc7_whole = [dense_fc7_feats[len(dense_fc7_feats)-1]]
        self.o_fc7_feats = dense_fc7_feats[:-1]

        result = ingress_msgs.msg.DenseRefexpLoadBBoxesResult()
        result.captions = self.o_captions
        result.scores = self.o_scores

        if not self._combined_semaphore:
            self._load_bboxes_service.set_succeeded(result)
        return result

    def _replace_object_names(self, captions, true_names):
        '''
        Replace object name in captions if it is not close to the true object name
        '''

        # input validity check:
        if len(captions) != len(true_names) + 1:
            # captions include a caption for the whole image, therefore, the length is 1 more than len(true_names)
            rospy.logerr("_replace_object_names: captions len != true_names len")
            return False

        for i in range(len(true_names)):
            meteor_score = self._meteor.score(captions[i], true_names[i])
            rospy.loginfo("captions: {}, true_names: {}, score {}".format(
                captions[i], true_names[i], meteor_score))
            if meteor_score < NAME_REPLACEMENT_METEOR_SCORE_THRESHOLD:
                # replace the whole caption
                captions[i] = true_names[i]
                # change score to 1.0 to indicate that the name is replaced TODO
                # self.o_scores[i] = 1.0

                # TODO replace noun only
                # l = captions[i].split() # split string into list
                # noun_idx = self._get_noun_idx(l)
                # l[noun_idx] = true_names[i]
                # captions[i] = " ".join(l) # join list into string

        return True

    def relevancy_clustering(self, goal):

        incorrect_idxs = goal.incorrect_idxs
        query = goal.query

        # safety checks
        null_result = ingress_msgs.msg.RelevancyClusteringResult()

        if len(self.o_boxes) == 0:
            self._relevancy_clustering_service.set_aborted(
                null_result, "DenseRefexpLoad needs to called before querying")
            return None
        for idx in incorrect_idxs:
            if idx > len(self.o_boxes)-1:
                self._relevancy_clustering_service.set_aborted(
                    null_result, "Invalid incorrect indices")
                return None
        if len(incorrect_idxs) > 0 and len(self._prev_query_breakdowns) == 0:
            self._relevancy_clustering_service.set_aborted(
                null_result, "Cannot correct query without history")
            return None

        # new query
        if len(incorrect_idxs) == 0:

            self._query_breakdowns[:] = []

            breakdown_phrases = []
            breakdown_phrases.append(str(query))

            for phrase in breakdown_phrases:
                query = phrase

                # dense image query
                beam_length = len(self.o_captions)
                client = actionlib.SimpleActionClient(
                    'localize_query', ingress_msgs.msg.LocalizeQueryAction)
                client.wait_for_server()
                goal = ingress_msgs.msg.LocalizeQueryGoal(query, beam_length, 6.0)
                client.send_goal(goal, done_cb=done_query)
                client.wait_for_result()
                query_result = client.get_result()
                query += " <EOS>"

                self._query_breakdowns.append([query, query_result])

        # still resolving previous query
        else:
            self._query_breakdowns = self._prev_query_breakdowns

        selection_orig_idx = []
        for (q_idx, qr) in enumerate(self._query_breakdowns):

            query = qr[0]
            query_result = qr[1]

            q_boxes, q_fc7_feats, q_captioning_losses, \
                q_similarity_ranks, q_orig_idx, q_similarity_score, dense_softmax = self.unpack_and_prune(
                    query_result, incorrect_idxs, query)

            boxes = q_boxes
            top_k_fc7_feats = q_fc7_feats
            scores = q_captioning_losses
            orig_idxs = q_orig_idx

            top_5_idxs = q_orig_idx[:5]
            best_meteor_sim = 0.0
            for idx_ref in top_5_idxs:
                # for idx_tar in top_5_idxs:
                if idx_ref != top_5_idxs[0]:
                    m_score = self._meteor.score(
                        self.o_captions[idx_ref], self.o_captions[top_5_idxs[0]])
                    if m_score > best_meteor_sim:
                        best_meteor_sim = m_score
                        print("relevancy_clustering 1", best_meteor_sim,
                              self.o_captions[idx_ref], self.o_captions[top_5_idxs[0]])

            # VGG16 classifier
            # p_top_k_fc7_feats = np.zeros((k,4096))
            # for (count, idx) in enumerate(p_orig_idx):
            #   top_k_fc7_feats[count,:] = self.o_fc7_feats[idx]

            # selection_idxs = np.where(labels == selected_cluster_label)[0]
            selection_idxs = self.k_means(q_captioning_losses, q_similarity_score, q_orig_idx, self.o_captions, query, slice_point=8, max_cluster_size=5,
                                          visualize=False, save_path='/home/mohitshridhar/Programs/referring-expressions/lib/tests/clustering/clustering_viz.png')
            if selection_idxs is None:
                continue

            is_visually_confident = True if (
                best_meteor_sim < 0.5 and dense_softmax[0] > 0.8) else False
            print("relevancy_clustering 2", is_visually_confident,
                  best_meteor_sim, dense_softmax[0])

            if not self._no_confidence_prune and is_visually_confident:
                selection_orig_idx = [top_5_idxs[0]]
            else:
                for idx in selection_idxs:
                    selection_orig_idx.append(q_orig_idx[idx])

        selection_orig_idx = np.unique(selection_orig_idx)
        #selection_orig_idx = q_orig_idx[:20]

        meteor_scores = [] * len(selection_orig_idx)
        for idx in selection_idxs:
            meteor_scores.append(q_similarity_score[idx])

        result = ingress_msgs.msg.RelevancyClusteringResult()
        result.selection_orig_idx = selection_orig_idx
        result.softmax_probs = dense_softmax
        result.meteor_scores = meteor_scores
        result.all_orig_idx = query_result.orig_idx

        if not self._combined_semaphore:
            self._relevancy_clustering_service.set_succeeded(result)

        return result

    def boxes_refexp_query(self, goal):
        eval_methods = ['noisy_or']

        incorrect_idxs = goal.incorrect_idxs
        selection_orig_idx = goal.selection_orig_idx
        query = goal.query

        #boxes = np.take(self.o_boxes, selection_orig_idx, axis=0)

        # ASSUMPTION: boxes are still aligned with 'top_k_fc7_feats' 'scores' 'orig_idxs'
        boxes = np.reshape(goal.boxes, (-1, 4))
        top_k_fc7_feats = np.take(self.o_fc7_feats, selection_orig_idx, axis=0)
        scores = np.take(self.o_scores, selection_orig_idx, axis=0)
        orig_idxs = selection_orig_idx

        # image region features
        batch_size = len(boxes)
        img_wd = int(self.img.shape[1])
        img_ht = int(self.img.shape[0])
        fc7_img = self.o_fc7_whole

        img_wd = float(img_wd)
        img_ht = float(img_ht)
        image_feature_length = len(fc7_img[0])

        # Any change to context_length value will also require a change in the deploy prototxt
        context_length = 10
        fc7_obj = np.zeros((batch_size, context_length, image_feature_length))
        context_fc7 = np.tile(fc7_img, (batch_size, context_length, 1))
        bbox_features = np.zeros((batch_size, context_length, 5))
        context_bbox_features = np.zeros((batch_size, context_length, 5), np.float16)

        context_boxes = np.zeros((batch_size, context_length, 4))
        context_bboxes = []
        bbox_idx = 0

        min_x = np.min(boxes[:, 0])
        min_y = np.min(boxes[:, 1])
        new_w = np.max(boxes[:, 0] + boxes[:, 2]) - min_x
        new_h = np.max(boxes[:, 1] + boxes[:, 3]) - min_y

        for bbox in boxes:
            # adjusted boxes
            delta_x = float(bbox[0] - min_x)
            delta_y = float(bbox[1] - min_y)
            delta_w = float(bbox[2])
            delta_h = float(bbox[3])

            # Object region features
            fc7_obj[bbox_idx, :] = top_k_fc7_feats[bbox_idx][:]  # h5file[str((image_id,bbox))][:]

            # Bounding box features
            # bbox_area_ratio = (bbox[2]*bbox[3])/(img_wd*img_ht)
            # bbox_x1y1x2y2 = [bbox[0]/img_wd, bbox[1]/img_ht,
            #                  min(1., (bbox[0]+bbox[2])/img_wd), min(1., (bbox[1]+bbox[3])/img_ht)]
            # obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
            # bbox_features[bbox_idx,:] = obj_bbox_features
            # context_bbox_features[bbox_idx,:] = [0,0,1,1,1]

            bbox_area_ratio = (delta_w*delta_h)/(new_w*new_h)
            bbox_x1y1x2y2 = [delta_x/new_w, delta_y/new_h,
                             min(1., (delta_x+delta_w)/new_w), min(1., (delta_y+delta_h)/new_h)]
            obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
            bbox_features[bbox_idx, :] = obj_bbox_features
            context_bbox_features[bbox_idx, :] = [0, 0, 1, 1, 1]

            # Context features
            other_bboxes = list(boxes)  # make a copy
            other_bboxes.pop(bbox_idx)

            # randomly sample context_length boxes
            if len(other_bboxes) > context_length-1:
                rand_sample = sorted(random.sample(range(len(other_bboxes)), context_length-1))
                other_bboxes = [other_bboxes[idx] for idx in rand_sample]

            context_bboxes.append(other_bboxes)

            for (other_bbox_idx, other_bbox) in enumerate(other_bboxes):
                # other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
                # other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
                #                        (other_bbox[0] + other_bbox[2]) / img_wd, (other_bbox[1] + other_bbox[3]) / img_ht]
                # other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
                # feats = top_k_fc7_feats[other_bbox_idx]
                # context_fc7[bbox_idx,other_bbox_idx,:] = feats
                # context_bbox_features[bbox_idx,other_bbox_idx,:] = other_bbox_features

                # context_boxes[bbox_idx,other_bbox_idx,:] = other_bbox

                delta_x = float(other_bbox[0] - min_x)
                delta_y = float(other_bbox[1] - min_y)
                delta_w = float(other_bbox[2])
                delta_h = float(other_bbox[3])

                other_bbox_area_ratio = (delta_w * delta_h) / (new_w * new_h)
                other_bbox_x1y1x2y2 = [delta_x / new_w, delta_y / new_h,
                                       (delta_x + delta_w) / new_w, (delta_y + delta_h) / new_h]
                other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
                feats = top_k_fc7_feats[other_bbox_idx]
                context_fc7[bbox_idx, other_bbox_idx, :] = feats
                context_bbox_features[bbox_idx, other_bbox_idx, :] = other_bbox_features

                context_boxes[bbox_idx, other_bbox_idx, :] = other_bbox

            bbox_idx += 1

        for elem in context_bboxes:
            elem.append([0, 0, img_wd-1, img_ht-1])

        # -----------------------------------------------
        # Intial Pass
        # ------------------------------------------------

        prefix_words_unfiltered = get_encoded_line(query, self.lang_model.vocab)

        prefix_words = []
        for word in prefix_words_unfiltered:
            if word != self.lang_model.vocab[UNK_IDENTIFIER]:
                prefix_words.append(word)
        prefix_words = [prefix_words] * batch_size
        output_captions, output_probs, \
            output_all_probs = self.lang_model.sample_captions_with_context(fc7_obj, bbox_features,
                                                                            context_fc7, context_bbox_features,
                                                                            prefix_words=prefix_words)
        all_stats = [gen_stats(output_prob) for output_prob in output_all_probs]
        all_stats_p_word = [stat['p_word'] for stat in all_stats]
        all_stats_p_word = np.reshape(all_stats_p_word, (batch_size, context_length))

        for method in eval_methods:
            if method == 'noisy_or':
                num_context_objs = min(context_length-1, len(boxes)-1)
                sort_all_stats_p_word = -np.sort(-all_stats_p_word[:, 0:num_context_objs])
                top_all_stats_p_word = np.hstack((sort_all_stats_p_word, all_stats_p_word[:, -1:]))
                stats = (1 - np.product(1-top_all_stats_p_word, axis=1))
            elif method == 'image_context_only':
                stats = all_stats_p_word[:, -1]
            elif method == 'max':
                stats = np.max(all_stats_p_word, axis=1)
            else:
                raise StandardError("Unknown eval method %s" % method)

            (sort_keys, sorted_stats) = zip(*sorted(enumerate(stats), key=lambda x: -x[1]))
            top_k = 10 if len(sort_keys) > 10 else len(sort_keys)
            top_bboxes = [boxes[k] for k in sort_keys[:top_k]]
            ref_probs_list = [stats[k] for k in sort_keys[:top_k]]

            if method == 'noisy_or':
                noisy_or_top_box = top_bboxes[0]
            elif method == "image_context_only":
                image_top_bbox = top_bboxes[0]

        probs = np.array(stats)
        orig_probs = np.array(probs)
        probs.sort()
        descending_probs = probs[::-1]

        # This is an abomination. Why you do this??????????????????!!!!!!!!!!!!!!!

        print ("Descending Probs")
        print (descending_probs)

        ref_probs = ref_probs_list

        # check if output probs are ambiguious
        is_ambiguous = False
        pred_caps = []

        print "Dis" + str(self._disambiguate)

        # and (len(descending_probs) >= 2 and (descending_probs[0]-descending_probs[1] < AMBUGUITY_THRESHOLD)):
        if self._disambiguate:
            # if False:
            # Second Pass for ambuguity resolve gen
            # ----------------------------------------

            # eos_idx = query.index(" <EOS>")
            # query = query[:eos_idx] + " that is" + query[eos_idx:]
            query += " is "
            # query += " in the "
            prefix_words_unfiltered = get_encoded_line(query, self.lang_model.vocab)

            prefix_words = []
            for word in prefix_words_unfiltered:
                if word != self.lang_model.vocab[UNK_IDENTIFIER]:
                    prefix_words.append(word)
            prefix_words = [prefix_words] * batch_size

            print("prefix_word {}".format(prefix_words))

            output_captions, output_probs, \
                output_all_probs = self.lang_model.sample_captions_with_context(fc7_obj, bbox_features,
                                                                                context_fc7, context_bbox_features,
                                                                                prefix_words=prefix_words)
            all_stats = [gen_stats(output_prob) for output_prob in output_all_probs]
            all_stats_p_word = [stat['p_word'] for stat in all_stats]
            all_stats_p_word = np.reshape(all_stats_p_word, (batch_size, context_length))

            for method in eval_methods:
                if method == 'noisy_or':
                    num_context_objs = min(context_length-1, len(boxes)-1)
                    sort_all_stats_p_word = -np.sort(-all_stats_p_word[:, 0:num_context_objs])
                    top_all_stats_p_word = np.hstack(
                        (sort_all_stats_p_word, all_stats_p_word[:, -1:]))
                    stats = (1 - np.product(1-top_all_stats_p_word, axis=1))
                elif method == 'image_context_only':
                    stats = all_stats_p_word[:, -1]
                elif method == 'max':
                    stats = np.max(all_stats_p_word, axis=1)
                else:
                    raise StandardError("Unknown eval method %s" % method)

                (sort_keys, sorted_stats) = zip(*sorted(enumerate(stats), key=lambda x: -x[1]))
                top_k = 10 if len(sort_keys) > 10 else len(sort_keys)
                top_bboxes = [boxes[k] for k in sort_keys[:top_k]]

                predictive_captions = [self.lang_model.sentence(
                    cap).lower().replace(' is', '') for cap in output_captions]
                sorted_predictive_captions = [predictive_captions[k] for k in sort_keys[:top_k]]

                # ref_probs = np.array(stats)
                ref_probs_list = [orig_probs[k] for k in sort_keys[:top_k]]

                ref_probs = np.array(ref_probs_list)
                ref_similarity_score = [self._meteor.score(
                    goal.query, cap) for cap in sorted_predictive_captions]
                ref_orig_idx = range(len(ref_similarity_score))
                selection_idxs = self.k_means(ref_probs, ref_similarity_score, ref_orig_idx, sorted_predictive_captions, goal.query,
                                              slice_point=6, max_cluster_size=6,
                                              rev_loss=False, norm_meteor=False, visualize=VISUALIZE, save_path='/home/mohit/Programs/referring-expressions/lib/tests/clustering/clustering_spat.png')

                for t in range(len(ref_similarity_score)):
                    print sorted_predictive_captions[t], ref_similarity_score[t], ref_orig_idx[t]

                is_ambiguous = True
                pred_caps = sorted_predictive_captions

                if method == 'noisy_or':
                    noisy_or_top_box = top_bboxes[0]
                elif method == "image_context_only":
                    image_top_bbox = top_bboxes[0]

            print "AMBIGUOUS EXPRESSION -----------------"
            print (sorted_predictive_captions)

        # -----------------------------------------------------------------

        top_orig_idxs = []
        for top_bbox in top_bboxes:
            top_box_ind = boxes.tolist().index(top_bbox.tolist())
            top_orig_idxs.append(orig_idxs[top_box_ind])

        result = ingress_msgs.msg.BoxesRefexpQueryResult()
        result.top_box_idx = top_orig_idxs[0]
        result.context_boxes_idxs = top_orig_idxs[1:]
        result.is_ambiguous = is_ambiguous
        result.predicted_captions = pred_caps
        result.probs = ref_probs  # orig_probs
        if not self._combined_semaphore:
            self._boxes_refexp_query_service.set_succeeded(result)

        # TODO: move VISUALIZE to settings file (duh!)
        if VISUALIZE:
            iter_idx = 0
            plt.figure(3)
            plt.cla()
            plt.imshow(self.img)
            plt.title('refexp')
            top_box = top_bboxes[iter_idx]
            top_box_ind = boxes.tolist().index(top_box.tolist())

            plt.gca().add_patch(plt.Rectangle((top_box[0], top_box[1]), top_box[2], top_box[3],
                                              fill=False, edgecolor='r', linewidth=6))
            top_context_box_ind = np.argmax(all_stats_p_word[top_box_ind])
            for context_box in context_boxes[top_box_ind]:
                plt.gca().add_patch(plt.Rectangle((context_box[0], context_box[1]), context_box[2],
                                                  context_box[3], fill=False, edgecolor='g', linewidth=1,
                                                  linestyle='dashed'))
            plt.axis('off')
            plt.figure(3)

            pause(10.0)

        self._prev_query_breakdowns = self._query_breakdowns

        plt.close()
        plt.close()

        return result

    def refexp_query(self, goal):

        self._combined_semaphore = True

        # relevancy clustering
        rc_goal = ingress_msgs.msg.RelevancyClusteringGoal(goal.query, goal.incorrect_idxs)
        result = self.relevancy_clustering(rc_goal)
        if result == None:
            self._query_service.set_aborted(ingress_msgs.msg.DenseRefexpQueryResult())
            return None

        # select bboxes
        selection_orig_idx = result.selection_orig_idx

        # TODO testing
        # goal.query = "kitchen table on the left"

        # final refexp grounding
        selected_boxes = np.take(self.o_boxes, selection_orig_idx, axis=0)
        r_goal = ingress_msgs.msg.BoxesRefexpQueryGoal(
            goal.query, selected_boxes.ravel(), selection_orig_idx, goal.incorrect_idxs)
        query_result = self.boxes_refexp_query(r_goal)
        self._query_service.set_succeeded(query_result)

        self._combined_semaphore = False
        return query_result

    # def refexp_query(self, goal):

    #   eval_methods = ['noisy_or']
    #   incorrect_idxs = goal.incorrect_idxs
    #   query = goal.query

    #   # safety checks
    #   null_result = ingress_msgs.msg.DenseRefexpQueryResult()

    #   if len(self.o_boxes) == 0:
    #     self._query_service.set_aborted(null_result, "DenseRefexpLoad needs to called before querying")
    #     return None
    #   for idx in incorrect_idxs:
    #     if idx > len(self.o_boxes)-1:
    #       self._query_service.set_aborted(null_result, "Invalid incorrect indices")
    #       return None
    #   if len(incorrect_idxs) > 0 and len(self._prev_query_breakdowns) == 0:
    #     self._query_service.set_aborted(null_result, "Cannot correct query without history")
    #     return None

    #   # new query
    #   if len(incorrect_idxs) == 0:

    #     self._query_breakdowns[:] = []

    #     # noun-phrase extraction (not used at the moment)
    #     # blob = TextBlob(query)
    #     # nps = blob.noun_phrases

    #     breakdown_phrases = []
    #     # for n_p in nps:
    #     #   breakdown_phrases.append(str(n_p))
    #     breakdown_phrases.append(str(query))

    #     for phrase in breakdown_phrases:
    #       query = phrase

    #       # dense image query
    #       beam_length = len(self.o_captions)
    #       client = actionlib.SimpleActionClient('localize_query', ingress_msgs.msg.LocalizeQueryAction)
    #       client.wait_for_server()
    #       goal = ingress_msgs.msg.LocalizeQueryGoal(query, beam_length, 6.0)
    #       client.send_goal(goal, done_cb=done_query)
    #       client.wait_for_result()
    #       query_result = client.get_result()
    #       query += " <EOS>"

    #       self._query_breakdowns.append([query, query_result])

    #   # still resolving previous query
    #   else:
    #     self._query_breakdowns = self._prev_query_breakdowns

    #   selection_orig_idx = []
    #   for (q_idx, qr) in enumerate(self._query_breakdowns):

    #     query = qr[0]
    #     query_result = qr[1]

    #     q_boxes, q_fc7_feats, q_captioning_losses, \
    #     q_similarity_ranks, q_orig_idx, q_similarity_score, dense_softmax = self.unpack_and_prune(query_result, incorrect_idxs, query)

    #     boxes = q_boxes
    #     top_k_fc7_feats = q_fc7_feats
    #     scores = q_captioning_losses
    #     orig_idxs = q_orig_idx

    #     # VGG16 classifier
    #     # p_top_k_fc7_feats = np.zeros((k,4096))
    #     # for (count, idx) in enumerate(p_orig_idx):
    #     #   top_k_fc7_feats[count,:] = self.o_fc7_feats[idx]

    #     # selection_idxs = np.where(labels == selected_cluster_label)[0]
    #     selection_idxs = self.k_means(q_captioning_losses, q_similarity_score, q_orig_idx, self.o_captions, query, slice_point=10, max_cluster_size=5, visualize=VISUALIZE)
    #     if selection_idxs is None: continue

    #     for idx in selection_idxs:
    #       selection_orig_idx.append(q_orig_idx[idx])

    #   selection_orig_idx = np.unique(selection_orig_idx)

    #   boxes = np.take(self.o_boxes, selection_orig_idx, axis=0)
    #   top_k_fc7_feats = np.take(self.o_fc7_feats, selection_orig_idx, axis=0)
    #   scores = np.take(self.o_scores, selection_orig_idx, axis=0)
    #   orig_idxs = selection_orig_idx

    #   # image region features
    #   batch_size = len(boxes)
    #   img_wd = int(self.img.shape[1])
    #   img_ht = int(self.img.shape[0])
    #   fc7_img = self.o_fc7_whole

    #   img_wd = float(img_wd)
    #   img_ht = float(img_ht)
    #   image_feature_length = len(fc7_img[0])

    #   # Any change to context_length value will also require a change in the deploy prototxt
    #   context_length = 10
    #   fc7_obj = np.zeros((batch_size,context_length,image_feature_length))
    #   context_fc7 = np.tile(fc7_img,(batch_size,context_length,1))
    #   bbox_features = np.zeros((batch_size,context_length,5))
    #   context_bbox_features = np.zeros((batch_size,context_length, 5),np.float16)

    #   context_boxes = np.zeros((batch_size,context_length, 4))
    #   context_bboxes = []
    #   bbox_idx = 0

    #   min_x = np.min(boxes[:,0])
    #   min_y = np.min(boxes[:,1])
    #   new_w = np.max(boxes[:,0] + boxes[:,2]) - min_x
    #   new_h = np.max(boxes[:,1] + boxes[:,3]) - min_y

    #   for bbox in boxes:
    #     # adjusted boxes
    #     delta_x = float(bbox[0] - min_x)
    #     delta_y = float(bbox[1] - min_y)
    #     delta_w = float(bbox[2])
    #     delta_h = float(bbox[3])

    #     # Object region features
    #     fc7_obj[bbox_idx,:] = top_k_fc7_feats[bbox_idx][:] # h5file[str((image_id,bbox))][:]

    #     # Bounding box features
    #     # bbox_area_ratio = (bbox[2]*bbox[3])/(img_wd*img_ht)
    #     # bbox_x1y1x2y2 = [bbox[0]/img_wd, bbox[1]/img_ht,
    #     #                  min(1., (bbox[0]+bbox[2])/img_wd), min(1., (bbox[1]+bbox[3])/img_ht)]
    #     # obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
    #     # bbox_features[bbox_idx,:] = obj_bbox_features
    #     # context_bbox_features[bbox_idx,:] = [0,0,1,1,1]

    #     bbox_area_ratio = (delta_w*delta_h)/(new_w*new_h)
    #     bbox_x1y1x2y2 = [delta_x/new_w, delta_y/new_h,
    #                      min(1., (delta_x+delta_w)/new_w), min(1., (delta_y+delta_h)/new_h)]
    #     obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
    #     bbox_features[bbox_idx,:] = obj_bbox_features
    #     context_bbox_features[bbox_idx,:] = [0,0,1,1,1]

    #     # Context features
    #     other_bboxes = list(boxes)  # make a copy
    #     other_bboxes.pop(bbox_idx)

    #     # randomly sample context_length boxes
    #     if len(other_bboxes) > context_length-1:
    #       rand_sample = sorted(random.sample(range(len(other_bboxes)),context_length-1))
    #       other_bboxes = [other_bboxes[idx] for idx in rand_sample]

    #     context_bboxes.append(other_bboxes)

    #     for (other_bbox_idx, other_bbox) in enumerate(other_bboxes):
    #       # other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
    #       # other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
    #       #                        (other_bbox[0] + other_bbox[2]) / img_wd, (other_bbox[1] + other_bbox[3]) / img_ht]
    #       # other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
    #       # feats = top_k_fc7_feats[other_bbox_idx]
    #       # context_fc7[bbox_idx,other_bbox_idx,:] = feats
    #       # context_bbox_features[bbox_idx,other_bbox_idx,:] = other_bbox_features

    #       # context_boxes[bbox_idx,other_bbox_idx,:] = other_bbox

    #       delta_x = float(other_bbox[0] - min_x)
    #       delta_y = float(other_bbox[1] - min_y)
    #       delta_w = float(other_bbox[2])
    #       delta_h = float(other_bbox[3])

    #       other_bbox_area_ratio = (delta_w * delta_h) / (new_w * new_h)
    #       other_bbox_x1y1x2y2 = [delta_x / new_w, delta_y / new_h,
    #                              (delta_x + delta_w) / new_w, (delta_y + delta_h) / new_h]
    #       other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
    #       feats = top_k_fc7_feats[other_bbox_idx]
    #       context_fc7[bbox_idx,other_bbox_idx,:] = feats
    #       context_bbox_features[bbox_idx,other_bbox_idx,:] = other_bbox_features

    #       context_boxes[bbox_idx,other_bbox_idx,:] = other_bbox

    #     bbox_idx += 1

    #   for elem in context_bboxes:
    #     elem.append([0,0,img_wd-1,img_ht-1])

    #   prefix_words_unfiltered = get_encoded_line(query, self.lang_model.vocab)

    #   prefix_words = []
    #   for word in prefix_words_unfiltered:
    #     if word != self.lang_model.vocab[UNK_IDENTIFIER]:
    #       prefix_words.append(word)
    #   prefix_words = [prefix_words] * batch_size
    #   output_captions, output_probs, \
    #   output_all_probs = self.lang_model.sample_captions_with_context(fc7_obj, bbox_features,
    #                                                                   context_fc7, context_bbox_features,
    #                                                                   prefix_words=prefix_words)
    #   all_stats = [gen_stats(output_prob) for output_prob in output_all_probs]
    #   all_stats_p_word = [stat['p_word'] for stat in all_stats]
    #   all_stats_p_word = np.reshape(all_stats_p_word, (batch_size, context_length))

    #   for method in eval_methods:
    #     if method == 'noisy_or':
    #       num_context_objs = min(context_length-1,len(boxes)-1)
    #       sort_all_stats_p_word = -np.sort(-all_stats_p_word[:,0:num_context_objs])
    #       top_all_stats_p_word = np.hstack((sort_all_stats_p_word,all_stats_p_word[:,-1:]))
    #       stats = (1 - np.product(1-top_all_stats_p_word,axis=1))
    #     elif method == 'image_context_only':
    #       stats = all_stats_p_word[:,-1]
    #     elif method == 'max':
    #       stats = np.max(all_stats_p_word,axis=1)
    #     else:
    #       raise StandardError("Unknown eval method %s" % method)

    #     (sort_keys, sorted_stats) = zip(*sorted(enumerate(stats), key=lambda x:-x[1]))
    #     top_k = 10 if len(sort_keys) > 10 else len(sort_keys)
    #     top_bboxes = [boxes[k] for k in sort_keys[:top_k]]

    #     if method == 'noisy_or':
    #       noisy_or_top_box = top_bboxes[0]
    #     elif method == "image_context_only":
    #       image_top_bbox = top_bboxes[0]

    #   top_orig_idxs = []
    #   for top_bbox in top_bboxes:
    #     top_box_ind = boxes.tolist().index(top_bbox.tolist())
    #     top_orig_idxs.append(orig_idxs[top_box_ind])

    #   result = ingress_msgs.msg.DenseRefexpQueryResult()
    #   result.top_box_idx = top_orig_idxs[0]
    #   result.context_boxes_idxs = top_orig_idxs[1:]
    #   if not self._combined_semaphore:
    #     self._query_service.set_succeeded(result)

    #   # TODO: move VISUALIZE to settings file (duh!)
    #   if VISUALIZE:
    #     iter_idx = 0
    #     plt.figure(3)
    #     plt.cla()
    #     plt.imshow(self.img)
    #     plt.title('refexp')
    #     top_box = top_bboxes[iter_idx]
    #     top_box_ind = boxes.tolist().index(top_box.tolist())

    #     plt.gca().add_patch(plt.Rectangle((top_box[0], top_box[1]),top_box[2], top_box[3],
    #                                       fill=False, edgecolor='r', linewidth=6))
    #     top_context_box_ind = np.argmax(all_stats_p_word[top_box_ind])
    #     for context_box in context_boxes[top_box_ind]:
    #       plt.gca().add_patch(plt.Rectangle((context_box[0], context_box[1]),context_box[2],
    #                                       context_box[3], fill=False, edgecolor='g', linewidth=1,
    #                                       linestyle='dashed'))
    #     plt.axis('off')
    #     plt.figure(3)

    #     pause(10.0)

    #   self._prev_query_breakdowns = self._query_breakdowns

    #   plt.close()
    #   plt.close()

    #   return result

    def load_and_query(self, goal):
        '''
        Not Working Yet!
        '''

        self._combined_semaphore = True

        loading_result = self.load_image(goal)
        query_result = self.refexp_query(goal)

        context = []
        for item in query_result.context_boxes_idxs:
            context.append(int(item))

        combined_result = ingress_msgs.msg.DenseRefexpLoadQueryResult()
        combined_result.captions = list(loading_result.captions)
        combined_result.scores = list(loading_result.captions)
        combined_result.boxes = list(loading_result.boxes)
        combined_result.top_box_idx = int(query_result.top_box_idx)
        combined_result.context_boxes_idxs = context

        self._combined_semaphore = False
        self._load_and_query_service.set_succeeded(combined_result)

    def comprehension_full_pipeline(self, experiment_paths, proposal_source='gt', visualize=False, eval_method=None):

        # Custom Image:
        # path = '/home/mohitshridhar/Programs/densecap/webcam/table_medium.png'
        # path = '/home/mohitshridhar/Programs/densecap/webcam/table_medium_2.png'
        # path = '/home/mohitshridhar/Programs/densecap/webcam/table_mult_dups.png'
        path = '/home/mohitshridhar/Programs/densecap/webcam/mult_obj_conf.png'
        # path = '/home/mohitshridhar/Programs/densecap/webcam/3_water_bottles.png'
        # path = '/home/mohitshridhar/Programs/densecap/webcam/3_blocks.png'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        o_boxes, o_fc7_whole, o_fc7_feats, o_obj_classes, o_captions, o_scores = self.extract_single_densecap_features(
            img, visualize=True)

        if eval_method is None:
            eval_methods = ['noisy_or', 'max', 'image_context_only']
        else:
            eval_methods = [eval_method]

        incorrect_idxs = []
        prev_query_breakdowns = None
        query_breakdowns = None

        while True:
            # new query
            query_breakdowns = []
            if len(incorrect_idxs) == 0:
                query = raw_input('Search Query: ').lower()

                if "quit" in query:
                    break

                # noun-phrase extraction
                # blob = TextBlob(query)
                # nps = blob.noun_phrases

                breakdown_phrases = []
                # for n_p in nps:
                #   breakdown_phrases.append(str(n_p))
                breakdown_phrases.append(str(query))

                for phrase in breakdown_phrases:
                    query = phrase

                    # dense image query
                    beam_length = len(o_captions)
                    client = actionlib.SimpleActionClient(
                        'localize_query', ingress_msgs.msg.LocalizeQueryAction)
                    client.wait_for_server()
                    goal = ingress_msgs.msg.LocalizeQueryGoal(query, beam_length, 6.0)
                    client.send_goal(goal, done_cb=done_query)
                    client.wait_for_result()
                    query_result = client.get_result()
                    query += " <EOS>"

                    query_breakdowns.append([query, query_result])

            # still resolving previous query
            else:
                # query = prev_query
                query_breakdowns = prev_query_breakdowns

            selection_orig_idx = []
            for (q_idx, qr) in enumerate(query_breakdowns):

                query = qr[0]
                query_result = qr[1]

                q_boxes, q_fc7_feats, q_captioning_losses, \
                    q_similarity_ranks, q_orig_idx, q_similarity_score, dense_softmax = self.unpack_and_prune(
                        query_result, incorrect_idxs)

                # extracting data from clusters
                p_boxes = q_boxes
                p_top_k_fc7_feats = q_fc7_feats
                p_scores = q_captioning_losses
                p_orig_idxs = q_orig_idx

                # VGG16 classifier
                # p_top_k_fc7_feats = np.zeros((k,4096))
                # for (count, idx) in enumerate(p_orig_idx):
                #   top_k_fc7_feats[count,:] = o_fc7_feats[idx]

                # selection_idxs = np.where(labels == selected_cluster_label)[0]
                selection_idxs = self.k_means(q_captioning_losses, q_similarity_score, q_orig_idx,
                                              o_captions, query, slice_point=10, max_cluster_size=5, visualize=True)
                if selection_idxs is None:
                    continue

                for idx in selection_idxs:
                    selection_orig_idx.append(p_orig_idxs[idx])

            selection_orig_idx = np.unique(selection_orig_idx)
            # selection_idxs = []

            # boxes = np.take(boxes, selection_idxs, axis=0)
            # top_k_fc7_feats = np.take(top_k_fc7_feats, selection_idxs, axis=0)
            # scores = np.take(scores, selection_idxs, axis=0)
            # orig_idxs = np.take(orig_idxs, selection_idxs, axis=0)

            boxes = np.take(o_boxes, selection_orig_idx, axis=0)
            top_k_fc7_feats = np.take(o_fc7_feats, selection_orig_idx, axis=0)
            scores = np.take(o_scores, selection_orig_idx, axis=0)
            orig_idxs = selection_orig_idx

            # image region features
            batch_size = len(boxes)
            img_wd = int(img.shape[1])
            img_ht = int(img.shape[0])
            fc7_img = o_fc7_whole

            img_wd = float(img_wd)
            img_ht = float(img_ht)
            image_feature_length = len(fc7_img[0])
            # Any change to context_length value will also require a change in the deploy prototxt
            context_length = 10
            fc7_obj = np.zeros((batch_size, context_length, image_feature_length))
            context_fc7 = np.tile(fc7_img, (batch_size, context_length, 1))
            bbox_features = np.zeros((batch_size, context_length, 5))
            context_bbox_features = np.zeros((batch_size, context_length, 5), np.float16)

            context_boxes = np.zeros((batch_size, context_length, 4))

            context_bboxes = []
            bbox_idx = 0

            min_x = np.min(boxes[:, 0])
            min_y = np.min(boxes[:, 1])
            new_w = np.max(boxes[:, 0] + boxes[:, 2]) - min_x
            new_h = np.max(boxes[:, 1] + boxes[:, 3]) - min_y

            for bbox in boxes:
                # adjusted boxes
                delta_x = float(bbox[0] - min_x)
                delta_y = float(bbox[1] - min_y)
                delta_w = float(bbox[2])
                delta_h = float(bbox[3])

                # Object region features
                # h5file[str((image_id,bbox))][:]
                fc7_obj[bbox_idx, :] = top_k_fc7_feats[bbox_idx][:]

                # Bounding box features
                # bbox_area_ratio = (bbox[2]*bbox[3])/(img_wd*img_ht)
                # bbox_x1y1x2y2 = [bbox[0]/img_wd, bbox[1]/img_ht,
                #                  min(1., (bbox[0]+bbox[2])/img_wd), min(1., (bbox[1]+bbox[3])/img_ht)]
                # obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
                # bbox_features[bbox_idx,:] = obj_bbox_features
                # context_bbox_features[bbox_idx,:] = [0,0,1,1,1]

                bbox_area_ratio = (delta_w*delta_h)/(new_w*new_h)
                bbox_x1y1x2y2 = [delta_x/new_w, delta_y/new_h,
                                 min(1., (delta_x+delta_w)/new_w), min(1., (delta_y+delta_h)/new_h)]
                obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
                bbox_features[bbox_idx, :] = obj_bbox_features
                context_bbox_features[bbox_idx, :] = [0, 0, 1, 1, 1]

                # Context features
                other_bboxes = list(boxes)  # make a copy
                other_bboxes.pop(bbox_idx)

                # randomly sample context_length boxes
                if len(other_bboxes) > context_length-1:
                    rand_sample = sorted(random.sample(range(len(other_bboxes)), context_length-1))
                    other_bboxes = [other_bboxes[idx] for idx in rand_sample]

                context_bboxes.append(other_bboxes)

                for (other_bbox_idx, other_bbox) in enumerate(other_bboxes):
                    # other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
                    # other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
                    #                        (other_bbox[0] + other_bbox[2]) / img_wd, (other_bbox[1] + other_bbox[3]) / img_ht]
                    # other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
                    # feats = top_k_fc7_feats[other_bbox_idx]
                    # context_fc7[bbox_idx,other_bbox_idx,:] = feats
                    # context_bbox_features[bbox_idx,other_bbox_idx,:] = other_bbox_features

                    # context_boxes[bbox_idx,other_bbox_idx,:] = other_bbox

                    delta_x = float(other_bbox[0] - min_x)
                    delta_y = float(other_bbox[1] - min_y)
                    delta_w = float(other_bbox[2])
                    delta_h = float(other_bbox[3])

                    other_bbox_area_ratio = (delta_w * delta_h) / (new_w * new_h)
                    other_bbox_x1y1x2y2 = [delta_x / new_w, delta_y / new_h,
                                           (delta_x + delta_w) / new_w, (delta_y + delta_h) / new_h]
                    other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
                    feats = top_k_fc7_feats[other_bbox_idx]
                    context_fc7[bbox_idx, other_bbox_idx, :] = feats
                    context_bbox_features[bbox_idx, other_bbox_idx, :] = other_bbox_features

                    context_boxes[bbox_idx, other_bbox_idx, :] = other_bbox

                bbox_idx += 1

            for elem in context_bboxes:
                elem.append([0, 0, img_wd-1, img_ht-1])

            prefix_words_unfiltered = get_encoded_line(query, self.lang_model.vocab)

            prefix_words = []
            for word in prefix_words_unfiltered:
                if word != self.lang_model.vocab[UNK_IDENTIFIER]:
                    prefix_words.append(word)
            prefix_words = [prefix_words] * batch_size
            output_captions, output_probs, \
                output_all_probs = self.lang_model.sample_captions_with_context(fc7_obj, bbox_features,
                                                                                context_fc7, context_bbox_features,
                                                                                prefix_words=prefix_words)
            all_stats = [gen_stats(output_prob) for output_prob in output_all_probs]
            all_stats_p_word = [stat['p_word'] for stat in all_stats]
            all_stats_p_word = np.reshape(all_stats_p_word, (batch_size, context_length))

            for method in eval_methods:
                if method == 'noisy_or':
                    num_context_objs = min(context_length-1, len(boxes)-1)
                    sort_all_stats_p_word = -np.sort(-all_stats_p_word[:, 0:num_context_objs])
                    top_all_stats_p_word = np.hstack(
                        (sort_all_stats_p_word, all_stats_p_word[:, -1:]))
                    stats = (1 - np.product(1-top_all_stats_p_word, axis=1))
                elif method == 'image_context_only':
                    stats = all_stats_p_word[:, -1]
                elif method == 'max':
                    stats = np.max(all_stats_p_word, axis=1)
                else:
                    raise StandardError("Unknown eval method %s" % method)

                (sort_keys, sorted_stats) = zip(*sorted(enumerate(stats), key=lambda x: -x[1]))
                top_k = 10 if len(sort_keys) > 10 else len(sort_keys)
                top_bboxes = [boxes[k] for k in sort_keys[:top_k]]

                if method == 'noisy_or':
                    noisy_or_top_box = top_bboxes[0]
                elif method == "image_context_only":
                    image_top_bbox = top_bboxes[0]

            if visualize:
                for iter_idx in range(len(selection_idxs)):
                    plt.figure(3)
                    plt.cla()
                    plt.imshow(img)
                    plt.title('refexp')
                    top_box = top_bboxes[iter_idx]
                    top_box_ind = boxes.tolist().index(top_box.tolist())

                    plt.gca().add_patch(plt.Rectangle((top_box[0], top_box[1]), top_box[2], top_box[3],
                                                      fill=False, edgecolor='r', linewidth=6))
                    top_context_box_ind = np.argmax(all_stats_p_word[top_box_ind])
                    for context_box in context_boxes[top_box_ind]:
                        plt.gca().add_patch(plt.Rectangle((context_box[0], context_box[1]), context_box[2],
                                                          context_box[3], fill=False, edgecolor='g', linewidth=1,
                                                          linestyle='dashed'))
                    plt.axis('off')
                    plt.figure(3)

                    pause(0.5)

                    # iterate on feedback
                    feedback = raw_input('\tthis one?  ').lower()

                    if "cancel" in feedback or "stop" in feedback:
                        incorrect_idxs[:] = []
                        break
                    elif "yes" in feedback:
                        incorrect_idxs[:] = []
                        print "\tComplete"
                        break
                    elif "no" in feedback:
                        incorrect_idxs.append(orig_idxs[top_box_ind])
                        prev_query_breakdowns = query_breakdowns
                        continue
                    else:
                        incorrect_idxs[:] = []
                        print "\tInvalid Feedback (Yes or No? or stop)"
                        break

                plt.close()
                plt.close()

        return None


def run_comprehension_experiment(dataset, experiment_paths, experiment_config, image_ids=None, no_confidence_prune=False, disambiguate=False, min_cluster_size=1):
    if experiment_config.exp_name == 'baseline' or experiment_config.exp_name.startswith('max_margin'):
        captioner = LanguageModel(experiment_config.test.lstm_model_file, experiment_config.test.lstm_net_file,
                                  experiment_config.vocab_file, device_id=0)
    elif experiment_config.exp_name.startswith('mil_context'):
        captioner = MILContextLanguageModel(experiment_config.test.lstm_model_file, experiment_config.test.lstm_net_file,
                                            experiment_config.vocab_file, device_id=0)
    else:
        raise StandardError("Unknown experiment name: %s" % experiment_config.exp_name)

    if experiment_config.exp_name == 'baseline' or experiment_config.exp_name.startswith('max_margin'):
        experimenter = ComprehensionExperiment(captioner, dataset, image_ids=image_ids)
    elif experiment_config.exp_name.startswith('mil_context'):
        experimenter = MILContextComprehension(captioner, dataset, image_ids=image_ids,
                                               no_confidence_prune=no_confidence_prune, disambiguate=disambiguate, min_cluster_size=min_cluster_size)
    else:
        raise StandardError("Unknown experiment name: %s" % experiment_config.exp_name)

    # results = experimenter.comprehension_full_pipeline(experiment_paths, proposal_source=experiment_config.test.proposal_source,
    #                                                 visualize=experiment_config.test.visualize)


def done_cb(goal_status, result):
    if DEBUG_PRINT:
        print 'Received Dense Localization Results'
        # print result


def done_query(goal_status, result):
    if DEBUG_PRINT:
        print 'Received Dense Query Results'


# Setup image publisher for localization
rospy.init_node('LocalizationImagePublisher', anonymous=True)

client = actionlib.SimpleActionClient('dense_localize', ingress_msgs.msg.LocalizeAction)
client.wait_for_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', required=True, help='Path to MSCOCO dataset')
    parser.add_argument('--dataset', required=True,
                        help='Name of the dataset. [Google_RefExp|UNC_RefExp]')
    parser.add_argument('--exp_name', required=True, nargs='+',
                        help='Type of model. [baseline|max-margin|mil_context_withNegMargin|mil_context_withPosNegMargin|all]')
    parser.add_argument('--split_name', required=True, help='Partition to test on')
    parser.add_argument('--proposal_source', required=True,
                        help='Test time proposal source [gt|mcg]')
    parser.add_argument('--visualize', action="store_true", default=False,
                        help='Display comprehension results')
    parser.add_argument('--no_confidence_prune', action="store_true", default=True)
    parser.add_argument('--disambiguate', action="store_true", default=False)
    parser.add_argument("--min_cluster_size", dest="min_cluster_size", type=int,
                        help="relevancy clustering minimum size", default=1)

    args = parser.parse_args()
    exp_paths = get_experiment_paths(args.coco_path)
    dataset = None

    if 'all' in args.exp_name:
        args.exp_name = ['baseline', 'max_margin',
                         'mil_context_withNegMargin', 'mil_context_withPosNegMargin']

    # Run all experiments first
    for exp_name in args.exp_name:
        exp_config = get_experiment_config(exp_paths, args.dataset, exp_name, mode='test',
                                           test_proposal_source=args.proposal_source, test_visualize=args.visualize)
        run_comprehension_experiment(dataset, exp_paths, exp_config, no_confidence_prune=args.no_confidence_prune,
                                     disambiguate=args.disambiguate, min_cluster_size=args.min_cluster_size)

    # run_meteor_score_server
