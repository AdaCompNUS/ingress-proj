import cv2
import numpy as np

import camera_params

def flip_img(img):
    '''
    flip image horizontally & then vertically
    '''
    img = cv2.flip(img, 0)
    img = cv2.flip(img, 1)

    return img

def flip_xywh_bboxes(bboxes):
    '''
    flip (x, y, w, h)
    '''
    bboxes = np.array(bboxes)
    bboxes[:,0] = ((camera_params.IMAGE_CROP_X_END - camera_params.IMAGE_CROP_X) - bboxes[:,0]) - bboxes[:,2] # W - x
    bboxes[:,1] = ((camera_params.IMAGE_CROP_Y_END - camera_params.IMAGE_CROP_Y) - bboxes[:,1]) - bboxes[:,3] # H - y
    # width and height don't change

    return bboxes
