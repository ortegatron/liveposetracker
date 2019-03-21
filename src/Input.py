# -*- coding: utf-8 -*-
import cv2
import sys
import time
import numpy as np

import pygame
# Cargar OpenPose:
sys.path.append('/usr/local/python')
from openpose import *
# from utils import poses2boxes
# from pymongo import MongoClient
# import json

from deep_sort.iou_matching import iou_cost
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepTracker
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.linear_assignment import min_cost_matching
from deep_sort.detection import Detection as ddet
from tools import generate_detections as gdet
from utils import poses2boxes

import Constants

class Input():
    def __init__(self, debug = False):
        #from openpose import *
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "160x160"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["default_model_folder"] = Constants.PATH + "/models/"
        self.openpose = OpenPose(params)

        max_cosine_distance = Constants.max_cosine_distance
        nn_budget = Constants.nn_budget
        self.nms_max_overlap = Constants.nms_max_overlap
        max_age = Constants.max_age
        n_init = Constants.n_init

        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepTracker(metric, max_age = max_age,n_init= n_init)

        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():         # Checks the stream
            self.frameSize = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        Constants.SCREEN_HEIGHT = self.frameSize[0]
        Constants.SCREEN_WIDTH = self.frameSize[1]


    def getCurrentFrameAsImage(self):
            frame = self.currentFrame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pgImg = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")
            return pgImg


    def run(self):
        result, self.currentFrame = self.capture.read()
        keypoints, self.currentFrame = self.openpose.forward(self.currentFrame, display = True)
        # print(keypoints)
        # Doesn't use keypoint confidence
        poses = keypoints[:,:,:2]
        # Get containing box for each seen body
        boxes = poses2boxes(poses)
        boxes_xywh = [[x1,y1,x2-x1,y2-y1] for [x1,y1,x2,y2] in boxes]
        features = self.encoder(self.currentFrame,boxes_xywh)
        # print(features)

        nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
        detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in zip(boxes_xywh, features, poses) if nonempty(bbox)]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes_det, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update( self.currentFrame, detections)

        for track in self.tracker.tracks:
            color = None
            if not track.is_confirmed():
                color = (0,0,255)
            else:
                color = (255,255,255)
            bbox = track.to_tlbr()
            cv2.rectangle(self.currentFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
            cv2.putText(self.currentFrame, "id%s - ts%s"%(track.track_id,track.time_since_update),(int(bbox[0]), int(bbox[1])-20),0, 5e-3 * 200, (0,255,0),2)


        # self.currentFrame = np.rot90(self.currentFrame)
        # self.currentFrame = cv2.flip(self.currentFrame, 1)
        # self.updateState()
        cv2.waitKey(1)
