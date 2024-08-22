import os, sys
import numpy as np
import torch, cv2, operator
from collections import deque

class TrackerInterface:
    """
    This class defines an interface for tracking objects in a running belt/camera using a pre-loaded tracking model.
    
    Attributes:
        tracker (object): The pre-loaded tracking model.
        roi (tuple): The region of interest for tracking.
        queue (deque): A queue to keep track of analyzed objects.
        dir_dict (dict): A dictionary to map directions to corresponding operator functions.
        operator (callable): The operator function used to check if an object has crossed the ROI.
    """
    def __init__(self, tracker, roi, dir):
        """
        Initializes the TrackerInterface class.

        Args:
            tracker (object): The pre-loaded tracking model.
            roi (tuple): Region of interest for tracking.
            dir (str): The direction of object movement to track (e.g., "up2down", "left2right").
        """
        self.tracker = tracker #centroid or ocsort
        self.roi = roi # which can be a line or a rectangle.
        self.dir = dir
        print("what is dir", dir)
        self.queue = deque(maxlen=25)
        self.dir_dict = {"up2down":operator.ge, 
                         "down2up": operator.le,
                         "left2right": operator.ge,
                         "right2left": operator.le}
        
        self.operator = self.dir_dict.get(dir)
        self.object_count = 0


    def update(self, dets, confs, classes): 
        """
        Updates the tracker with new detections.

        Args:
            dets (list): List of new detections, each as a bounding box [x1, y1, x2, y2].

        Returns:
            dict: Updated tracking results.
        """
        result = self.tracker.run(dets, confs, classes)
        if isinstance(result, tuple):
            result = {i: bbox for i, bbox in enumerate(result)}
        return result
        

    def run(self, frame, dets, confs, classes):
        """ 
        Runs the tracker and checks if any object has crossed the ROI.

        Args:
            frame (array): The image frame on which tracking is performed.
            dets (list): List of new detections.
            confs (list): List of confidence scores.
            classes (list): List of class indices.

        Returns:
            tuple: Modified frame with ROI drawn, tracking results, and object count.
        """
        result = {}
        tracker_res = self.update(dets=dets, confs=confs, classes=classes)
        if len(tracker_res) > 0:  # Checking if any bounding box was there
            # for tracker_dict in tracker_res:
            for id, bbox in tracker_res.items():
                if not self.objAnalysed(id):  # Checking if the object has already been analysed
                    if self.checkRoiCrossed(bbox):  # Checking if object has crossed the ROI 
                        result[id] = bbox
                        self.object_count += 1
                        self.queue.append(id)
                        
#[14, 383, 499, 896] 
        # Draw ROI on the frame
        # frame_with_roi = self.draw_roi(frame)
        
        return result, self.object_count              #, frame_with_roi

    def checkRoiCrossed(self, bbox):
        """
        Checks if the given bounding box has crossed the ROI.

        Args:
            bbox (list): Bounding box [x1, y1, x2, y2].

        Returns:
            bool: True if the object has crossed the ROI, otherwise False.
        """
        x1, y1, x2, y2 = bbox
        if self.dir in ["up2down", "down2up"]:
            cord = int((y1 + y2) / 2)
        else:
            cord = int((x1 + x2) / 2)
        crossedFlag = False

        if self.operator(cord, self.roi):

            crossedFlag = True
        return crossedFlag
    
    def objAnalysed(self, obj_id):
        """
        Checks if the object with the given ID has already been analysed.

        Args:
            obj_id (int): The ID of the object.

        Returns:
            bool: True if the object has been analysed, otherwise False.
        """
        analysedFlag = False
        if obj_id in self.queue:
            analysedFlag = True
        return analysedFlag


    def draw_roi(self, frame):
        """
        Draws the ROI on the frame.

        Args:
            frame (array): The image frame on which ROI is to be drawn.

        Returns:
            array: The image frame with the ROI drawn.
        """
        height, width = frame.shape[:2]

        if isinstance(self.roi, (float, int)) and self.roi < 1:  # Check if ROI is normalized
            roi = int(self.roi * height) if self.dir in ["up2down", "down2up"] else int(self.roi * width)
        else:
            roi = self.roi

        if isinstance(roi, (float, int)):
            if self.dir in ["up2down", "down2up"]:
                cv2.line(frame, (0, int(roi)), (width, int(roi)), (0, 255, 0), 2)
            else:
                cv2.line(frame, (int(roi), 0), (int(roi), height), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        return frame
    
    