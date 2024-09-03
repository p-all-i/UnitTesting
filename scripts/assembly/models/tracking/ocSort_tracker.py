import os
import numpy as np
from assembly.models.tracking.ocsort.ocsort import OCSort


class ocTracker:
    def __init__(self, det_thresh=0.45, movement_direction="left2right", ROI=300):
        self.det_thresh = det_thresh
        self.direction = movement_direction
        self.ROI = ROI

        self.tracker = OCSort(det_thresh=self.det_thresh, movement_direction=self.direction, ROI=ROI)

    
    def process_output(self, tracker_output):
        result = {}
        for (i, (startX, startY, endX, endY, idx, class_name)) in enumerate(tracker_output):
            bbox = [startX, startY, endX, endY]

            result[idx] = bbox
        return result
    

    def run(self, dets, confs, classes):
        tracker_res = self.tracker.update(dets=dets, confs=confs, classes=classes)

        res = self.process_output(tracker_output=tracker_res)

        return res