import os, cv2, sys, datetime, time
import numpy as np
from assembly.interfaces.inferenceInterface import inferenceInterface
from assembly.interfaces.assemblyInterface import AssemblyInterface
from assembly.interfaces.trackerInterface import TrackerInterface


# Class that takes in the group info for a particular camera 
# and builds a custom pipeline for it depending on the json file
class Interface:
    """
    This class defines the Interface for processing an image through a pipeline of 
    tracking and/or region of interest (ROI) based inference.
    
    Attributes:
        group_info (dict): Configuration details for the pipeline.
        ModelDict (dict): Dictionary of pre-loaded models.
        TrackerDict (dict): Dictionary of pre-loaded trackers.
        camera_id (str): The ID of the camera for which this interface is built.
    """

    def __init__(self, group_info, ModelDict, TrackerDict, camera_id):
        """
        Initializes the Interface class.

        Args:
            group_info (dict): Configuration information for the pipeline.
            ModelDict (dict): Dictionary of pre-loaded models.
            TrackerDict (dict): Dictionary of pre-loaded trackers.
            camera_id (str): The ID of the camera.
        """
        self.group_info = group_info ## group_info same as camera or belt info
        self.ModelDict = ModelDict
        self.TrackerDict = TrackerDict
        self.camera_id = camera_id
        self.ground_truth = dict()
        self.build()


    # Method that build the pipeline based on the group info
    def build(self):
        """
        Builds the pipeline based on the provided group information.

        Note:
            - Sets up the tracker if specified.
            - Initializes ROI processors for cropping and inference.
        """
        self.tracker_status = True if "tracker" in self.group_info["steps"] else False
        if self.tracker_status:
            self.tracker_model = AssemblyInterface(model=self.ModelDict[self.group_info["tracker"]["model_id"]], threshold=0.75)
            self.tracker = TrackerInterface(tracker=self.TrackerDict[self.camera_id], roi=self.group_info["tracker"]["roi"]["line"], dir=self.group_info["tracker"]["roi"]["direction"])
        #tracker in trackerinterface is the tracker model ocsort or centroid that only being used for video frames
        self.roi_processors = dict() 
        for roi_id, interface_info in self.group_info["cropping"].items():
            self.roi_processors[roi_id] = inferenceInterface(interface_info=interface_info, ModelDict=self.ModelDict)

            self.ground_truth[roi_id] = interface_info["ground_truth"]

    # Method to crop the roi from the image
    def cropping(self, image, roi):
        """
        Crops the image based on the provided ROI.

        Args:
            image (array): The image to be cropped.
            roi (tuple): The region of interest as (x1, y1, x2, y2).

        Returns:
            array: The cropped image.
        """
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2].copy()
        return crop
    
    
    # Method that passes an image through the built pipeline
    def run(self, image): 
        """
        Runs the image through the built pipeline and returns the output.

        Args:
            image (array): The image on which the pipeline is to be run.

        Returns:
            dict: Dictionary containing tracker and ROI processor outputs.
        """
        x = time.time()
        res_dict = {"tracker":[], "cropping":[]}
        # res_dict = {}
        actual_image = image.copy()
        images = [image]
        frame_with_roi = image 
        object_count = 0 
        if self.tracker_status:
            # adding roi and direction in res_dict 

            res_dict["roi"] = self.group_info["tracker"]["roi"]["line"]           #[-/]
            res_dict["direction"] = self.group_info["tracker"]["roi"]["direction"]     #[-/]
            
            
            
            # Passing image to detection model
            bboxes, classes, scores = self.tracker_model(image=image)
            
            # Passing the detections to tracker
            # frame_with_roi = self.tracker.draw_roi(frame_with_roi)
            result_bbox, object_count = self.tracker.run(images, bboxes, classes, scores)
            #{2: [199, 313, 760, 918]}

            if len(result_bbox) == 0:
                return res_dict, object_count
            else:
                images = []
                for id, bbox in result_bbox.items():
                    
                    images.append(self.cropping(image=actual_image, roi=bbox))
                    res_dict["tracker"].append(bbox)

        #{'tracker': [[1770, 426, 1918, 654], [1167, 423, 1532, 663], [
        # 542, 423, 877, 662], [3, 426, 232, 674]], 'cropping': []}

        for ind, image in enumerate(images):
            # Initialize roi to None
            roi = [0,0,1,1]
            # Check if tracker results are present and not empty
            if "tracker" in res_dict and len(res_dict["tracker"]) > 0:
                # Extract bounding box coordinates
                roi = res_dict["tracker"][ind] #orignal cordinates for detection
                
            
            # Initialize roi_res dictionary
            roi_res = {}
            x = time.time()
            # Iterate through roi_processors and run with extracted roi
            for roi_id, processer in self.roi_processors.items():
                
                roi_res[roi_id] = processer.run(image=image, roi=roi) #image = cropped but roi = absolute wrt real image 
                
            # Append roi_res to cropping results in res_dict
            res_dict["cropping"].append(roi_res)
        
        




        return res_dict, object_count
        