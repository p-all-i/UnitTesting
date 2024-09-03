import os
import sys
import numpy as np
from pathlib import Path
import torch
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv() 

# Inference class for Assembly
class AssemblyInterface:
    """
    This class defines an interface for running inference using a pre-loaded assembly model.
    
    Attributes:
        model (object): The pre-loaded assembly model for inference.
        conf_thresh (float): Confidence threshold for filtering model predictions.
        loggerObj (object): Logger object for logging information, defaults to None.
    """
    def __init__(self, model, threshold, loggerObj=None, custom_nms=False):
        """
        Initializes the AssemblyInterface class.

        Args:
            model (object): The pre-loaded assembly model.
            threshold (float): Confidence threshold for inference.
            loggerObj (object): Logger for logging, defaults to None.
            custom_nms (bool): Flag to indicate usage of custom Non-Maximum Suppression, defaults to False.
        """
        # Model dict
        self.loggerObj = loggerObj
        self.model = model
        self.conf_thresh = threshold

    # Method that calls the preprocess of the model selected
    def preProcess(self, image):
        """
        Preprocesses the image before running it through the model.

        Args:
            image (array): The input image.

        Returns:
            array: The preprocessed image.
        """
        input = self.model.preProcess(image)
        return input

    # Method that calls the forward of the model selected
    def forward(self, input):
        """
        Runs the preprocessed image through the model's forward pass.

        Args:
            input (array): The preprocessed image.

        Returns:
            array: The model's raw output.
        """
        pred_output = self.model.forward(input)
        return pred_output
    
    # Method that calls the postprocess of the model selected
    def postProcess(self, pred_output):
        """
        Post-processes the model's raw output to get the final results.

        Args:
            pred_output (array): The model's raw output.

        Returns:
            tuple: Bounding boxes, classes, and confidence scores.
        """
        bboxes, classes, scores = self.model.postProcess(pred_output)
        return bboxes, classes, scores
    
    # Methods that can be used to call the Assembly interface on an image
    def __call__(self, image):
        """
        Makes the class callable. Simplifies the process of running an image through the entire pipeline.

        Args:
            image (array): The input image.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores.
        """
        input = self.preProcess( image=image)
        pred_output = self.forward( input=input)
        bboxes, classes, scores = self.postProcess( pred_output=pred_output)
        print(f"\n\n\n ----------------------classes after post_processing: {classes}\n\n\n")
        
        if isinstance(self.conf_thresh, float):
            bboxes, classes, scores = self.applyConfThresh(bboxes=bboxes, classes=classes, confs=scores)
        else:
            bboxes, classes, scores = self.applyClasswiseConfThresh(bboxes=bboxes, classes=classes, confs=scores)
        # res_json = self.createResJson(bboxes=bboxes, classes=classes, conf=scores)
        # custom_nms((bboxes, classes, scores))
        return bboxes, classes, scores
    
    # Method to create a resjson in the standard format
    def createResJson(self, bboxes, classes, conf):
        """
        Creates a result JSON in a standard format.

        Args:
            bboxes (list): List of bounding boxes.
            classes (list): List of class indices.
            conf (list): List of confidence scores.

        Returns:
            list: The result in JSON format.
        """
        result = dict()
        class_names = self.model.classes
        classes = [class_names[int(class_ind)] for class_ind in classes]
        for id, class_name in enumerate(classes):
            if class_name in list(result.keys()):
                result[class_name]["count"]+=1
                result[class_name]["boxes"].append(bboxes[id])
            else:
                result[class_name] = {"partName": class_name, "count": 1, "boxes":[bboxes[id]]}
        return list(result.values())
    
    # Method to apply thresholding on the results
    # Dynamic thresholding
    def applyConfThresh(self, bboxes, classes, confs):
        """
        Applies confidence thresholding to filter the model's output.

        Args:
            bboxes (list): List of bounding boxes.
            classes (list): List of class indices.
            confs (list): List of confidence scores.

        Returns:
            tuple: Filtered bounding boxes, classes, and confidence scores.
        """
        result = {"boxes":[], "classes":[], "conf":[]}
        for box, class_name, conf in zip(bboxes, classes, confs):
            if conf>=self.conf_thresh:
                result["boxes"].append(box)
                result["classes"].append(class_name)
                result["conf"].append(conf)
        return result["boxes"], result["classes"], result["conf"]
        

    # Method to apply classwise thresholding
    def applyClasswiseConfThresh(self, bboxes, classes, confs):
        """
        Applies confidence thresholding to filter the model's output.

        Args:
            bboxes (list): List of bounding boxes.
            classes (list): List of class indices.
            confs (list): List of confidence scores.

        Returns:
            tuple: Filtered bounding boxes, classes, and confidence scores.
        """
        result = {"boxes":[], "classes":[], "conf":[]}
        for box, class_name, conf in zip(bboxes, classes, confs):
        # Check if the class_name exists in self.conf_thresh
            if class_name in self.conf_thresh:
                # Apply thresholding only if class_name is in self.conf_thresh
                if conf >= self.conf_thresh[class_name]:
                    result["boxes"].append(box)
                    result["classes"].append(class_name)
                    result["conf"].append(conf)
        
        return result["boxes"], result["classes"], result["conf"]
                    

    
    

