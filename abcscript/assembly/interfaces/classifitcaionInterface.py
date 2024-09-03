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

# Inference class for Classification
class ClassificationInterface:
    """
    This class defines an interface for running classification inference 
    using a pre-loaded classification model.
    
    Attributes:
        model (object): The pre-loaded classification model for inference.
        conf_thresh (float): Confidence threshold for filtering model predictions.
        loggerObj (object): Logger object for logging information, defaults to None.
    """
    def __init__(self, model, threshold, loggerObj):
        """
        Initializes the ClassificationInterface class.

        Args:
            model (object): The pre-loaded classification model.
            threshold (float): Confidence threshold for inference.
            loggerObj (object): Logger for logging, defaults to None.
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
            tuple: Class name and confidence score.
        """
        class_name, class_conf = self.model.postProcess(pred_output)
        return class_name, class_conf
    
    # Methods that can be used to call the Assembly interface on an image
    def __call__(self, image):
        """
        Makes the class callable. Simplifies the process of running an image through the entire pipeline.

        Args:
            image (array): The input image.

        Returns:
            tuple: Class name and confidence score.
        """
        input = self.preProcess( image=image)
        pred_output = self.forward( input=input)
        class_name, class_conf = self.postProcess( pred_output=pred_output)
        # res_json = self.createResJson(class_name=class_name, class_conf=class_conf)
        return class_name, class_conf
    
    # Method to create a resjson in the standard format
    def createResJson(self, class_name, class_conf):
        """
        Creates a result JSON in a standard format.

        Args:
            class_name (str): The predicted class name.
            class_conf (float): The confidence score for the predicted class.

        Returns:
            dict: The result in JSON format.
        """
        return {"name": class_name, "conf": class_conf}
    
    # Method to apply thresholding on the results
    # Dynamic thresholding
    def applyConfThresh(self):
        # Not implemented yet
        pass