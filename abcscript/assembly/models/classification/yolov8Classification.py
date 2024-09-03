import os, sys, cv2, torch, datetime
import numpy as np
from ultralytics import YOLO


# Yolo V8 classifictaion model definition
class YoloV8Classification:
    """
    This class defines an interface for running classification inference using a pre-loaded YOLOv8 model.
    
    Attributes:
        model_weights (str): Path to the model weights.
        classes (list): List of classes the model can detect.
        imgsz (tuple): Tuple representing the image size the model expects.
        score_thresh (float): Confidence threshold for filtering model predictions.
        device (str): Device to run the inference on ("cuda" or "cpu").
        model (object): The pre-loaded YOLOv8 model.
    """
    def __init__(self, model_weights, classes, score_thresh=0.7, device="cuda", imgsz=(64, 64)):
        """
        Initializes the YoloV8Classification class.

        Args:
            model_weights (str): Path to the model weights file.
            classes (list): List of class names the model can detect.
            score_thresh (float): Confidence threshold.
            device (str): Device type ("cuda" or "cpu").
            imgsz (tuple): Image size (width, height) the model expects.
        """
        self.model_weights = model_weights # Model weights
        self.classes = classes # Classes the model have to detect
        self.imgsz = imgsz # Image size model expects
        self.score_thresh = score_thresh # Confidence threshold
        self.device = device # device to run the inference on 
        self.model = self.load_model() # Loading the yolo model
        self.to(device)

    # Method to load the yolov8 model
    def load_model(self):
        """
        Loads the YOLOv8 model.

        Returns:
            object: The loaded YOLOv8 model.
        """
        return YOLO(self.model_weights) 

    # Method to preprocess the image -- *nothing to to here
    def preProcess(self, image):
        """
        Preprocesses the image. Placeholder as no preprocessing is required here.

        Args:
            image (array): The input image.

        Returns:
            array: The same input image.
        """
        return image

    # Method to pass image through the model
    def forward(self, input):
        """
        Runs the preprocessed image through the model's forward pass.

        Args:
            input (array): The preprocessed image.

        Returns:
            array: The model's raw output.
        """
        pred = self.model.predict(device=self.device, source=input, imgsz=self.imgsz, verbose=False)
        return pred[0]

    # Method to postprocess the result in proper format
    def postProcess(self, pred_output):
        """
        Post-processes the model's raw output to get the final results.

        Args:
            pred_output (array): The model's raw output.

        Returns:
            tuple: Class name with highest confidence and its confidence score.
        """
        # Top 5 results
        class_id = pred_output.probs.top5  
        # Getting the class with highest confidence
        class_name = self.classes[class_id[0]]
        # getting the confidence value
        class_conf = float(list(np.array((pred_output.probs.top5conf.cpu())))[0])
        return class_name, class_conf

    # Method to put model to specific device
    def to(self, device="cuda"):
        """
        Moves the model to the specified device.

        Args:
            device (str): The device to move the model to ("cuda" or "cpu").
        """
        if device=="cpu":
            self.model.cpu()
        else:
            self.model.cuda()

    # Method to call for direct inferencing
    def __call__(self, image):
        """
        Makes the class callable. Simplifies the process of running an image through the entire pipeline.

        Args:
            image (array): The input image.

        Returns:
            tuple: Class name with highest confidence and its confidence score.
        """
        input = self.preProcess(image=image)
        pred_output = self.forward(input=input)
        result = self.postProcess(pred_output=pred_output)
        class_name, class_conf = result
        return class_name, class_conf