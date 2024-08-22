import os, cv2, sys, datetime
import numpy as np
from assembly.interfaces.assemblyInterface import AssemblyInterface
from assembly.interfaces.classifitcaionInterface import ClassificationInterface
import time


# Class the defines the pipeline that needs to be run 
class inferenceInterface:
    """
    Interface for running inference on regions of interest (ROIs) in an image.
    
    Attributes:
        interface_info (dict): Information about the interface, including ROI and model details.
        ModelDict (dict): Dictionary containing pre-loaded models.
    """

    def __init__(self, interface_info, ModelDict):
        """
        Initializes the inferenceInterface class.
        
        Args:
            interface_info (dict): Configuration information for the interface.
            ModelDict (dict): Dictionary of pre-loaded models.
        """
        self.interface_info = interface_info
        self.ModelDict = ModelDict
        self.build()
        # self.x = time.time()
        


    # Method to build the inference pipeline for a particular ROI based on input JSON
    def build(self):
        """
        Builds the inference pipeline based on the provided JSON configuration.
        
        Note:
            - Adds the model(s) to be used for inference.
            - Sets up any additional settings such as ROI and threshold.
        """
        
        # setting the main roi to crop
        self.main_croping = self.interface_info["roi"]
                # Checking the type of model1 to inference 
        self.model1_type = self.interface_info["model_1"][0]["type"]
        if self.model1_type == "detection":
            # Creating Assembly interface
            
            self.model1 = AssemblyInterface(model=self.ModelDict[self.interface_info["model_1"][0]["model_id"]], threshold=self.interface_info["model_1"][0]["threshold"]["score_thresh"])
            print(f"[INFO] {datetime.datetime.now()} Loaded Assemblly Interface with modelId {self.interface_info['model_1'][0]['model_id']} for cameraID: {self.main_croping}")
        else:
            # Creating Classifictaion interface
            self.model1 = ClassificationInterface(model=self.ModelDict[self.interface_info["model_1"][0]["model_id"]], threshold=self.interface_info["model_1"][0]["threshold"]["score_thresh"], loggerObj=None)
            print(f"[INFO] {datetime.datetime.now()} Loaded Classification Interface with modelId {self.interface_info['model_1'][0]['model_id']} for cameraID: {self.main_croping}")
        # A dcitionary to save the loded model2 information based on class names
        self.model2 = dict()
        if self.interface_info.get("model_2", None) is not None:
            for model_info in self.interface_info["model_2"]:
                self.model2[model_info["class_name"]] = ClassificationInterface(model=self.ModelDict[model_info["model_id"]], threshold=model_info["threshold"]["score_thresh"], loggerObj=None)
                print(f"[INFO] {datetime.datetime.now()} Loaded Classification Interface with modelId {model_info['model_id']} for cameraID: {self.main_croping}")


    # Method to crop the roi from the image 
    # Can handle case if the roi is normalised as well
    def cropping(self, image, roi):
        """
        Crops the image based on the provided ROI
        
        Args:
            image (array): The image to be cropped.
            roi (tuple): The region of interest as (x1, y1, x2, y2).
            
        Returns:
            array: The cropped image.
        """
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2].copy()
        return crop
    
    # Main method to run a image through the built pipeline 
    def run(self, image, roi=[]):
        """
        Runs the image through the built pipeline and returns the output.
        
        Args:
            image (array): The image on which inference is to be run. # cropped according to the tracker yolo detection
            
        Returns:
            Output: The inference output wrapped in an Output object.
        """
        # Converting ROI to proper format
        main_cropping = self.checkRoi(roi=self.main_croping, image_shape=image.shape) # returning absolute value roi = shape of cropped image = [0,0, height, width]
        # roi = 0 , 0 , 1, 1 from the croppin info 
        #main cropping is [0, 0, 148, 228] because roi is 0011
        # Main cropping
        cropped_image = self.cropping(image=image, roi=main_cropping) #needed when roi =! 0011

        main_res = {"detection":{}, "classification":{}}
        
        
        if self.model1_type == "detection":
            
            # If its a detection model
            bboxes, classes, scores = self.model1(image=cropped_image)
            


            # [array([ 76,   0, 104, 228]), array([  3,  27,  38, 228]), array([ 57,  36,  99, 228]), 
            # array([ 88,   0, 135, 228]), array([ 45,  44,  73, 228]), array([ 79,  70, 119, 225]), array([ 22,  34,  59, 228]), 
            # array([ 96,   0, 120, 228])]
            # 
            # 
            # 
            #  ['4f1a9b88-2d13-4db8-94b4-87d3e7991mo2', .... '4f1a9b88-2d13-4db8-94b4-87d3e7991mo2', 
            # '4f1a9b88-2d13-4db8-94b4-87d3e7991mo2', '4f1a9b88-2d13-4db8-94b4-87d3e7991mo2'] [0.94711053, ....
            #  0.7712379, 0.768784, 0.76537377]




            # Retracing bboxes to actual image
            
            bboxes = self.retrace(top_left_corner=main_cropping[:2], bounding_boxes=bboxes) # Just passing top left of main cropping
            
            for box, class_name, score in zip(bboxes, classes, scores):
                print("[DEBUG] Processing Detection Class:", class_name)
                det_res = {}
                
                det_res["box"] = self.retrace_box(top_left_pt=roi[:2], bbox=box) # retracing to the absolute image incase of tracker is enabled
                det_res["det_score"] = score
                if class_name in list(self.model2.keys()):
                    # Cropping the object
                    cropped_object = self.cropping(image, roi=box)
                    # passing image to modelas 

                    classfication_name, class_conf = self.model2[class_name](image=cropped_object)
                    # Adding result to dict
                    det_res["class_name"] = classfication_name
                    det_res["class_score"] = class_conf
                    
                
                if class_name not in main_res["detection"].keys():
                    main_res["detection"][class_name] = [det_res]
                else:
                    main_res["detection"][class_name].append(det_res)

        else:
            class_name, class_conf = self.model1(image=cropped_image)
            main_res["classification"] = {"class_score":class_conf, "class_name": class_name, "box":self.retrace_box(top_left_pt=roi[:2], bbox=main_cropping)}
        
        
        return main_res
       
    # Method to retrace the bounding boxes to the actual image
    def retrace(self, top_left_corner, bounding_boxes):
        """
        Retrace bounding boxes coordinates to match the original image.
        
        Args:
            top_left_corner (tuple): The (x, y) coordinates of the top-left corner of the crop in the original image.
            bounding_boxes (list): List of bounding boxes, where each bounding box is represented as a list [x1, y1, x2, y2].
            
        Returns:
            list: List of retraced bounding boxes.
        """
        
        retraced_bounding_boxes = []
        for bbox in bounding_boxes:
            retraced_bounding_boxes.append(self.retrace_box(top_left_pt=top_left_corner, bbox=bbox))
            
        return retraced_bounding_boxes
    

    # Method to retrace a single box to the given point
    def retrace_box(self, top_left_pt, bbox):
        """
        Retrace a single bbox coordinate to match the original image.
        
        Args:
            top_left_corner (tuple): The (x, y) coordinates of the top-left corner of the crop in the original image.
            bounding_box (list): bbox represented as a list [x1, y1, x2, y2].
            
        Returns:
            list: retraced bounding box.
        """
        print("what is top left pt here", top_left_pt)
        if len(top_left_pt) >0:
            x, y = top_left_pt
            x1, y1, x2, y2 = bbox
            bbox = [x1 + x, y1 + y, x2 + x, y2 + y]
        return bbox
    

    # Method to check if the roi is normalised and convert it properly
    def checkRoi(self, roi, image_shape):
        """
        Checks if a given roi is normalised or not and convert it properly
        
        Args:
            roi (list): roi represented as a list [x1, y1, x2, y2].
            image_shape (tuple): height, width and channel of the image
            
        Returns:
            list: proper roi
        """
        x1, y1, x2, y2 = roi
        if all([val<=1 for val in roi]):
            height, width, _ = image_shape
            x1, y1, x2, y2 = abs(int(x1*width)), abs(int(y1*height)), abs(int(x2*width)), abs(int(y2*height))
        return [int(val) for val in [x1, y1, x2, y2]]


    

"""main_res = {
    "detection": {
        "4f1a9b88-2d13-4db8-94b4-87d3e7991mo2": [
            {
                "box": [adjusted_coordinates_for_first_bbox],  # retraced coordinates based on main cropping
                "det_score": 0.94711053,
                "class_name": class_name_1,  # result from model2
                "class_score": class_conf_1,  # result from model2
            },
            {
                "box": [adjusted_coordinates_for_second_bbox],
                "det_score": 0.93252957,
                "class_name": class_name_2,
                "class_score": class_conf_2,
            },
            # Additional detections...
        ]
    },
    "classification": {}
}"""