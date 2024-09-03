import os, sys, cv2, torch, datetime
import numpy as np
from ultralytics import YOLO


class YoloV8:
    """
    This class defines an interface for running object detection using a pre-loaded YOLOv8 model.
    
    Attributes:
        model_weights (str): Path to the model weights.
        classes (list): List of classes the model can detect.
        imgsz (tuple): Tuple representing the image size the model expects.
        score_thresh (float): Confidence threshold for filtering model predictions.
        iou_thresh (float): Intersection-over-Union threshold for non-max suppression.
        device (str): Device to run the inference on ("cuda" or "cpu").
        model (object): The pre-loaded YOLOv8 model.
    """

    def __init__(self, model_weights, classes, iou_thres=0.3, device="cuda", imgsz=(640, 640)):
        """
        Initializes the YoloV8 class.

        Args:
            model_weights (str): Path to the model weights file.
            classes (list): List of class names the model can detect.
            score_thresh (float): Confidence threshold.
            iou_thres (float): IOU threshold for non-max suppression.
            device (str): Device type ("cuda" or "cpu").
            imgsz (tuple): Image size (width, height) the model expects.
        """
        self.model_weights = model_weights # Model weights
        self.classes = classes # Classes the model have to detect
        self.imgsz = imgsz # Image size model expects
        self.score_thresh = float(os.getenv("YOLOV8_THRESH")) # Confidence threshold
        self.iou_thresh = iou_thres # IOU threshold
        self.device = device # device to run the inference on 
        self.model = self.load_model() # Loading the yolo model
        print(f"\n\n\n ----------------------inside yolov8 custom class model.names: {self.model.names}\n\n\n")
        
        if model_weights.endswith('.pt'):
            self.to(device)
        # self.to(device)
        self.warm_up()


    # Method to load the yolov8 model
    def load_model(self):
        """
        Loads the YOLOv8 model.

        Returns:
            object: The loaded YOLOv8 model.
        """
        return YOLO(self.model_weights, task='detect') 

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
    def forward(self, input, iou_thresh=None):
        """
        Runs the preprocessed image through the model's forward pass.

        Args:
            input (array): The preprocessed image.

        Returns:
            array: The model's raw output.
        """
        iou_th = iou_thresh if iou_thresh is not None else self.iou_thresh
        pred = self.model.predict(device=self.device, source=input, imgsz=self.imgsz, conf=self.score_thresh, iou=iou_th,  verbose=False)
        return pred[0]

    # Method to postprocess the result in proper format
    def postProcess(self, pred_output):
        """
        Post-processes the model's raw output to get the final results.

        Args:
            pred_output (array): The model's raw output.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        pred_boxes_array = [np.array(i).astype(int).tolist() for i in pred_output.boxes.xyxy.cpu()]
        scores = [float(i) for i in pred_output.boxes.conf.cpu()]
        classes = [int(i) for i in pred_output.boxes.cls.cpu()]
        print(f"\n\n\n ----------------------inside yolov8 custom class def postProcess(self, pred_output) classes: {classes}\n\n\n")

        # Getting class names
        class_names = [self.classes[class_ind] for class_ind in classes]
        print(f"\n\n\n ----------------------inside yolov8 custom class def postProcess(self, pred_output) self.class_names: {self.classes}\n\n\n")
        print(f"\n\n\n ----------------------inside yolov8 custom class def postProcess(self, pred_output) class_names: {class_names}\n\n\n")

        return pred_boxes_array, class_names, scores

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
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        input = self.preProcess(image=image)
        pred_output = self.forward(input=input)
        result = self.postProcess(pred_output=pred_output)
        boxes, class_names, scores = result
        return boxes, class_names, scores
    

    def warm_up(self):
        """
        Warms up the model by passing a few dummy images through it.
        """
        print("Warming up the model...")
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)  # Create a black dummy image of size 800x800
        for _ in range(2):  # Pass the dummy image twice
            input = self.preProcess(image=dummy_image)
            self.forward(input=input)
        print("Model warm-up completed.")
