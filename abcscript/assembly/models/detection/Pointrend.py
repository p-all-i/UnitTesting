import torch
import cv2, os, time
import numpy as np
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from assembly.models.detection.resizing import ResizeShortestEdge
from detectron2.config import get_cfg
# import PointRend project
from detectron2.projects import point_rend


class PointRend:
    """
    This class defines an interface for running object detection using a pre-loaded Pointrend model.
    
    Attributes:
        cfg (object): The configuration object for the Pointrend model.
        model (object): The pre-loaded Pointrend model.
        classes (list): The list of classes the model can detect.
        aug (object): The augmentation transform to be applied on input images.
    """
    def __init__(self, model_weights, config_path, classes, device="cuda"):
        """
        Initializes the FasterRCNN class.

        Args:
            model_weights (str): Path to the model weights.
            config_path (str): Path to the configuration file.
            classes (list): List of class names the model can detect.
            score_thresh (float): Confidence threshold for inference.
            device (str): Device to run inference ("cuda" or "cpu").
        """
        print("inside PointRend ")
        # Preparing config
        self.cfg = self.editCfg(model_weights=model_weights, config_path=config_path, classes=classes)
        # Loading model
        self.model = self.load_model()
        # Putting model to GPU
        self.to(device)
        self.classes = classes
        # Loading Aug function
        self.aug = ResizeShortestEdge([self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)
        self.warm_up()
    # Function to load the cfg file and make changes to it 
    # Depending on the params passed to it
    def editCfg(self, model_weights, config_path, classes):
        """
        Edits the configuration parameters for the model.

        Args:
            model_weights (str): Path to the model weights.
            config_path (str): Path to the config file.
            classes (list): List of classes the model can detect.
            score_thresh (float): Confidence threshold for inference.

        Returns:
            object: The updated configuration object.
        """
        cfg = get_cfg()
        cfg.PATIENCE = 10
        cfg.DATALOADER.REPEAT_SQRT = True
        # Add PointRend-specific config
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(config_path)
        # cfg.merge_from_file(params["params"]["config_path"]) 
        cfg.MODEL.WEIGHTS = model_weights   # path to the model we just trained
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(0.7)  # set a custom testing threshold
        self.classes = classes
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.classes)
        # cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(self.classes)
        return cfg
    
    # Function to load the model 
    # Based on the cfg file loaded
    def load_model(self):
        """
        Loads the Pointrend model based on the provided configuration.

        Returns:
            object: The loaded model.
        """
        model = build_model(self.cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        
        return model

    # Function that does the proper processing 
    # And converting to the proper format for model inferecing
    def preProcess(self, image):
        """
        Preprocesses the input image for inference.

        Args:
            image (array): The input image.

        Returns:
            dict: Dictionary containing the processed image and its dimensions.
        """
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        if self.input_format=='RGB':
                self.np_image = self.np_image[:,:,::-1]
        height,width = image.shape[:2]
        image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        
        inputs = {"image": image, "height": height, "width": width}
        
        return inputs

    # Passing the inputs to the model and doing the inference
    def forward(self, input):
        """
        Runs the preprocessed image through the model.

        Args:
            input (dict): Dictionary containing the processed image and its dimensions.

        Returns:
            dict: The model's raw output.
        """
        with torch.no_grad():
            predictions = self.model([input])[0]
        return predictions
    
    # Function to do the postprocessing and return output in proper format
    def postProcess(self, pred_output):
        """
        Post-processes the model's raw output.

        Args:
            pred_output (dict): The model's raw output.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        pred_output = pred_output["instances"]

        mask_array = pred_output.pred_masks.cpu().numpy()
        mask_array = np.moveaxis(mask_array, 0, -1)
        classes = pred_output.pred_classes.cpu().numpy()
        boxes = pred_output.pred_boxes.tensor.cpu().numpy().astype("int")
        scores = pred_output.scores.cpu().numpy()
        # Not returning the masks
        class_names = [self.classes[class_ind] for class_ind in classes]
        
        return boxes, class_names, scores
    
    # Method to put model to specific device
    def to(self, device="cuda"):
        """
        Moves the model to the specified device.

        Args:
            device (str): Device to move the model to ("cuda" or "cpu").
        """
        if device=="cpu":
            self.model.cpu()
        else:
            self.model.cuda()
    
    # A Method that mergers everything together
    def __call__(self, image):
        """
        Makes the class callable for easier inference.

        Args:
            image (array): The input image.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        x = time.time()
        input = self.preProcess(image=image)
        print("\n \n \n pointrend taking time for preprocessing", time.time()-x , "\n \n \n " )
        pred_output = self.forward(input=input)
        print("pointrend taking time for forward processing", time.time()-x)
        result = self.postProcess(pred_output=pred_output)
        print("pointrend taking time for poat processing", time.time()-x)
        boxes, classses, scores = result
        return boxes, classses, scores
    

    def warm_up(self):
        """
        Warms up the model by passing a few dummy images through it.
        """
        print("Warming up the model...")
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)  # Create a black dummy image of size 800x800
        for _ in range(4):  # Pass the dummy image twice
            input = self.preProcess(image=dummy_image)
            self.forward(input=input)
        print("Model warm-up completed.")
