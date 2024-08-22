from typing import Any
from dotenv import load_dotenv
import os, requests, sys, cv2, datetime, traceback, torch, gc, base64, time
from collections import OrderedDict
import numpy as np
from assembly.models.ModelManager import modelManager
from assembly.models.TrackerManager import trackerManager
from assembly.interfaces.InterfaceCreation import InterfaceCreation
from assembly.input_validation.schema import cameraInfo_schema
from assembly.input_validation.validation import validateInput


load_dotenv()  

# Class that store all the necessary infomration from models to inference pipelines
class GlobalParameters():
    """
    A class responsible for managing and storing necessary configurations
    ranging from models to inference pipelines for the application.

    Attributes:
    - model_params (dict): Metadata for models.
    - active_models (list): List of currently active models.
    - ModelDict (OrderedDict): Dictionary holding loaded models.
    - TrackerDict (OrderedDict): Dictionary holding loaded trackers.
    """
    def __init__(self, device, CONFIG_DATA, loggerObj, EXCHANGE_PUBLISH, QUEUE_PUBLISH, MODELS_DIR):
        """
        Initializes the global parameters from configuration data.

        Args:
        - device (str): The device on which the models will be loaded.
        - CONFIG_DATA (dict): Configuration data containing camera, tracker, and model information.
        - loggerObj (object): Object to handle logging activities.
        """
        # --------------   GLOBAL VARIABLES   ---------------------
        load_dotenv()  # loading .env file
        # IP parameters
        # self.SOCKET_URL = os.getenv("SOCKET_I6P_ADDRESS")
        self.device = device
        self.loggerObj = loggerObj
        self.ModelDict = OrderedDict()
        self.TrackerDict = OrderedDict()
        self.MODELS_DIR = MODELS_DIR
        self.active_models = None
        # Data about each of the belts(inference pipelines)
        self.cameraParams = CONFIG_DATA.get("cameraInfo", {})
        # meta data about the trackers
        self.tracker_params = CONFIG_DATA.get("trackersInfo", [])
        
        # Loading model configs - model meta data
        self.loadModelparmas(MODEL_DATA=CONFIG_DATA.get("modelsInfo", []))

        # Loading Modeldict
        self.loadModels()

        # Loading Trackerdict
        self.loadTrackers()

        # Loading RabbitMQ Params
        self.EXCHANGE_PUBLISH = EXCHANGE_PUBLISH
        self.QUEUE_PUBLISH = QUEUE_PUBLISH

        print(f"[INFO] {datetime.datetime.now()} GLOBAL PARAMETERS FOR Assemby Loaded!")
        print(f"[INFO] {datetime.datetime.now()} Loaded GP configs ")
        self.loggerObj.logger.info(f"GLOBAL PARAMETERS FOR Assembly LOADED!!")
    
    # Method to load the model config data into proper format in GP
    def loadModelparmas(self, MODEL_DATA):
        """
        Organizes the model data in a dictionary format for easy access.

        Args:
        - MODEL_DATA (dict): Raw model data to be organized.
        """
        self.model_params = {}
        for model_data in MODEL_DATA:
            self.model_params[model_data["id"]] = {"model_path": model_data["weights"], "model_properties": model_data}

    def get_model_params(self, model_id):
        return self.model_params.get(model_id, {})
    
    # method to load the models
    def loadModels(self):
        """Load the models based on active configurations."""
        # Current active models 
        self.active_models = self.extract_values()
        # Loading the model dictionary
        self.ModelDict = modelManager(model_params=self.model_params, loggerObj=self.loggerObj, MODELS_DIR=self.MODELS_DIR).load_models(model_dict=self.ModelDict, active_models=self.active_models, device="cuda")
        # # Warm up each model
        # for model_id, model in self.ModelDict.items():
        #     print(f"Warming up model: {model_id}")
        #     model_warmup = ModelWarmUp(model)
        #     model_warmup.warm_up_model()
    
    # method to load the trackers
    def loadTrackers(self):
        """Load the trackers based on active configurations."""
        self.active_trackers = self.extract_active_trackers()
        self.DumbTrackers(active_trackers=self.active_trackers)
        self.TrackerDict = trackerManager(loggerObj=self.loggerObj).load_trackers(trackerDict=self.TrackerDict, active_trackers=self.active_trackers)

    def read_classes_file(self, classes_file):
        try:
            with open(classes_file, 'r') as f:
                classes = f.read().strip().split('\n')
            return classes
        except FileNotFoundError:
            print(f"Error: {classes_file} not found.")
            return []
        except IOError as e:
            print(f"Error reading {classes_file}: {e}")
            return []
        

    # Method to update the camera params
    def updatecamerParams(self, cameraParams):
        """
        Update the camera parameters with new configurations.

        Args:
        - cameraParams (dict): New camera configuration data.
        """
        for cameraId in cameraParams.keys():
            self.cameraParams[cameraId] = cameraParams[cameraId]

    # Method to extract active models 
    def extract_values(self, key="model_id"):
        """Recursively pull values of specified key from nested JSON."""
        # takes out all the models that are in camera info
        arr = []
        def extract(cameraParams, arr, key):
            """Recursively search for values of key in JSON tree."""
            if isinstance(cameraParams, dict):
                for k, v in cameraParams.items():
                    if isinstance(v, (dict, list)):
                        extract(v, arr, key)
                    elif k == key:
                        arr.append(v)
            elif isinstance(cameraParams, list):
                for item in cameraParams:
                    extract(item, arr, key)
            return arr
        # list of all active models
        activeModels = extract(self.cameraParams, arr, key)

        return list(set(activeModels))
    
    # Method to extract the cameras with trackers activate
    def extract_active_trackers(self):
        """
        Identify cameras with active trackers.

        Returns:
        - List of camera IDs with active trackers.
        """ #### ADD ERROR HANDLING HERE ####
        active_trackers = {}
        for cameraID in self.cameraParams.keys():
            for camerainfo in self.cameraParams[cameraID]:
                if "tracker" in camerainfo["steps"]:
                    trackerinfo = camerainfo["tracker"]["roi"]
                    # if cameraID in list(active_trackers.keys()):
                    #     active_trackers[cameraID].append(trackerinfo)
                    # else:
                    active_trackers[cameraID] = trackerinfo
        print("why no tracker 1? " , active_trackers)
        return active_trackers
    
    # Function to remove the loaded trackers
    def DumbTrackers(self, active_trackers):
        for camera_id in list(self.TrackerDict.keys()):
            if camera_id not in list(active_trackers.keys()):
                del self.TrackerDict[camera_id]

    
    
        
# Class that extracts frame from the Queue and provides to the main loop
class extractFrameVCO:
    """
    A class responsible for extracting frames from the Queue and providing 
    them to the main loop for further processing.
    """
    
    # Main function
    @staticmethod
    def read(thread_master):
        """
        Extracts message from the thread master
        and returns the image along with its metadata.
        
        Args:
        - thread_master (object): The master thread from which the message is to be read.
        
        Returns:
        - dict: Dictionary containing the camera ID as the key, and a sub-dictionary 
                with the image, group_id, and iterator as values.
        """
        # Initialize an ordered dictionary to store the image and its metadata.
        img_master = OrderedDict()
        # Reading the message from the thread master
        message = thread_master.read()

        if len(message)>0:
            img_master[message[0]["cameraId"]] = message[0].copy()

        return img_master
    

class getConfigData:
    """
    A class to fetch configuration data from a provided endpoint.
    """

    @staticmethod
    def getData(loggerObj, url):
        """
        A static method to retrieve configuration data from a predefined endpoint.
        
        Args:
        - loggerObj (object): Logger object to log information and errors.
        
        Returns:
        - dict: Dictionary containing configuration data fetched from the endpoint.
        """
        while True:  # Start an infinite loop for retry mechanism
            try:
                # Fetching data from the predefined endpoint
                response = requests.get(url=f"{url}")
                response.raise_for_status()  # Will raise an exception for HTTP error codes
                CONFIG_JSON = response.json()["data"]
                
                # Logging the fetched configuration data
                logger_message = f"Config data received \n\n {' - - - '*9}"
                print(f"[INFO] {datetime.datetime.now()} {logger_message}")
                loggerObj.logger.info(logger_message)
                
                return CONFIG_JSON  # Break the loop and return if successful
            except Exception as e:
                # Log the exception details
                # traceback.print_exception(*sys.exc_info())
                logger_message = f'Exception occurred while receiving config data: {e}'
                print(f"[ERROR] {datetime.datetime.now()} {logger_message}")
                loggerObj.logger.error(logger_message)
                print(f"[ERROR] {datetime.datetime.now()} Server not running")
                loggerObj.logger.error("Server not running")
            
            # Wait for 5 seconds before retrying
            time.sleep(5)


    def read_uuids_from_file(self, file_path):
        
        one = os.getenv("MODEL_WEIGHTS_DIR")
        
        file_path = os.path.join(one, file_path)
        
        with open(file_path, 'r') as file:
            
            uuids = file.read().splitlines()
        return uuids
    


    def update_config_classes(self, CONFIG_JSON, loggerObj):
        """
        Updates CONFIG_JSON by replacing class file paths with actual UUIDs from the files.
        """
        

        for model in CONFIG_JSON.get("modelsInfo", []):
            class_file_path = model["params"].get("classes")
            
            # if class_file_path and os.path.exists(class_file_path):
            uuids = self.read_uuids_from_file(class_file_path)
            model["params"]["classes"] = uuids

        return CONFIG_JSON
    







# Function to remove the loaded models from GPU
def DumpModels(GP, active_models):
    # Set models to CPU and delete them
    for model_id in list(GP.ModelDict.keys()):
        #for model_id in list(GP.ModelDict[model_type].keys()):
        if model_id not in active_models:
            GP.ModelDict[model_id].to("cpu")# putting model to GPU
            del GP.ModelDict[model_id] # deleting model

    # Clear any unused memory in PyTorch
    torch.cuda.empty_cache()


# Funtion that takes in the GP and varient change INFO
# Makes the necessary changes
def updateVariants(GP, CONFIG_JSON, loggerObj):
    cameraJson = CONFIG_JSON
    # Validating the cameraInfo reached
    status, reason= validateInput.validate(schema=cameraInfo_schema, json=cameraJson)
    if status:
        print(f"[INFO] {datetime.datetime.now()} Varient change input validation Successfull")
        loggerObj.logger.info(f"Varient change input validation Done")
    else:
        print(f"[ERROR] {datetime.datetime.now()} Varient change input validation Failed!!!")
        loggerObj.logger.info(f"Varient change input validation Failed!!!")
        print(f"[ERROR] {datetime.datetime.now()} Varient change input validation Failed!!!")
        print(f"[ERROR] {datetime.datetime.now()} Resaon: {reason}")
        loggerObj.logger.info(f"[ERROR] {datetime.datetime.now()} Resaon: {reason}")
        Exception("Varient change input validation Failed!!!") 

    GP.updatecamerParams(cameraParams=cameraJson)
    print(f"[INFO] {datetime.datetime.now()} Updated CameraParams in GP")
    loggerObj.logger.info(f"Updated CameraParams in GP")
    # Getting current active models
    new_models = GP.extract_values()
    print(f"[INFO] {datetime.datetime.now()} Extracted New Working models")
    loggerObj.logger.info(f"Extracted New Working models")
    # Dumping old models
    DumpModels(GP=GP, active_models=new_models)
    print(f"[INFO] {datetime.datetime.now()} Dumped Unused Models")
    loggerObj.logger.info(f"Dumped Unused Models")
    # Updating model dict
    GP.loadModels()
    print(f"[INFO] {datetime.datetime.now()} Loaded New Models")
    loggerObj.logger.info(f"Loaded New Models")
    # Updating TrackerDict
    GP.loadTrackers()
    print(f"[INFO] {datetime.datetime.now()} Loaded New Trackers")
    loggerObj.logger.info(f"Loaded New Trackers")
    # Loading the new Interface objects
    GP = InterfaceCreation().create(GP=GP, loggerObj=loggerObj)
    print(f"[INFO] {datetime.datetime.now()} Loaded New Interfaces")
    loggerObj.logger.info(f"Loaded New Interfaces")


# Function to select device available
def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        arg = 'mps'
    else:  # revert to CPU
        arg = 'cpu'

    return torch.device(arg)