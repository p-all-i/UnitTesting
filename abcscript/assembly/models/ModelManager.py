import os, sys, time, datetime, json
import numpy as np
from collections import OrderedDict
from assembly.models.detection.Frcnn_model import FasterRCNN
from assembly.models.detection.Pointrend import PointRend
from assembly.models.detection.Yolov8_model import YoloV8
from assembly.models.classification.yolov8Classification import YoloV8Classification
import struct
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from io import BytesIO
import yaml
import torch, tempfile

# Class that is responsible for loading the models
class modelManager:
    """
    This class manages the loading of machine learning models based on their unique identifiers (UUIDs)
    and other provided parameters.
    
    Attributes:
        loggerObj (object): An object for logging various steps and errors.
        model_params (dict): A dictionary containing the properties and parameters for each model to be loaded.
    """

    def __init__(self, model_params, loggerObj, MODELS_DIR):
        """
        Initializes the modelManager class.

        Args:
            model_params (dict): The parameters for each model, indexed by UUIDs.
            loggerObj (object): The logging object.
        """
        self.loggerObj = loggerObj
        self.model_params = model_params
        self.MODELS_DIR = MODELS_DIR
        print(f"[INFO] {datetime.datetime.now()} ModelManager started")
        self.loggerObj.logger.info("[INFO] ModelManager started")
        self.model_key = os.environ.get("MODEL_KEY")
        self.config_key = os.environ.get("MODEL_KEY")
        print("model_params:::::" , self.model_params)




    @staticmethod
    def decryption_config(encrypted_file_path, key):
        file_name = encrypted_file_path.split("/")[-1].split(".")[0]

        if not os.path.exists(encrypted_file_path): raise Exception(f" {encrypted_file_path} File not found")
        
        with open(encrypted_file_path, 'rb') as file:
            print("config decryption started")
            iv = file.read(16)
            print("iv reading")
            print(f"what is iv {iv}")
            encrypted_data = file.read()
            print("encryption read")

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(encrypted_data)
        decrypted_data = unpad(decrypted_data, AES.block_size)
        print("unpadded sucessfully")

        print("what is file name", file_name)
        # if file_name == "args":
        print("config decryption ended")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='wb')
        temp_file.write(decrypted_data)
        temp_file.seek(0)  # Move back to the start of the file to read it
        print(f"weights after dcr {temp_file}")
        
        return temp_file.name







    @staticmethod
    def decryption_model(encrypted_file_path, key):
        file_name = ".".join(encrypted_file_path.split("/")[-1].split(".")[:-1])

        if not os.path.exists(encrypted_file_path): raise Exception(f" {encrypted_file_path} File not found")
        
        with open(encrypted_file_path, 'rb') as file:
            print("weights decryption started")
            iv = file.read(16) #b"\xce5\xa8&\x19&\xaf\x9e\x89\x88\x02`\xcc\xef\x08Q" #file.read(16)
            # print("iv reading")
            # print(f"what is iv {iv}")
            encrypted_data = file.read()

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(encrypted_data)
        decrypted_data = unpad(decrypted_data, AES.block_size)

        print("what is file name", file_name)


        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth', mode='wb')
        temp_file.write(decrypted_data)
        temp_file.seek(0)
        print(f"model after dcr {temp_file.name}")
        return temp_file.name



    def validate_yaml(self, file_path):
        print("i am here")
        with open(file_path, 'r') as file:
            try:
                yaml.safe_load(file)
                print("YAML file is valid")
            except yaml.YAMLError as exc:
                print("Error in YAML file:", exc)





    def load_models(self, model_dict, active_models, device="cuda"):
        """
        Load the models specified in the active_models list and populate the model_dict.

        Args:
            model_dict (dict): The dictionary to populate with loaded models, indexed by UUIDs.
            active_models (list): A list of UUIDs specifying which models should be active.
            device (str): The device where the models should be loaded ("cuda" or "cpu").

        Returns:
            dict: The updated model_dict containing the loaded models.
        """
        # Iterating through the model lists in GP
        for uuid, params in self.model_params.items():
            model_type = params["model_properties"]["type"]
            model_key = params["model_properties"]["modelKey"]

            print(f"where is my config {params}")


            # Skip loading if model is not in the list of active models or already loaded
            if (uuid not in active_models) or (uuid in model_dict.keys()):
                continue
            
            if model_key == 1: # Faster RCNN


                model_weights = os.path.join(self.MODELS_DIR, params["model_path"])
                print(f"checking weights path {model_weights}")
                if model_weights.endswith(".bin"):
                    # model_weights = os.path.join(os.getenv('MODEL_DIR'), 'best.bin')
                    weights_key = b'-\x01\xe0\xcb>\x1cq;\xa1\xd5U\xda\xf2\x81\xfa\xeb\xa3\xc8#V\xe43\x9e+\xaf\\ ,\x1c^\x06\xf6'

                    # weights_key = bytes.fromhex(self.model_key)
                    # weights_key = b'\xc1\x04/Y\x92\xcap6\x81V\x99\x96\x88?=\x17]!/N\xfd\xa8>\x15S4\xd7,\xf6]\xd0G'
                    model_weights = self.decryption_model(model_weights, weights_key)
                else:
                    print("are you coming in weights")
                    model_weights = model_weights

                config_path = os.path.join(self.MODELS_DIR, params["model_properties"]["config"])
                print(f"what is config path {config_path}")
                if config_path.endswith(".bin"):
                    # config_path = os.path.join(os.getenv('MODEL_DIR'), 'args.bin')
                    config_key = b"o\x8aUS\xf1\xc6`\xd4Oo\x9a\xacl\x8a\x06\xd9g\xbdL\x91\x8cw\xdd\xe8\xf5\xa0\xc32-\xcah\x11"
                    # config_key = bytes.fromhex(self.config_key)
                    # config_key = b'a\x114\x06d\xdb?l\xa5bf\\\x9a\xd8\x0f\xba\xe6\xbf&\xbe\xcfk\x19\x8a\xc66\xa5|@\xabw\xb5'

                    config_path = self.decryption_config(config_path, config_key)
                    print
                    self.validate_yaml(config_path)
                else:
                    print("are you coming in config")
                    config_path = config_path



                print(f"what is model weights and config {model_weights} {config_path}")
                model_dict[uuid] = FasterRCNN(model_weights=model_weights, config_path=config_path, classes= params["model_properties"]["params"]["classes"], device=device)

                self.loggerObj.logger.info("[INFO] FasterRCNN model loaded with uuid {uuid}")
                print(f"[INFO] {datetime.datetime.now()} FasterRCNN model loaded with uuid {uuid}")

            elif model_key == 0: # YoloV8 Detection model
                # print("yolo model params: ", os.path.join(self.MODELS_DIR, params["model_path"]),  params["model_properties"]["params"])["clases"])
                classes_print = params["model_properties"]["params"]["classes"]
                model_dict[uuid] = YoloV8(model_weights=os.path.join(self.MODELS_DIR, params["model_path"]), classes= classes_print, device=device) # Need to add rest of the params later
                print("------------------------------------")
                print(f"[INFO] {datetime.datetime.now()} initial classes {classes_print}")
                print("------------------------------------")
                self.loggerObj.logger.info("[INFO] YoloV8-det model loaded with uuid {uuid}")
                print(f"[INFO] {datetime.datetime.now()} YoloV8-det model loaded with uuid {uuid}")

            elif model_key == 2: # Pointrend model

                model_weights = os.path.join(self.MODELS_DIR, params["model_path"])
                self.loggerObj.logger.info(f"checking config path {model_weights}")
                print(f"checking config path {model_weights}")
                if model_weights.endswith(".bin"):
                    # model_weights = os.path.join(os.getenv('MODEL_DIR'), 'best.bin')
                    weights_key = b"\x06C3\x13x[D\x12\xd3I\x1b\xed\x089\x04\x02G\xf3\x92\x8e\xfb\x1a5g\x8f\x95\x16\xe3$/\x1c\xe5"
                    # weights_key = bytes.fromhex(self.model_key)
                    # weights_key = b'\xc1\x04/Y\x92\xcap6\x81V\x99\x96\x88?=\x17]!/N\xfd\xa8>\x15S4\xd7,\xf6]\xd0G'
                    model_weights = self.decryption_model(model_weights, weights_key)
                else:
                    model_weights = model_weights

                config_path = os.path.join(self.MODELS_DIR, params["model_properties"]["config"])
                self.loggerObj.logger.info(f"checking config path {config_path}")
                if config_path.endswith(".bin"):
                    # config_path = os.path.join(os.getenv('MODEL_DIR'), 'args.bin')
                    # config_key = bytes.fromhex(self.config_key)
                    config_key = b"\x06C3\x13x[D\x12\xd3I\x1b\xed\x089\x04\x02G\xf3\x92\x8e\xfb\x1a5g\x8f\x95\x16\xe3$/\x1c\xe5"
                    config_path = self.decryption_config(config_path, config_key)
                    # self.validate_yaml(config_path)
                else:
                    config_path = config_path

                print("pointrend called in modelManager")
                model_dict[uuid] = PointRend(model_weights=model_weights, config_path=config_path, classes= params["model_properties"]["params"]["classes"], device=device)

                self.loggerObj.logger.info("[INFO] PointRend model loaded with uuid {uuid}")
                print(f"[INFO] {datetime.datetime.now()} PointRend model loaded with uuid {uuid}")


            elif model_key == 3: # YoloV8 classifictaion model
                model_dict[uuid] = YoloV8Classification(model_weights=os.path.join(self.MODELS_DIR, params["model_path"]), classes= params["model_properties"]["params"]["classes"], device=device)# Need to add rest of the params later
 
                self.loggerObj.logger.info("[INFO] YoloV8-classification model loaded with uuid {uuid}")
                print(f"[INFO] {datetime.datetime.now()} YoloV8-classifiction model loaded with uuid {uuid}")

            else:
                self.loggerObj.logger.exception("[INFO] Tried loading a model that is not implementing")
                print(f"[INFO] {datetime.datetime.now()} Tried loading a model that is not implementing")
                raise Exception("[INFO] Model not implemented")
            
        print("what is model dict", model_dict)
        return model_dict
    


    def WC_forUT(self):

        for uuid, params in self.model_params.items():
            model_type = params["model_properties"]["type"]
            model_key = params["model_properties"]["modelKey"]

            weights_key = os.path.join(self.MODELS_DIR, params["model_path"])
            config_key = os.path.join(self.MODELS_DIR, params["model_properties"]["config"])
            weights_key = b"\x06C3\x13x[D\x12\xd3I\x1b\xed\x089\x04\x02G\xf3\x92\x8e\xfb\x1a5g\x8f\x95\x16\xe3$/\x1c\xe5"
            config_key = b"\x06C3\x13x[D\x12\xd3I\x1b\xed\x089\x04\x02G\xf3\x92\x8e\xfb\x1a5g\x8f\x95\x16\xe3$/\x1c\xe5"
            model_weights = self.decryption_model(model_weights, weights_key)
            config_path = self.decryption_config(config_path, config_key)
            classes = params["model_properties"]["params"]["classes"]
            
            return model_weights, config_path, classes

        raise ValueError("No model parameters found.")








    

