import os, sys, requests, json, time
import threading
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv("./.env")
# time.sleep(300000)
import datetime, logging, traceback, sys, torch, cv2
from assembly.model_utils.initialize import  GlobalParameters, extractFrameVCO, select_device, DumpModels, getConfigData
from assembly.model_utils.initiate_loggers import InitLoggers 
from assembly.model_utils.Visualisor import VisualizeResults
from assembly.components.FileVideoStream import NodeCommServer, varientchange_server
from assembly.analysis.analysis_logic import AnalysisLogic
from assembly.interfaces.InterfaceCreation import InterfaceCreation
from assembly.input_validation.validation import validateInput
from assembly.components.mainProcessor import MainProcessor
from assembly.components.prepOutput import OutputPrep

from assembly.components.transmitterManager import TransmitterManager
# from configupdate import ConfigUpdater
import multiprocessing as mp




# for multi threading
import multiprocessing
# from transmitter import Transmitter 
# Fix the random seed for reproducibility
torch.manual_seed(0)
# Select the device for running the models (typically CPU or GPU)
device = select_device(device='')

## Loading environment variables
max_bytes = int(os.getenv("MAXBYTES_LOGGER"))
backup_count = int(os.getenv("BACKUPCOUNT_LOGGER"))
EXCHANGE_PUBLISH = os.getenv("EXCHANGE_PUBLISH")
QUEUE_PUBLISH = os.getenv("QUEUE_PUBLISH")
url1 = os.getenv('initial_data_endpoint')

dockerid = os.environ.get("DOCKER_ID")

# if dockerid is None:
#     dockerid = os.getenv("DOCKER_ID")
    


url = f"{url1}?dockerId={dockerid}"
pika_host = os.getenv('pika_host')
# Redis host
redis_host = os.getenv("redis_host")

# Saving path  
SAVE_DIR = os.getenv("SAVE_DIR")
# Model weights DIR
MODELS_DIR = os.getenv("MODEL_WEIGHTS_DIR")
# LOGS DIR
LOGS_DIR = os.getenv("LOGS_DIR")

# ImageQserver params
IMAGESERVER_HOST = os.getenv("IMAGESERVER_HOST")
IMAGESERVER_PORT = os.getenv("IMAGESERVER_PORT")
# Consuming Queue
# Fetch the name of the queue from the command-line arguments
# Fetch the name of the queue from the command-line arguments
CONSUMING_ROUTING_KEYS = sys.argv[1:]

# Initialize the logger for logging various operations and errors
loggerObj = InitLoggers(max_bytes, backup_count, save_path=LOGS_DIR)
# Load class UUIDs and UUID-Class Map from the input data

# Initialize the analysis logic
analysisLogic = AnalysisLogic(loggerObj=loggerObj) # Analysis logic
# createOutputs = CreateOutputs() # Outputclass creates
outputprepObj = OutputPrep()
# Fetch the configuration data from the backend
CONFIG_JSON = getConfigData().getData(loggerObj=loggerObj, url= url)
#-------------------------------------------------------------------------------

CONFIG_JSON = getConfigData().update_config_classes(CONFIG_JSON, loggerObj)

print("config", CONFIG_JSON)
classes = CONFIG_JSON['modelsInfo'][0]['params']
uuid_class_map = CONFIG_JSON['modelsInfo'][0]['uuid_class_map']
# Instantiate the visualization utility for results
Visualisor = VisualizeResults(uuid_class_map=uuid_class_map)



# # # Validating the received data
validation_res = validateInput.validate_main(inputdata=CONFIG_JSON, loggerObj=loggerObj)
if validation_res:
    print(f"[INFO] {datetime.datetime.now()} Input validation completed Succesfully")
    loggerObj.logger.info(f"Input validation completed Succesfully")
else:
    print(f"[INFO] {datetime.datetime.now()} Input validation failed!!!")
    loggerObj.logger.exception(f"Input validation failed!!!")
    sys.exit(1)




# Initialize global parameters (GP) using the configuration data
try:
    # MainGP
    GP = GlobalParameters(device=device, CONFIG_DATA=CONFIG_JSON, loggerObj=loggerObj, EXCHANGE_PUBLISH= EXCHANGE_PUBLISH, QUEUE_PUBLISH= QUEUE_PUBLISH, MODELS_DIR=MODELS_DIR)
    print(f"[INFO] {datetime.datetime.now()} GLOBAL PARAMETERS ARE LOADED.")
    loggerObj.logger.info(f"GLOBAL PARAMETERS ARE LOADED.")
except Exception as e:
    traceback.print_exception(*sys.exc_info())
    loggerObj.logger.exception(f'Please check the GLOBAL PARAMETERS or SCP and Socket connections!!! {e}')
    print(f"[ERROR] {datetime.datetime.now()} {e}" )
    print(f"[INFO] {datetime.datetime.now()} Please check the GLOBAL PARAMETERS or SCP and Socket connections!!!")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Print traceback to a string
    traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    # Log the traceback string
    print(f"[ERROR] {datetime.datetime.now()} {e}" )
    loggerObj.logger.error("-----------------------------------")
    loggerObj.logger.error("An error occurred:\n%s", traceback_string)
    loggerObj.logger.error("-----------------------------------")
    sys.exit(1)


# Create interfaces for each of the camera id, based on the config file
try:
    # Interfaces
    GP = InterfaceCreation().create(GP=GP, loggerObj=loggerObj)
    print("[INFO] Created Interfaces for: ", list(GP.interfaceObjs.keys()))
    print(f"[INFO] {datetime.datetime.now()} Interface Creation Completed")
    loggerObj.logger.info(f"Interface Creation Completed")
except Exception as e:
    traceback.print_exception(*sys.exc_info())
    loggerObj.logger.exception(f'Interface Creation Failed!!! {e}')
    print(f"[ERROR] {datetime.datetime.now()} {e}" )
    print(f"[INFO] {datetime.datetime.now()} Interface Creation Failed!!! ")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Print traceback to a string
    traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    # Log the traceback string
    print(f"[ERROR] {datetime.datetime.now()} {e}" )
    loggerObj.logger.error("-----------------------------------")
    loggerObj.logger.error("An error occurred:\n%s", traceback_string)
    loggerObj.logger.error("-----------------------------------")
    sys.exit(1)

consuming_queue = os.getenv("CONSUME_QUEUE")
# Server for handling messageing between rabbitmq and python

thread_master = None  # Initialize thread_master outside try block
transmitterlog = loggerObj.transmitter_logging()

#
try:
    # ImageQserver Starting
    output_sender = NodeCommServer(exchange_publish_name=GP.EXCHANGE_PUBLISH, publishing_queue=GP.QUEUE_PUBLISH, consuming_queue=consuming_queue, host=pika_host, loggerObj=loggerObj)
    output_sender.start()

    varientchangeServer = varientchange_server(exchange_publish_name= "update_config", publishing_queue = dockerid, consuming_queue= dockerid, host=pika_host, loggerObj=loggerObj)
    varientchangeServer.start()
    # Initialize and start the transmitter manager
    transmitter_manager = TransmitterManager(CONFIG_JSON,transmitterlog)
    transmitter_manager.start_transmitters()
    
    # Get the shared queue from the transmitter manager
    shared_queue = transmitter_manager.shared_queue
    
    loggerObj.logger.info("thread_master(FileVideoStream) created")
    print(f"[INFO] {datetime.datetime.now()} thread_master(FileVideoStream) created ")
except Exception as e:
    error_msg = f"Error at FileVideoStream initiation: {e}"
    print(error_msg)
    loggerObj.logger.exception(error_msg)  # Log exception with traceback
    print(f"[ERROR] {datetime.datetime.now()} {error_msg}")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print(exc_type, exc_value)  # Print exception type and value for immediate insight
    thread_master = None  # Ensure thread_master is properly initialized on error


# Main
main_processor = MainProcessor(GP, loggerObj, analysisLogic, output_sender, outputprepObj, Visualisor, SAVE_DIR, shared_queue, transmitter_manager,varientchangeServer)

print(f"[INFO] {datetime.datetime.now()} Starting Main python script")
main_processor.run()