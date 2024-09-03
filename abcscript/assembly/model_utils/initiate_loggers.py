import os
import cv2
import numpy as np
import logging
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
import psutil
load_dotenv()  # loading env variables


class InitLoggers:
    def __init__(self, max_bytes, backup_count, save_path):
        print(f'os.getenv("MAXBYTES_LOGGER")----------{max_bytes}')
        # Setting formatter
        self.formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s:%(name)1s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        # setting up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # wont show output on console
        os.makedirs(save_path, exist_ok=True)
        self.handler = RotatingFileHandler(os.path.join(save_path, "pythonX.log"),   
                                    maxBytes=self.max_bytes,
                                    backupCount=self.backup_count,)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        self.loop_logger = logging.getLogger("loop_logger")
        self.loop_logger.setLevel(logging.DEBUG)
        self.loop_logger.propagate = False  # wont show output on console
        self.loop_handler = RotatingFileHandler(os.path.join(save_path, "loop_log.log"), 
                                        maxBytes=self.max_bytes,
                                        backupCount=self.backup_count,)
        self.loop_handler.setFormatter(self.formatter)
        self.loop_logger.addHandler(self.loop_handler)

        self.debug_logger = logging.getLogger("debug_logger")
        self.debug_logger.setLevel(logging.DEBUG)
        self.debug_logger.propagate = False  # wont show output on console
        self.debug_handler = RotatingFileHandler(os.path.join(save_path, "debug_log.log"),   
                                            maxBytes=self.max_bytes,
                                            backupCount=self.backup_count,)
        self.debug_handler.setFormatter(self.formatter)
        self.debug_logger.addHandler(self.debug_handler)

        self.queuing_logger = logging.getLogger("queuing_logger")
        self.queuing_logger.setLevel(logging.DEBUG)
        self.queuing_logger.propagate = False  # wont show output on console
        self.queuing_handler = RotatingFileHandler(os.path.join(save_path, "queuing_log.log"),  
                                            maxBytes=self.max_bytes,
                                            backupCount=self.backup_count,)
        self.queuing_handler.setFormatter(self.formatter)
        self.queuing_logger.addHandler(self.queuing_handler)



    def log_process_info(logger, message):
        process = psutil.Process(os.getpid())
        children = process.children(recursive=True)
        logger.info(message)
        logger.info(f"Main Process ID: {process.pid}, Name: {process.name()}")
        for child in children:
            logger.info(f"Sub Process ID: {child.pid}, Name: {child.name()}, Status: {child.status()}, Started at: {time.ctime(child.create_time())}")


    def transmitter_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        formatter = logging.Formatter('%(asctime)s [%(levelname)8s:%(name)1s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        cam_logs = os.getenv("LOGS_DIR")
        # os.makedirs("./camera_logs/", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(f"{cam_logs}/camera.log", maxBytes=10000000, backupCount=5)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger