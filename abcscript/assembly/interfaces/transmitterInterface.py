import logging
import os
import threading
import time
import json
import multiprocessing
from ctypes import *
from dotenv import load_dotenv
import cv2
import numpy as np

from assembly.transmitters.Transmitter_streamer import ContinuousStreamingTransmitter
from assembly.transmitters.Transmitter_software import SoftwareTriggerTransmitter
from assembly.transmitters.Transmitter_hardware import HardwareTriggerTransmitter
from assembly.transmitters.Transmitter_GStreamer import GstreamerTransmitter






class TransmitterInterface:
    def __init__(self, shared_queue, camera_ip, transmitter_config, stop_event, transmitter_id, transmitterlog):
        self.queue = shared_queue  # Shared queue for inter-process communication
        self.camera_ip = camera_ip  # IP address of the camera
        self.transmitter_config = transmitter_config  # Configuration for the transmitter
        self.transmitter_id = transmitter_id
        self.stop_event = stop_event  # Event to signal stopping the transmitter
        self.logger = transmitterlog
        self._create_transmitter()

    def _create_transmitter(self):
        """Create the appropriate transmitter based on the configuration."""
        self.logger.info(f"Working with camera type: {self.transmitter_config['camera_type']}")
        if self.transmitter_config['camera_type'] == 'CCTV':
            self.transmitter = GstreamerTransmitter(self.queue, self.camera_ip, self.transmitter_config, self.transmitter_id, self.logger)


        elif self.transmitter_config['camera_type'] == 'streaming':
            self.logger.info(f"Transmitter Initialization started for {self.transmitter_config}")
            self.transmitter = ContinuousStreamingTransmitter(self.queue, self.camera_ip, self.transmitter_config, self.transmitter_id, self.logger )
            self.logger.info("Transmitter Initialization completed")


        elif self.transmitter_config['camera_type'] == 'Software Trigger':
            self.logger.info(f"Transmitter Initialization started for {self.transmitter_config}")
            self.transmitter = SoftwareTriggerTransmitter(self.queue, self.camera_ip, self.transmitter_config, self.transmitter_id, self.logger)
            self.logger.info("Transmitter Initialization completed")

             
        elif self.transmitter_config['camera_type'] == 'Hardware Trigger':
            self.logger.info(f"Transmitter Initialization started for {self.transmitter_config}")
            self.transmitter = HardwareTriggerTransmitter(self.queue, self.camera_ip, self.transmitter_config, self.transmitter_id, self.logger)
            self.logger.info("Transmitter Initialization completed")
    def run(self):
        """Run the transmitter until the stop event is set."""
        self.transmitter.run(self.stop_event)
        

    def stop(self):
        """Stop the transmitter."""
        self.transmitter.stop()