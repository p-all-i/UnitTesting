import os
import sys
import threading
import time
import uuid
from ctypes import *
import numpy as np
import cv2
from dotenv import load_dotenv
# from MvCameraControl_class import *
from assembly.MVcameracontrolclasscode.MvCameraControl_class import *
import multiprocessing as mp
import redis
import json
import logging
import shlex
import subprocess as sp


class GstreamerTransmitter:
    def __init__(self, shared_queue, camera_ip, transmitter_config):
        self.queue = shared_queue
        self.camera_ip =  "192.168.10.20"
        self.transmitter_config = transmitter_config
        self.gstreamer_exe = 'gst-launch-1.0'
        self.username = "admin"
        self.password = "Frinks%402020"
        self.video_path = f"rtspt://{self.username}:{self.password}@{camera_ip}:554/Streaming/Channels/101?transportmode=unicast"
        # self.video_path = f"rtspt://admin:Frinks%402020@192.168.10.20:554/Streaming/Channels/101?transportmode=unicast"
        self.height = 1080
        self.width = 1920
        self.gstreamer_source = 'rtspsrc'
        self.gstreamer_latency = 'latency=0'
        self.g_bExit = False
        # self.logger = self.setup_logging()
        self.processHandle = None
        self.frame_count = 0

    # def setup_logging(self):
    #     logger = logging.getLogger(__name__)
    #     logger.setLevel(logging.DEBUG)
    #     logger.propagate = False
    #     formatter = logging.Formatter('%(asctime)s [%(levelname)8s:%(name)1s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    #     os.makedirs("./camera_logs/", exist_ok=True)
    #     handler = logging.handlers.RotatingFileHandler("./camera_logs/camera.log", maxBytes=10000000, backupCount=5)
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)
    #     return logger

    def start_gstreamer(self):
        print("start gstreamer")
        ok = sp.Popen(shlex.split(
            f'{self.gstreamer_exe} --quiet {self.gstreamer_source} location={self.video_path} {self.gstreamer_latency} ! decodebin ! videoconvert ! video/x-raw,width={self.width},height={self.height},format=BGR ! fdsink'), stdout=sp.PIPE)
        print("object what", ok)
        return ok 
    def process_frame(self, raw_image):
        self.frame_count += 1  # Increment the frame counter
        image = np.frombuffer(raw_image, np.uint8).reshape((self.height, self.width, 3))
        id = str(uuid.uuid4())
        frame_info = {
            "id": id,
            "camera_ip": self.camera_ip,
            "image": image,
            "frame_count": self.frame_count,  # Add the frame counter to frame_info
        }
        self.queue.put(frame_info)
        # self.logger.info("Captured Frame")


        save_in = os.getenv("SAVE_DIR")
        filename = os.path.join(save_in, f"{self.frame_count}_transm.jpg")
        print("images saved")
        cv2.imwrite(filename, image)

    def run(self):
        print("1")
        self.processHandle = self.start_gstreamer()
        # time_between_restarts = 5  # Seconds to sleep between sender restarts
        print("end gs streamer")

        while not self.g_bExit:
            print("gbexit false")
            try:
                print("here")
                raw_image = self.processHandle.stdout.read(self.width * self.height * 3)
                print("going to leth condition", raw_image)
                if len(raw_image) < self.width * self.height * 3:
                    # self.logger.error(f"Frame capture failed----{self.camera_ip}")
                    print("is lenth short")
                    self.processHandle.terminate()
                    self.processHandle.stdout.close()
                    self.processHandle = self.start_gstreamer()
                    continue

                self.process_frame(raw_image)
                print("2")
            except KeyboardInterrupt:
                self.processHandle.stdout.close()
                self.processHandle.terminate()
                self.processHandle.wait()
                # self.logger.info("\nEXITING...")
                break

    def stop(self):
        self.g_bExit = True
        if self.processHandle and self.processHandle.poll() is None:
            self.processHandle.terminate()
            self.processHandle.wait()

if __name__ == "__main__":
    shared_queue = mp.Queue()
    camera_ip = sys.argv[1]
    transmitter_config = {
        # "tcp_port": sys.argv[2],
        "height": int(sys.argv[2]),
        "width": int(sys.argv[3])
        
    }

    transmitter = GstreamerTransmitter(shared_queue, camera_ip, transmitter_config)
    transmitter.run()
