import logging
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
# from MvCameraControl_class import *
import multiprocessing as mp
import redis
import json
from queue import Queue
class SoftwareTriggerTransmitter:
    def __init__(self, shared_queue, camera_ip, transmitter_config, transmitter_id, transmitterlog, stop_event):

        # self.redis_queue = mp.Queue()

        self.pubsub = None
        self.redis_client = None
    


        self.queue = shared_queue
        self.camera_ip = camera_ip
        self.transmitter_id = transmitter_id
        self.transmitter_config = transmitter_config
        self.cam = None
        self.logger = transmitterlog
        self.redis_queue = mp.Queue()
        self.stop_event = stop_event
        
        self.init_redis(self.transmitter_id)
        self.g_bExit = False
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.nConnectionNum = -1
        self.nPayloadSize = 0
        self.logger = transmitterlog
        load_dotenv('../.env')
        self.processHandle = None
        self.isProcessCreated = False
        self.featureFilePath_inConfig = True
        
        



        self.frame_count = 0
        
        print("Transmitter Id::", transmitter_id)
        
        self.run(self.stop_event)
    
    def featurepathload(self):
        if self.transmitter_config["feature_path"]:
            one = os.getenv("CAMERA_CONFIGS_DIR")
            # two = os.path.join(one, "transmitterfeature")
            featureFilePath = os.path.join(one,self.transmitter_config["feature_path"])
            self.logger.info(f"what is the path {featureFilePath}")
            if len(featureFilePath) > 0:
                ret = self.cam.MV_CC_FeatureLoad(featureFilePath)
                # self.logger.info(f"file uploaded")
        return ret

    def clean_up_process(self):
        # Method to handle cleanup of processes and camera connections
        if self.processHandle is None or not self.processHandle.is_alive():
            if self.isProcessCreated:
                print("Killing process")
                self.processHandle.terminate()
                self.processHandle.join()
                self.nConnectionNum = -1
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()

    def configure_camera(self):
        configure_flag = True
        self.logger.info(f"configure_camera started")
        ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
        if self.deviceList.nDeviceNum == 0:
            configure_flag = False
            return configure_flag
        
        for i in range(self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            count = 0
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                count = count + 1
                nip1 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000) >> 24
                nip2 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000) >> 16
                nip3 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00) >> 8
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF)
                composed_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                self.logger.info(f"Found camera--{composed_ip}")
                if composed_ip == self.camera_ip:
                    self.nConnectionNum = i
                    self.logger.info(f"camera ip check{composed_ip}")

        if self.nConnectionNum == -1:
            configure_flag = False
            return configure_flag

        if self.deviceList.nDeviceNum > 0:
            if int(self.nConnectionNum) >= self.deviceList.nDeviceNum:
                self.logger.info("Input error!")
                configure_flag = False
                return configure_flag

            self.cam = MvCamera()
            stDeviceList = cast(self.deviceList.pDeviceInfo[int(self.nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.logger.info("Create handle fail! ret[0x%x]" % ret)
                
                configure_flag = False
                return configure_flag
  
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            
            self.logger.info(f"Open device {ret} not fail {self.camera_ip}" )
            if ret != 0:
                self.logger.info("Open device fail! ret[0x%x]" % ret)
                configure_flag = False
                return configure_flag
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        self.logger.info("Warning: Set Packet Size fail! ret[0x%x]" % ret)
                        configure_flag = False
                        return configure_flag
                else:
                    self.logger.info("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                    configure_flag = False
                    return configure_flag
            if self.featureFilePath_inConfig:
                ret = self.featurepathload()
                if ret != 0:
                    print("load feature fail! ret {} [0x%x]" % ret)
                    self.logger.info("load feature fail! ret[0x%x]" % ret)
                    self.logger.info(f"load feature fail! {self.camera_ip}")
                    configure_flag = False
                    return configure_flag
                else:
                    self.logger.info("load feature passed! ret[0x%x]" % ret)
            elif self.transmitter_config["streaming_config"] != "null":
                config = self.transmitter_config["streaming_config"]["properties"]
                self.configure(config)
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            if ret != 0:
                print("are you coming here in error of trigger mode")
                self.logger.info("Set trigger mode fail! ret[0x%x]" % ret)
                configure_flag = False
                return configure_flag
            ret = self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
            if ret != 0:
                self.logger.info("Could not set trigger source software")               
                configure_flag = False
                return configure_flag
            ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
            if ret != 0:
                self.logger.info("Set acquisition frame rate enable fail! ret[0x%x]" % ret)               
                configure_flag = False
                return configure_flag
            # Get payload size
            expParam = MVCC_INTVALUE()
            memset(byref(expParam), 0, sizeof(MVCC_INTVALUE))
            # Get payload size
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                self.logger.info("Get payload size fail! ret[0x%x]" % ret)        
                configure_flag = False
                return configure_flag
            self.nPayloadSize = stParam.nCurValue
            print("completed once")
        self.logger.info(f"what is configure flag {configure_flag}")
        return configure_flag

    def process_frame(self, configId):
        self.frame_count += 1  # Increment the frame counter
        numpy_array = np.array(self.pData, dtype=np.uint8)
        image = numpy_array.reshape(self.stFrameInfo.nHeight, self.stFrameInfo.nWidth)
        id = str(uuid.uuid4())
        self.logger.info(f"")
        frame_info = {
            "id": id,
            "beltId": self.transmitter_config["camera_id"],
            "camera_ip": self.camera_ip,
            "configId" : configId,
            "image": image,
            "frame_count": self.frame_count,  # Add the frame counter to frame_info
            "iterator" : 0,
            "groupId": 122,
            "groupLimit": 1,
            "extraInfo": ""
        }
        self.queue.put(frame_info)
        self.logger.info("Captured Frame")
        # save_in = os.getenv("SAVE_DIR")
        # # Save frame to disk
        # filename = os.path.join(save_in, f"{self.frame_count}_{id}_transm.jpg")
        # filename = os.path.join(save_in, f"{self.frame_count}_transm.jpg")
        # print("images saved")
        # cv2.imwrite(filename, image)
        self.logger.info(f"Saved Frame for {self.camera_ip} {self.frame_count}")

    def configure(self, config):
        properties = config.get('properties', {})
        # Direct mapping of certain keys to their specific setter methods
        key_specific_methods = {
            'Height': self.cam.MV_CC_SetIntValue,
            'Width': self.cam.MV_CC_SetIntValue,
            # 'Gamma': self.cam.MV_CC_SetFloatValue,
            'ExposureTime': self.cam.MV_CC_SetFloatValue,
            'Gain': self.cam.MV_CC_SetFloatValue,
            'TriggerMode': self.cam.MV_CC_SetEnumValue,
            'PixelFormat': self.cam.MV_CC_SetEnumValue,
            # 'gain_auto': self.cam.MV_CC_SetStringValue,
            # 'exposure_auto': self.cam.MV_CC_SetStringValue,
        }
        for key, value in properties.items():
            if key in key_specific_methods:
                # Handle cases where value might be a direct value or a dictionary
                if isinstance(value, dict):
                    value = value.get('value')
                if value is not None:
                    setter = key_specific_methods[key]  # Function to set the value
                    print("Setting:", key, "Value:", value, "Using method:", setter)
                    try:
                        setter(key, value)
                        self.logger.info(f"Set {key} to {value}")
                    except Exception as e:
                        self.logger.error(f"Error setting {key}: {str(e)}")

    def init_redis(self, transmitter_id):
        # Initialize Redis client for communication
        self.redis_client = redis.StrictRedis(host=os.getenv('REDIS_ENDPOINT'), port=os.getenv('REDIS_PORT')) # to publish the message to the channel
        
        self.redis_subscriber_thread = threading.Thread(target=self.redis_subscribe)
        self.logger.info(f"{self.redis_subscriber_thread}")
        self.redis_subscriber_thread.start()

# '''
#         # Start the queue length checker thread
#         self.queue_length_thread = threading.Thread(target=self.check_queue_length)
#         self.logger.info(f"Starting queue length checker thread: {self.queue_length_thread}")
#         self.queue_length_thread.start()
#         '''

    def redis_subscribe(self):
        # Subscribe to Redis channel for software trigger messages
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(self.transmitter_id)   #("transmitter") #("360482ad-a06c-46de-bcef-08936c96e2c9")
        t = time.time()

        for message in self.pubsub.listen():

            if message['type'] == "message":
                data = message['data'].decode('utf-8')
                if data == "Stop_Redis_Thread":
                    self.logger.info(f"Got message to stop {data}")
                    self.redis_subscriber_thread.join()
                    self.pubsub.unsubscribe(self.transmitter_id)
                    self.redis_client.close()


            if not isinstance(message['data'], int):
                # self.logger.info(f"what message {message['data']} and getting it from ---------- {self.transmitter_id}")
                self.redis_queue.put(message['data'].decode('utf-8'))

                if time.time() - t > 5 :
                    self.logger.info(f"queue size{self.redis_queue.qsize()}")
                    t = time.time()
                
                if not self.redis_queue.empty():
                    self.logger.info(f"Redis queue :: {self.redis_queue} {message['data'].decode('utf-8')} for {self.transmitter_id} ")

    def check_queue_length(self):
        while True:
            queue_length = self.redis_queue.qsize()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.logger.info(f"[{current_time}] Current redis_queue length: {queue_length}")
            time.sleep(3)


    def update_config(self, message):
        # self.logger.info("Checking for new messages in redis_queue")
        # if not self.redis_queue.empty():
            # message = self.redis_queue.get()
        self.logger.info(f"Processing message: for {self.transmitter_id} ---------and the message is -------> {message}")
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            self.logger.info("Stop grabbing fail! ret[0x%x]" % ret)
            return False
        if isinstance(message, dict):
            currCapture = message
        else:
            currCapture = json.loads(message)
        cameraConfig = currCapture["config"]
        self.configId = currCapture["config"]["configId"]
        self.configure(cameraConfig)
        # self.configure_camera_settings(cameraConfig)
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.logger.info("Start grabbing fail! ret[0x%x]" % ret)
            return False
        return True
    # else:
    #     self.logger.info("No new messages in redis_queue")  
    #     return False
    
    def run(self, stop_event):
        
        while True:
            t = time.time()
            if not stop_event.is_set():
                configure_flag = self.configure_camera()
                if not configure_flag:
                    self.logger.info(f"why false here {configure_flag}")
                    continue
                self.logger.info(f"Camera configuration completed for {self.transmitter_id}")
                self.pData = (c_ubyte * self.nPayloadSize)()
                self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
                memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
                ret = self.cam.MV_CC_StartGrabbing()
                if ret != 0:
                    self.logger.info("Start grabbing failed! ret[0x%x]" % ret)
                    continue
                self.logger.info(f"The redis queue here {self.redis_queue}")
                if self.redis_queue.empty():
                    self.logger.info(f"redis is empty after configuration ")

                while not stop_event.is_set():
                    # self.logger.info(f"The redis queue here {self.redis_queue}")
                    if not self.redis_queue.empty():
                        message = self.redis_queue.get()
                        Update_status = self.update_config(message)
                        if Update_status == False:
                            self.logger.info("Setting Config Fail in update, Loosing last trigger")
                            break
                        ret=self.cam.MV_CC_SetCommandValue("TriggerSoftware")
                        if ret!=0:
                            self.logger.info("Loosing the last trigger ---> command value error[0x%x]" % ret)
                            break
                        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.pData), self.nPayloadSize, self.stFrameInfo, 20000)
                        if ret != 0:
                            self.logger.info(f"Loosing the last trigger --->failed to capture {ret} ")
                            break
                        if ret == 0:
                            self.process_frame(self.configId)
                            x = time.time() - t

                            self.logger.info(f"what is time after capture ")
                        
                    # else:
                    #     break
            else:
                break
        self.stop()
    def stop(self):
        """Stop the transmitter and release resources."""
        t = time.time()
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            x = time.time() - t 
            self.logger.info(f"stoppimg the Grabbing , and time taken --> {x}")
            self.cam.MV_CC_CloseDevice()
            x = time.time() - t 
            self.logger.info(f"CloseDevice, and time taken --> {x}")
            self.cam.MV_CC_DestroyHandle()
            x = time.time() - t 
            self.logger.info(f"DestroyHandle, and time taken --> {x}")
        x = time.time() - t 
        self.logger.info(f"Total stopping time taken --> {x}")

        self.pubsub = self.redis_client.publish(self.transmitter_id, message = "Stop_Redis_Thread" )










        # self.stop_event_redis.set()
        # self.redis_subscriber_thread.join()


        # # Optional: Clean up Redis client if needed
        # self.redis_client.close()
        # self.logger.info("Transmitter stopped and resources released.")



            



#  File "/python-transmitter/scripts/assembly/transmitters/Transmitter_software.py", line 280, in redis_subscribe
#     self.redis_subscriber_thread.join()
#   File "/opt/conda/lib/python3.10/threading.py", line 1093, in join
#     raise RuntimeError("cannot join current thread")
# RuntimeError: cannot join current thread
# Transmitter Id:: 04c952fc-953d-4edb-9088-08f3782bf4bc
# completed once
# Exception in thread Thread-5 (redis_subscribe):
# Traceback (most recent call last):
#   File "/opt/conda/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
#     self.run()
#   File "/opt/conda/lib/python3.10/threading.py", line 953, in run
#     self._target(*self._args, **self._kwargs)
#   File "/python-transmitter/scripts/assembly/transmitters/Transmitter_software.py", line 280, in redis_subscribe
#     self.redis_subscriber_thread.join()
#   File "/opt/conda/lib/python3.10/threading.py", line 1093, in join
#     raise RuntimeError("cannot join current thread")
