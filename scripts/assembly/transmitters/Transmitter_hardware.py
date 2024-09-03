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
import multiprocessing as mp
import redis
import json

class HardwareTriggerTransmitter:
    def __init__(self, shared_queue, camera_ip, transmitter_config, transmitter_id, transmitterlog):
        self.queue = shared_queue
        self.camera_ip = camera_ip
        self.transmitter_config = transmitter_config
        self.cam = None
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
        self.redis_queue = mp.Queue()
        self.redis_client = None
        self.frame_count = 0
        self.transmitter_id = transmitter_id
        self.init_redis()
        self.configId = None

    def featurepathload(self):
        if self.transmitter_config["feature_path"]:
            one = os.getenv("CAMERA_CONFIGS_DIR")
            # two = os.path.join(one, "transmitterfeature")
            featureFilePath = os.path.join(one,self.transmitter_config["feature_path"])
            self.logger.info(f"what is the feature path {featureFilePath} {os.path.exists(featureFilePath)}")
            if len(featureFilePath) > 0:
                ret = self.cam.MV_CC_FeatureLoad(featureFilePath)
        return ret
    

    def clean_up_process(self):
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
        ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
        if self.deviceList.nDeviceNum == 0:
            configure_flag = False
            return configure_flag
        for i in range(self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                nip1 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000) >> 24
                nip2 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000) >> 16
                nip3 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00) >> 8
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF)
                composed_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                self.logger.info(f"Found camera--{composed_ip}")
                if composed_ip == self.camera_ip:
                    self.nConnectionNum = i

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
            if ret != 0:
                print(f"Open device fail! ret[0x%x]" % ret)
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
                    configure_flag = False
                    return configure_flag
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            if ret != 0:
                self.logger.info("Set trigger mode fail! ret[0x%x]" % ret)
                configure_flag = False
                return configure_flag
            ret = self.cam.MV_CC_SetEnumValueByString("TriggerSource", self.transmitter_config["trigger_line"])
            if ret != 0:
                self.logger.info("Could not set trigger source")
                configure_flag = False
                return configure_flag
            ret = self.cam.MV_CC_SetEnumValueByString("TriggerActivation", "RisingEdge")
            if ret != 0:
                self.logger.info("Could not set trigger activation")
                configure_flag = False
                return configure_flag
#---------###########--------------------------------------------------------
            # ret = self.cam.MV_CC_SetIntValue("LineDebouncerTime", int(float(self.transmitter_config["trigger_debounce"]) * 1000000)) #
            # if ret != 0:
            #     print("MV_CC_SetEnumValue    LineDebouncerTime ")
            #     # self.logger.info("Set trigger debounce fail! ret[0x%x]" % ret)
            #     configure_flag = False
            #     return configure_flag

            # ret = self.cam.MV_CC_SetFloatValue("TriggerDelay", float(self.transmitter_config["trigger_delay"]) * 1000000) #
            # if ret != 0:
            #     print("TriggerDelay")
            #     self.logger.info("Set trigger delay fail! ret[0x%x]" % ret)
            #     configure_flag = False
            #     return configure_flag

            # ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False) #
            # if ret != 0:
            #     print("AcquisitionFrameRateEnable")
            #     self.logger.info("Set acquisition frame rate enable fail! ret[0x%x]" % ret)
            #     
            #     configure_flag = False
            #     return configure_flag

#------------------------####------------------------------------------------------
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                self.logger.info("Get payload size fail! ret[0x%x]" % ret)                
                configure_flag = False
                return configure_flag
            self.nPayloadSize = stParam.nCurValue
            if self.featureFilePath_inConfig:
                ret = self.featurepathload()
                if ret != 0:
                    self.logger.info("load feature fail! ret[0x%x]" % ret)
                    configure_flag = False
                    return configure_flag

            elif self.transmitter_config["streaming_config"] != "null":
                config = self.transmitter_config["streaming_config"]["properties"]
                self.configure(config)
        return configure_flag

    def process_frame(self, configId):
        self.frame_count += 1
        numpy_array = np.array(self.pData, dtype=np.uint8)
        image = numpy_array.reshape(self.stFrameInfo.nHeight, self.stFrameInfo.nWidth)
        id = str(uuid.uuid4())
        frame_info = {
            "id": id,
            "beltId": self.transmitter_config["camera_id"],
            "camera_ip": self.camera_ip,
            "configId" : configId,
            "image": image,
            "frame_count": self.frame_count,
            "iterator" : 0,
            "groupId": 122,
            "groupLimit": 1,
            "extraInfo": ""
        }
        self.queue.put(frame_info)
        self.logger.info("Captured Frame")
    
        # save_in = os.getenv("SAVE_DIR")
        # # Save frame to disk
        # # filename = os.path.join(save_in, f"{self.frame_count}_{id}_transm.jpg")
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
                'Gamma': self.cam.MV_CC_SetFloatValue,
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

    def init_redis(self):
        self.redis_client = redis.StrictRedis(host=os.getenv('REDIS_ENDPOINT'), port=os.getenv('REDIS_PORT'))
        self.redis_subscriber_thread = threading.Thread(target=self.redis_subscribe)
        self.redis_subscriber_thread.start()

    def redis_subscribe(self):
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.transmitter_id) 
        for message in pubsub.listen():
            
            if not isinstance(message['data'], int):
                self.redis_queue.put(message['data'].decode('utf-8'))
                self.logger.info("the message", message)
                

    def update_config(self):
        configure_flag = True
        if not self.redis_queue.empty():    
            ret = self.cam.MV_CC_StopGrabbing()
            self.logger.info(f"updating with message")
            if ret != 0:
                self.logger.info("Start grabbing fail! ret[0x%x]" % ret)
                print("Stop grabbing fail! ret[0x%x]" % ret)
                configure_flag = False
                return configure_flag
            message = self.redis_queue.get()
            currCapture = json.loads(message)
            cameraConfig = currCapture["config"]        #[iterator]
            self.configId = currCapture["config"]["configId"]
            self.logger.info(f"what is configid {self.configId}")
            self.logger.info(f"what is config of the pub{cameraConfig}")
            self.configure(cameraConfig)
            ret = self.cam.MV_CC_StartGrabbing()
            self.logger.info(f"started grabbing ")
            if ret != 0:
                self.logger.info("Start grabbing fail! ret[0x%x]" % ret)
                configure_flag = False
                return configure_flag
            
        else:
            configure_flag = False

    def run(self, stop_event):
        while True:
            if not stop_event.is_set():
                configure_flag = self.configure_camera()
                if not configure_flag:
                    continue
                print("camera configuration completed")
                self.pData = (c_ubyte * self.nPayloadSize)()
                self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
                memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))

                ret = self.cam.MV_CC_StartGrabbing()
                if ret != 0:
                    self.logger.info("Start grabbing fail! ret[0x%x]" % ret)
                    self.g_bExit = True
                    continue

                while not stop_event.is_set():
                    if not self.redis_queue.empty():
                        self.update_config()
                    ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.pData), self.nPayloadSize, self.stFrameInfo, 1000)
                    if ret != 0:
                        self.logger.info(f"failed to capture {ret}")
                    if ret == 0:
                        self.process_frame(self.configId)
                    else:
                        self.logger.info("No_data[0x%x]" % ret)
                        time.sleep(0.1)
            else:
                break

        self.stop()



    def stop(self):
        self.g_bExit = True
        if self.isProcessCreated and self.processHandle and self.processHandle.is_alive():
            self.processHandle.join()
            self.nConnectionNum = -1
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
