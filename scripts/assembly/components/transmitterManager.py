import multiprocessing
from assembly.interfaces.transmitterInterface import TransmitterInterface
import logging
import os
from queue import Queue
import time
class TransmitterManager:
    def __init__(self, config, transmitterlog):
        self.config = config  # Configuration for transmitters
        self.processes = {}  # Dictionary to keep track of subprocesses
        self.interfaces = {}  # Dictionary to keep track of transmitter interfaces and stop events
        self.shared_queue = multiprocessing.Queue()  # Shared queue for inter-process communication
        self.logger = transmitterlog
        

    def start_transmitters(self):
        """Start all transmitters based on the provided configuration."""
        
        for transmitter_id, transmitter_config in self.config["transmitterInfo"].items():
            t = time.time()
            stop_event = multiprocessing.Event()  # Create a stop event for each transmitter
            
            interface = TransmitterInterface(self.shared_queue, transmitter_config["camera_ip"], transmitter_config, stop_event, transmitter_id, self.logger)
            self.interfaces[transmitter_id] = {"interfaceTrans": interface, "stop_event": stop_event}
            process = multiprocessing.Process(target=self.run_transmitter, args=(interface,))
            self.processes[transmitter_id] = process
            
            self.logger.info(f"Starting process for transmitter ID: {transmitter_id}")
            process.start()
            x = time.time() - t
            self.logger.info(f"time taken to start the process---> {x}")

    def run_transmitter(self, interface):
        """Run the transmitter interface."""
        self.logger.info(f"going to the run of manager")
        interface.run()

    def stop_transmitters(self):
        """Stop all running transmitters."""

        for transmitter_id, process in self.processes.items():
            t = time.time()
            self.logger.info(f"Reached stop transmitter in manager {transmitter_id}")
            if transmitter_id in self.interfaces:
                


                self.logger.info(f"Stopping transmitter for ID: {transmitter_id}")
                hh = time.time()
                
                self.interfaces[transmitter_id]["stop_event"].set()  # Signal the transmitter to stop
                set_time = time.time() - hh
                self.logger.info(f"Time taken to set event stop , {set_time}")
                xx = time.time()
                self.interfaces[transmitter_id]["interfaceTrans"].stop()
                stop_time = time.time() - xx
                self.logger.info(f"time taken stop {stop_time}")

            a = time.time()  
            if process.is_alive():
                self.logger.info(f"process is alive")
                process.join()  # Ensure the process has terminated
            b = time.time() - a
            self.logger.info(f"time taken for process to join , {b}")


            x = time.time() - t
            self.logger.info(f"Time taken to stop: {x} for the tranmitter :: {transmitter_id}")
        









































# import multiprocessing
# # from transmitter_interface import TransmitterInterface
# import json
# from ..interfaces.transmitterInterface import TransmitterInterface
# class TransmitterManager:
#     def __init__(self, config):
#         self.config = config
#         # self.shared_queues = {}
#         self.processes = {}
#         self.shared_queue = multiprocessing.Queue()
#         self.work = None
#     def start_transmitters(self):
#         for transmitter_id, transmitter_config in self.config["transmitterInfo"].items():
#             # Create a separate queue for each transmitter
#     # ---------------        # self.shared_queues[transmitter_id] = self.shared_queue-----------------
#         # The shared queue is stored in the self.shared_queues dictionary with the transmitter_id as the key.
#         # This allows for easy access to the queue associated with each transmitter.
#             process = multiprocessing.Process(target=self.run_transmitter, args=(self.shared_queue, transmitter_config))
#         # A new multiprocessing.Process is created.
#         # target=self.run_transmitter specifies that the run_transmitter method will be executed in the new process.
#         # args=(shared_queue, transmitter_config) passes the shared queue and transmitter configuration to the run_transmitter method when the process starts.
#             self.processes[transmitter_id] = process
#         # The process is stored in the self.processes dictionary with the transmitter_id as the key.
#         # This allows for management and tracking of the process for each transmitter.
#             process.start()
#         # This will invoke the run_transmitter method in a 
#         # separate process, allowing the transmitter to operate 
#         # concurrently with other processes.

        
#     # for each transmitter id 
#     def run_transmitter(self, shared_queue, transmitter_config):
#         self.work = TransmitterInterface(shared_queue, transmitter_config["camera_Ip"], transmitter_config)

#         frame_info = self.work.run()
#         print(frame_info)

#     def stop_transmitters(self):
#         for transmitter_id, process in self.processes.items():
#             #stop the transmitter process 
#             self.work.stop()

#             if process.is_alive():
#                 process.terminate()
#                 process.join()

# if __name__ == "__main__":
#     with open("config.json", "r") as f:
#         config = json.load(f)
#     manager = TransmitterManager(config)
#     manager.start_transmitters()



"""{
    "transmitterInfo": {
        "transmitterUUID_1": {
            "type": "trigger",
            "camera_type": "MVS",
            "trigger_type": "software",
            "camera_ip": "192.168.1.1",
            "camera_id": "camera_1",
            "group_id": "group_1",
            "extra_info": {"location": "Factory A"}
        },
        "transmitterUUID_2": {
            "type": "streaming",
            "camera_type": "CCTV",
            "camera_ip": "192.168.1.2",
            "camera_id": "camera_2",
            "group_id": "group_2",
            "extra_info": {"location": "Factory B"}
        }
    }
}"""