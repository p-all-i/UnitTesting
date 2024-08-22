import logging, os, datetime, sys, cv2, pika, base64, time, json
from dotenv import load_dotenv
import numpy as np
import imagezmq
from queue import Queue
from threading import Thread
import functools
import queue
import threading
import requests
import cv2  # Assuming OpenCV is used for image processing
from requests.exceptions import RequestException
import io
from PIL import Image
import numpy as np
print=functools.partial(print, flush=True)

class NodeCommServer:
    """
    A class responsible for handling message queue communication between nodes.
    
    Attributes:
    -----------
    exchange_subscribe : str
        Name of the exchange to subscribe to
    exchange_publish : str
        Name of the exchange to publish messages
    consuming_queue : str
        Name of the queue to consume messages from
    publishing_queue : str
        Name of the queue to publish messages to
    loggerObj : object
        Logger object to record logs
    
    Methods:
    --------
    start():
        Initializes the server, sets up connections, and declares exchanges and queues.
    read():
        Reads messages from the consuming queue.
    send(message):
        Sends a message to the publishing queue.
    """
    
    def __init__(self, exchange_publish_name, publishing_queue, consuming_queue, host, loggerObj):
        self.exchange_publish = exchange_publish_name
        self.publishing_queue = publishing_queue
        self.consuming_queue = consuming_queue
        self.loggerObj = loggerObj
        self.host = host
        self.channel = None
        self.connection = None
        self.loggerObj.logger.info("NodeCommServer initialized.")

    def start(self):
        """
        Initializes the server, sets up connections, and declares exchanges and queues.
        """
        try:
            creds = pika.PlainCredentials('guest', 'guest')
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, credentials=creds, heartbeat=15, retry_delay=1, connection_attempts=10)
            )
            self.channel = self.connection.channel()

            self.channel.exchange_declare(exchange=self.exchange_publish, exchange_type='direct')
            self.channel.queue_declare(queue=self.publishing_queue, durable=False, arguments={'x-message-ttl': 30000})
            self.channel.queue_bind(exchange=self.exchange_publish, queue=self.publishing_queue, routing_key=self.publishing_queue)
            self.channel.queue_declare(queue=self.consuming_queue, durable=False, arguments={'x-message-ttl': 30000})

            print(f"[INFO] {datetime.datetime.now()} Connected with RabbitMQ Qs!!!")
            self.loggerObj.logger.info(f"Connected with RabbitMQ Qs!")
        except Exception as e:
            self.loggerObj.logger.error(f"Error in NodeCommServer start: {str(e)}")
            print(f"[ERROR] {datetime.datetime.now()} Error in NodeCommServer start: {str(e)}")
            self.channel = None

    def read(self):
        """
        Reads a message from the consuming queue.
        
        Returns:
        --------
        dict or None
            Returns the message as a JSON object if available, otherwise returns None.
        """
        if not self.channel:
            self.loggerObj.logger.error("Channel is not initialized.")
            print(f"[ERROR] {datetime.datetime.now()} Channel is not initialized.")
            return None
        try:
            method_frame, header_frame, body = self.channel.basic_get(queue=self.consuming_queue)
            if method_frame:
                print("routing_key: ", method_frame.routing_key)
                self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                print(f"[INFO] {datetime.datetime.now()} Read a file from Queue for {self.consuming_queue} with key {method_frame.routing_key}!!!")
                self.loggerObj.queuing_logger.info(f"Read a file from Queue for {self.consuming_queue} with key {method_frame.routing_key}!!!")
                self.loggerObj.loop_logger.info(f"Read a file from Queue for {self.consuming_queue} with key {method_frame.routing_key}!!!")
                message = json.loads(body)
                message["camera_id"] = method_frame.routing_key
                # if 'variant_change' in message:
                #     message["type"] = "variant_change"
                # else:
                #     message["type"] = "data"
                return message
            else:
                return None
            
        except Exception as e:
            self.loggerObj.logger.exception(f"Error in NodeCommServer read: {e}")
            print(f"[ERROR] {datetime.datetime.now()} Error in NodeCommServer read: {e}")
            self.channel = None
        return None

    def send(self, message):
        """
        Sends a message to the publishing queue.
        
        Parameters:
        -----------
        message : dict
            The message to be sent.
        """
        if not self.channel:
            self.loggerObj.logger.error("Channel is not initialized.")
            print(f"[ERROR] {datetime.datetime.now()} Channel is not initialized.")
            return
        
        message_json = json.dumps(message)
        while True:
            try:
                self.channel.basic_publish(exchange=self.exchange_publish, routing_key=self.publishing_queue, body=message_json)
                break
            except Exception as e:
                self.loggerObj.logger.error(f"Error in send method: {str(e)}")
                print(f"[ERROR] {datetime.datetime.now()} Error in send method: {str(e)}")
                self.start()
        print(f"[INFO] {datetime.datetime.now()} Publishing result for {message['cameraId']} to the result Queue")
        self.loggerObj.queuing_logger.info(f"Publishing result for {message['cameraId']} to the result Queue")
        self.loggerObj.loop_logger.info(f"Publishing result for {message['cameraId']} to the result Queue")

class ImageQServer:
    def __init__(self, host, tcp_port, loggerObj, queueSize=128, rgb=True):
        """
        Initialize the ImageQServer object.

        Args:
            host (str): Host address of the server.
            tcp_port (int): TCP port for the server to listen on.
            loggerObj (Logger): Logger object for logging information.
            queueSize (int, optional): Maximum size of the queue. Defaults to 128.

        Attributes:
            stopped (bool): Flag to control the running of the server.
            image_hub (imagezmq.ImageHub): Object to receive images over ZMQ.
            queue (Queue): Queue to store received images.
            host (str): Host address.
            tcp_port (int): TCP port.
            queueSize (int): Maximum size of the queue.
            loggerObj (Logger): Logger object.
        """
        self.stopped = False
        self.image_hub = None
        self.queue = Queue(maxsize=queueSize)
        self.host = host
        self.tcp_port = tcp_port
        self.queueSize = queueSize
        self.loggerObj = loggerObj
        self.rgb = rgb

    def start(self):
        """
        Start the image server and begin receiving images.

        This method initializes the ImageHub object and starts a separate
        thread that runs the update method to continuously receive images.

        Returns:
            ImageQServer: Returns the instance of the class for chaining.
        """
        self.image_hub = imagezmq.ImageHub(
            open_port=f"tcp://{self.host}:{self.tcp_port}")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """
        Continuously receive images and add them to the queue.

        This method runs in a separate thread started by the start method.
        It receives images from the image hub, logs information about them,
        and adds them to the queue. It also sends a reply back after each
        reception of an image.
        """
        while True:  # show streamed images until Ctrl-C
            if self.stopped:
                break
            st1 = time.time()
            print("Starting time Getting image: ", st1)
            self.loggerObj.loop_logger.info(f"Starting time Getting image: {st1}")
            rpi_name, image = self.image_hub.recv_image()
            print("Time for getting the frames: ", time.time()-st1, time.time())
            self.loggerObj.loop_logger.info(f"Time for getting the frames: {time.time()-st1} , {time.time()}")

            # Converting image to 3 channel
            if self.rgb:
                if len(list(image.shape))==2:
                    image = cv2.merge([image, image, image])
            
            camera_id, group_id, iterator, group_limit, extra_info = rpi_name.split("|")

            print("Received Extrainfo ", extra_info)
            self.loggerObj.loop_logger.info(f"Received Extrainfo {extra_info}")

            self.loggerObj.queuing_logger.info(f"Putting frame in queue for ----- {rpi_name} image shape:{image.shape}")
            print(f"Putting frame in queue for ----- {rpi_name} image shape:{image.shape}")
            if self.queue.qsize() == self.queueSize:
                self.loggerObj.queuing_logger.info(f"Queue_reached_limit")
            st2 = time.time()
            self.queue.put({
                "cameraId": camera_id,
                "groupId": group_id,
                "iterator": iterator,
                "groupLimit": group_limit,
                "extraInfo": extra_info, 
                "image": image
            })
            print("Time for Putting image to python Q: ", time.time()-st2, time.time())
            self.loggerObj.loop_logger.info(f"Time for Putting image to python Q: {time.time()-st2} , {time.time()}")
            
            self.image_hub.send_reply(b'OK')

    def read(self):
        """
        Retrieve and return the next image from the queue.

        This method returns the next image in the queue. If the queue is empty,
        it will return an empty list.

        Returns:
            list: A list containing the next image from the queue, if available.
        """
        ct = 1
        image_res = []
        while (ct > 0 and self.queue.qsize() > 0):
            if self.stopped:
                break
            image_res.append(self.queue.get())
            ct -= 1
        return image_res

    def stop(self):
        """
        Stop the server.

        This method sets the stopped flag to True, which stops the image
        receiving loop in the update method. It also clears the queue.
        """
        self.stopped = True
        self.queue = None





class ImageSender:
    def __init__(self, post_image_endpoint, loggerObj):
        self.post_image_endpoint = post_image_endpoint
        self.loggerObj = loggerObj
        self.frame_queue = queue.Queue()  # Initialize a thread-safe queue
        print("[INFO] Started ImageSender")
        self.loggerObj.logger.info("ImageSender initialized with endpoint: {}".format(post_image_endpoint))


    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        self.loggerObj.logger.info("ImageSender update thread started")
        return self



    def add_to_end(self, image, belt_id, config_id):
        # Simply enqueue the image and belt_id as a tuple
        self.frame_queue.put((image, belt_id, config_id))
        self.loggerObj.logger.info(f"Added image to queue with belt_id: {belt_id}, config_id: {config_id}")



    
    def update(self):
        next_print_time = time.time() + 5  # Set the initial next print time
        self.loggerObj.logger.info("Update loop started")
        while True:
            if time.time() >= next_print_time:
                queue_size = self.frame_queue.qsize()
                self.loggerObj.logger.info(f"Image sender Queue size: {queue_size}")
                print(f"Image sender Queue size: {queue_size}")  # Print queue size
                next_print_time = time.time() + 5  # Update the next print time

            if self.frame_queue.empty():  # Check if the queue is empty
                time.sleep(0.001)
                continue  # Skip processing if the queue is empty

            frame, belt_id,config_id = self.frame_queue.get()  # Dequeue an item from the queue
            # rpi_name = f"{belt_id}|{self.server_id}"
            self.loggerObj.logger.info(f"Processing frame from queue with belt_id: {belt_id}, config_id: {config_id}")

            try:
                # Convert the frame to a format suitable for sending via HTTP
                # Convert the frame to a PIL Image and then to base64
                image = Image.fromarray(frame)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Prepare data for POST request
                data = {
                    'image': img_str,
                    'configId': config_id
                }
                self.loggerObj.logger.info(f"Sending POST request to {self.post_image_endpoint} with belt_id: {belt_id} and config_id: {config_id}")

                # Optionally, set headers if the server requires a specific Content-Type
                headers = {'Content-Type': 'image/jpeg'}

                response = requests.post(self.post_image_endpoint, json=data, headers=headers)
                response.raise_for_status()  # Raise an error for bad status codes
                self.loggerObj.logger.info(f"POST request successful with status code: {response.status_code}")

            except RequestException as e:
                self.loggerObj.error(f"Error happened while sending frames: {e}")
                print('Error sending image:', e)





