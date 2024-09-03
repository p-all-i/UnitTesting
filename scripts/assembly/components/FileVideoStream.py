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
            self.loggerObj.logger.info(f"Connected with RabbitMQ Qs! for Image sending")
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



class varientchange_server:
    def __init__(self, exchange_publish_name, publishing_queue, consuming_queue, host, loggerObj):
        self.exchange_publish = exchange_publish_name
        self.publishing_queue = publishing_queue
        self.consuming_queue = consuming_queue
        self.loggerObj = loggerObj
        self.host = host
        self.channel = None
        self.connection = None
        self.loggerObj.logger.info("varientchangeServer initialized.")


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

            print(f"[INFO] {datetime.datetime.now()} Connected with RabbitMQ Qs!!! for getting varient change")
            self.loggerObj.logger.info(f"Connected with RabbitMQ Qs!for getting varient change")
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
            # self.loggerObj.logger.error("Channel is not initialized.")
            # print(f"[ERROR] {datetime.datetime.now()} Channel is not initialized.")
            return None
        try:
            
            # self.loggerObj.logger.error("to the try of read of varient change")
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





