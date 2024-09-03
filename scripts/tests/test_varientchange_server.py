import unittest
from unittest.mock import MagicMock, patch, call
import pika
import json
from datetime import datetime
from assembly.components.FileVideoStream import varientchange_server
# Import the varientchange_server class from the appropriate module
# from your_module import varientchange_server

class TestVarientChangeServer(unittest.TestCase):
    
    def setUp(self):
        # Mock logger object
        self.mock_logger = MagicMock()
        self.mock_logger.logger = MagicMock()
        self.mock_logger.queuing_logger = MagicMock()
        self.mock_logger.loop_logger = MagicMock()
        
        # Initialize varientchange_server with mock objects
        self.server = varientchange_server(
            exchange_publish_name="test_exchange",
            publishing_queue="test_publish_queue",
            consuming_queue="test_consume_queue",
            host="localhost",
            loggerObj=self.mock_logger
        )
    
    @patch('pika.BlockingConnection')
    def test_start_success(self, mock_blocking_connection):
        # Mock connection and channel
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_blocking_connection.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        
        # Call start method
        self.server.start()
        
        # Verify connection and channel setup
        mock_blocking_connection.assert_called_once_with(pika.ConnectionParameters(
            host="localhost",
            credentials=pika.PlainCredentials('guest', 'guest'),
            heartbeat=15,
            retry_delay=1,
            connection_attempts=10
        ))
        mock_channel.exchange_declare.assert_called_once_with(exchange="test_exchange", exchange_type='direct')
        mock_channel.queue_declare.assert_any_call(queue="test_publish_queue", durable=False, arguments={'x-message-ttl': 30000})
        mock_channel.queue_declare.assert_any_call(queue="test_consume_queue", durable=False, arguments={'x-message-ttl': 30000})
        mock_channel.queue_bind.assert_called_once_with(exchange="test_exchange", queue="test_publish_queue", routing_key="test_publish_queue")
        self.assertIsNotNone(self.server.channel)
        self.mock_logger.logger.info.assert_called_with("Connected with RabbitMQ Qs!for getting varient change")

    @patch('pika.BlockingConnection')
    def test_start_failure(self, mock_blocking_connection):
        # Simulate an exception during connection
        mock_blocking_connection.side_effect = Exception("Connection failed")
        
        # Call start method
        self.server.start()
        
        # Verify the error handling
        self.assertIsNone(self.server.channel)
        self.mock_logger.logger.error.assert_called_with("Error in NodeCommServer start: Connection failed")

    def test_read_success(self):
        # Mock channel and message
        mock_channel = MagicMock()
        mock_method_frame = MagicMock()
        mock_method_frame.routing_key = "camera_1"
        mock_body = json.dumps({"key": "value"}).encode('utf-8')
        
        self.server.channel = mock_channel
        mock_channel.basic_get.return_value = (mock_method_frame, MagicMock(), mock_body)
        
        # Call read method
        result = self.server.read()
        
        # Verify the message processing
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["camera_id"], "camera_1")
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=mock_method_frame.delivery_tag)
        self.mock_logger.queuing_logger.info.assert_called_once_with(
            f"Read a file from Queue for test_consume_queue with key {mock_method_frame.routing_key}!!!"
        )
    
    def test_read_no_message(self):
        # Mock channel to return no message
        mock_channel = MagicMock()
        self.server.channel = mock_channel
        mock_channel.basic_get.return_value = (None, None, None)
        
        # Call read method
        result = self.server.read()
        
        # Verify no message is returned
        self.assertIsNone(result)
    
    def test_read_failure(self):
        # Mock channel and simulate an exception
        mock_channel = MagicMock()
        self.server.channel = mock_channel
        mock_channel.basic_get.side_effect = Exception("Read failed")
        
        # Call read method
        result = self.server.read()
        
        # Verify error handling
        self.assertIsNone(result)
        self.assertIsNone(self.server.channel)
        self.mock_logger.logger.exception.assert_called_once_with("Error in NodeCommServer read: Read failed")

if __name__ == '__main__':
    unittest.main()
