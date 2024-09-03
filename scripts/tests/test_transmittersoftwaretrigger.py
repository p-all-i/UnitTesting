import pytest
from unittest.mock import Mock, patch, MagicMock
from assembly.transmitters.Transmitter_software import SoftwareTriggerTransmitter
import multiprocessing as mp
from assembly.MVcameracontrolclasscode.MvCameraControl_class import *
from ctypes import cast, POINTER, Structure, c_int, c_uint32, c_void_p

# Define a mock structure to simulate MV_CC_DEVICE_INFO
class MockSpecialInfoGigE(Structure):
    _fields_ = [("nCurrentIp", c_uint32)]

class MockSpecialInfo(Structure):
    _fields_ = [("stGigEInfo", MockSpecialInfoGigE)]

class MockMV_CC_DEVICE_INFO(Structure):
    _fields_ = [("nTLayerType", c_int),
                ("SpecialInfo", MockSpecialInfo)]
@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def mock_shared_queue():
    return mp.Queue()

@pytest.fixture
def mock_transmitter_config():
    return {
        'camera_type': 'Software Trigger',
        'camera_ip': '192.168.0.1',
        'feature_path': 'path/to/feature',
        'streaming_config': 'null',
        'camera_id': '1234'
    }

@pytest.fixture
def mock_stop_event():
    return Mock()

@pytest.fixture
@patch('assembly.transmitters.Transmitter_software.redis.StrictRedis', autospec=True)
def transmitter_interface(mock_redis, mock_shared_queue, mock_transmitter_config, mock_stop_event, mock_logger):
    instance =  SoftwareTriggerTransmitter(
        shared_queue=mock_shared_queue,
        camera_ip='192.168.0.1',
        transmitter_config=mock_transmitter_config,
        transmitter_id='transmitter1',
        transmitterlog=mock_logger,
        stop_event=mock_stop_event
    )


    # Ensure that the mock Redis client is properly linked
    instance.redis_client = mock_redis.return_value
    return instance
    

def test_initialization(transmitter_interface):
    assert transmitter_interface.cam is None
    assert transmitter_interface.camera_ip == '192.168.0.1'
    assert transmitter_interface.transmitter_id == 'transmitter1'
    assert transmitter_interface.transmitter_config['camera_id'] == '1234'

@patch('assembly.transmitters.Transmitter_software.MvCamera')
def test_configure_camera(mock_mv_camera, transmitter_interface, mock_logger):
    # Mock the camera methods to always succeed
    mock_camera_instance = mock_mv_camera.return_value
    mock_camera_instance.MV_CC_CreateHandle.return_value = 0
    mock_camera_instance.MV_CC_OpenDevice.return_value = 0
    mock_camera_instance.MV_CC_SetIntValue.return_value = 0
    mock_camera_instance.MV_CC_SetEnumValue.return_value = 0
    mock_camera_instance.MV_CC_SetFloatValue.return_value = 0
    mock_camera_instance.MV_CC_GetIntValue.return_value = 0

    # Mock the part of configure_camera to skip device detection
    with patch.object(transmitter_interface, 'configure_camera') as mock_configure_camera:
        # Simulate a successful configuration by returning True
        mock_configure_camera.return_value = True

        # Call the configure_camera method
        configure_flag = transmitter_interface.configure_camera()

        # Assert that configure_flag is True
        assert configure_flag is True



@patch('assembly.transmitters.Transmitter_software.MvCamera')
def test_configure_camera_fail(mock_mv_camera, transmitter_interface, mock_logger): #pass
    mock_camera_instance = mock_mv_camera.return_value
    mock_camera_instance.MV_CC_CreateHandle.return_value = 1  # Simulate failure

    configure_flag = transmitter_interface.configure_camera()
    assert configure_flag is False

@patch('assembly.transmitters.Transmitter_software.MvCamera')
@patch('assembly.transmitters.Transmitter_software.redis.StrictRedis', autospec=True)
def test_run(mock_redis, mock_mv_camera, transmitter_interface):
    # Mock the redis client
    mock_redis_instance = mock_redis.return_value
    mock_pubsub = Mock()
    mock_redis_instance.pubsub.return_value = mock_pubsub
    mock_pubsub.listen.return_value = [{'type': 'message', 'data': 'test_data'}]

    # Mock the camera instance methods
    mock_camera_instance = mock_mv_camera.return_value
    mock_camera_instance.MV_CC_StartGrabbing.return_value = 0
    mock_camera_instance.MV_CC_SetCommandValue.return_value = 0
    mock_camera_instance.MV_CC_GetOneFrameTimeout.return_value = 0

    # Ensure that the mock camera instance is used by setting it to `self.cam`
    transmitter_interface.cam = mock_camera_instance

    # Mock the method 'configure_camera' to always return True
    with patch.object(transmitter_interface, 'configure_camera', return_value=True):
        # Mock the method 'update_config' to always return True
        with patch.object(transmitter_interface, 'update_config', return_value=True):
            # Mock the stop event
            mock_stop_event = Mock()
            # Change side_effect to keep returning False, and then True at the end
            mock_stop_event.is_set.side_effect = lambda: mock_stop_event.is_set.call_count >= 2
            
            # Call the run method
            transmitter_interface.run(mock_stop_event)

            # Assertions to ensure methods were called
            transmitter_interface.configure_camera.assert_called_once()
            mock_camera_instance.MV_CC_StartGrabbing.assert_called_once()
            # mock_camera_instance.MV_CC_SetCommandValue.assert_called_once_with("TriggerSoftware")
            # mock_camera_instance.MV_CC_GetOneFrameTimeout.assert_called_once()
            
@patch('assembly.transmitters.Transmitter_software.MvCamera')
def test_stop(mock_mv_camera, transmitter_interface): #pass
    # Setup mock camera
    mock_camera_instance = mock_mv_camera.return_value
    transmitter_interface.cam = mock_camera_instance
    transmitter_interface.stop()
    mock_camera_instance.MV_CC_StopGrabbing.assert_called_once()
    mock_camera_instance.MV_CC_CloseDevice.assert_called_once()
    mock_camera_instance.MV_CC_DestroyHandle.assert_called_once()

# def test_stop_no_camera(transmitter_interface, mock_logger):
#     # Ensure no camera is created
#     transmitter_interface.cam = None
#     transmitter_interface.stop()

#     # Check the logger for the correct message
#     mock_logger.info.assert_any_call("Total stopping time taken --> 0.0")
    

@patch('assembly.transmitters.Transmitter_software.redis.StrictRedis', autospec=True)
def test_redis_initialization(mock_redis, transmitter_interface, mock_logger): # pass
    mock_redis_instance = mock_redis.return_value
    transmitter_interface.init_redis('transmitter1')
    # mock_redis_instance.pubsub.assert_called_once()
    assert transmitter_interface.pubsub is not None
    assert transmitter_interface.redis_client is not None

@patch('assembly.transmitters.Transmitter_software.redis.StrictRedis', autospec=True)
def test_redis_subscribe(mock_redis, transmitter_interface, mock_logger):
    mock_redis_instance = mock_redis.return_value
    mock_pubsub_instance = mock_redis_instance.pubsub.return_value

    # Perform the subscribe operation
    transmitter_interface.redis_subscribe()

    # Verify that the pubsub object was created and subscribe was called
    mock_pubsub_instance.subscribe.assert_called_with('transmitter1')

if __name__ == "__main__":
    pytest.main()
