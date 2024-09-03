import pytest
from unittest.mock import Mock, patch
from assembly.interfaces.transmitterInterface import TransmitterInterface


@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def mock_shared_queue():
    return Mock()

@pytest.fixture
def mock_transmitter_config_streaming():
    return {
        'camera_type': 'streaming',
        'camera_ip': '192.168.0.1',
        'some_other_config': 'value1'
    }

@pytest.fixture
def mock_transmitter_config_software():
    return {
        'camera_type': 'Software Trigger',
        'camera_ip': '192.168.0.1',
        'some_other_config': 'value1'
    }

@pytest.fixture
def mock_stop_event():
    return Mock()

@pytest.fixture
def transmitter_interface(mock_shared_queue, mock_transmitter_config_streaming, mock_stop_event, mock_logger):
    return TransmitterInterface(
        shared_queue=mock_shared_queue,
        camera_ip='192.168.0.1',
        transmitter_config=mock_transmitter_config_streaming,
        stop_event=mock_stop_event,
        transmitter_id='transmitter1',
        transmitterlog=mock_logger
    )

def test_initialization(transmitter_interface, mock_logger):
    assert transmitter_interface.transmitter is None
    assert transmitter_interface.camera_ip == '192.168.0.1'
    assert transmitter_interface.transmitter_id == 'transmitter1'
    mock_logger.info.assert_not_called()  # Ensure no log is called during init

@patch('assembly.interfaces.transmitterInterface.ContinuousStreamingTransmitter')
def test_create_transmitter_streaming(MockContinuousStreamingTransmitter, transmitter_interface, mock_logger):
    transmitter_interface._create_transmitter()
    mock_logger.info.assert_any_call("Working with camera type: streaming")
    assert transmitter_interface.transmitter is not None
    MockContinuousStreamingTransmitter.assert_called_once()

@patch('assembly.interfaces.transmitterInterface.SoftwareTriggerTransmitter')
def test_create_transmitter_software_trigger(MockSoftwareTriggerTransmitter, mock_shared_queue, mock_transmitter_config_software, mock_stop_event, mock_logger):
    transmitter_interface = TransmitterInterface(
        shared_queue=mock_shared_queue,
        camera_ip='192.168.0.1',
        transmitter_config=mock_transmitter_config_software,
        stop_event=mock_stop_event,
        transmitter_id='transmitter1',
        transmitterlog=mock_logger
    )
    transmitter_interface._create_transmitter()
    mock_logger.info.assert_any_call("Transmitter Initialization completed")
    assert transmitter_interface.transmitter is not None
    MockSoftwareTriggerTransmitter.assert_called_once()

def test_run(transmitter_interface):
    with patch.object(transmitter_interface, '_create_transmitter') as mock_create_transmitter:
        transmitter_interface.run()
        mock_create_transmitter.assert_called_once()

@patch('assembly.interfaces.transmitterInterface.ContinuousStreamingTransmitter')
def test_stop(MockContinuousStreamingTransmitter, transmitter_interface):
    # Assume that a transmitter is created
    transmitter_interface._create_transmitter()  # mock data (real)
    transmitter_interface.stop() # calling the function
    MockContinuousStreamingTransmitter.return_value.stop.assert_called_once() # expected output using mock of transmitter function 

def test_stop_no_transmitter(transmitter_interface, mock_logger):
    # Ensure no transmitter is created
    transmitter_interface.transmitter = None # mock data (real)
    transmitter_interface.stop() # calling d func

    # Check the logger for the correct message
    mock_logger.info.assert_called_once_with("Transmitter not created; nothing to stop.") # expected o/p asserted

if __name__ == "__main__":
    pytest.main()
