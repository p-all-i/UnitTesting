import pytest
from unittest.mock import Mock, patch
import multiprocessing
from assembly.components.transmitterManager import TransmitterManager
from assembly.interfaces.transmitterInterface import TransmitterInterface
@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def mock_config():
    return {
        "transmitterInfo": {
            "transmitter1": {
                "camera_ip": "192.168.0.1",
                "some_other_config": "value1"
            },
            "transmitter2": {
                "camera_ip": "192.168.0.2",
                "some_other_config": "value2"
            }
        }
    }

@pytest.fixture
def transmitter_manager(mock_config, mock_logger):
    return TransmitterManager(config=mock_config, transmitterlog=mock_logger)

def test_start_transmitters(monkeypatch, transmitter_manager, mock_logger):
    # Mocking TransmitterInterface and multiprocessing.Process
    mock_interface = Mock()
    mock_process = Mock()

    # Mock the TransmitterInterface constructor
    with patch('assembly.interfaces.transmitterInterface', return_value=mock_interface):
        # Mock the multiprocessing.Process
        # with patch('assembly.components.transmitterManager.TransmitterManager.multiprocessing.Process', return_value=mock_process):
        with patch('multiprocessing.Process', return_value=mock_process):
            transmitter_manager.start_transmitters()

            assert len(transmitter_manager.processes) == 2  # We have two transmitters in the config
            assert len(transmitter_manager.interfaces) == 2

            # Ensure the logger was called with the correct messages
            mock_logger.info.assert_any_call("Starting process for transmitter ID: transmitter1")
            mock_logger.info.assert_any_call("Starting process for transmitter ID: transmitter2")

            # Check that the process start method was called
            assert mock_process.start.call_count == 2

def test_run_transmitter(monkeypatch, transmitter_manager, mock_logger):
    # Mock the interface
    mock_interface = Mock()

    transmitter_manager.run_transmitter(mock_interface)

    # Ensure the logger was called
    mock_logger.info.assert_called_once_with("going to the run of manager")

    # Ensure the interface's run method was called
    mock_interface.run.assert_called_once()

def test_stop_transmitters(monkeypatch, transmitter_manager, mock_logger):
    # Mocking the transmitter processes and interfaces
    mock_interface1 = Mock()
    mock_interface2 = Mock()
    mock_process1 = Mock()
    mock_process2 = Mock()

    transmitter_manager.interfaces = {
        "transmitter1": {"interfaceTrans": mock_interface1, "stop_event": Mock()},
        "transmitter2": {"interfaceTrans": mock_interface2, "stop_event": Mock()},
    }
    transmitter_manager.processes = {
        "transmitter1": mock_process1,
        "transmitter2": mock_process2,
    }

    mock_process1.is_alive.return_value = True
    mock_process2.is_alive.return_value = True

    transmitter_manager.stop_transmitters()

    # Check that the stop_event was set and interface stop was called exactly once
    for transmitter_id in transmitter_manager.interfaces:
        stop_event = transmitter_manager.interfaces[transmitter_id]["stop_event"]
        interface_trans = transmitter_manager.interfaces[transmitter_id]["interfaceTrans"]

        stop_event.set.assert_called_once()
        interface_trans.stop.assert_called_once()

    # Ensure the processes were joined
    assert mock_process1.join.call_count == 1
    assert mock_process2.join.call_count == 1

if __name__ == "__main__":
    pytest.main()
