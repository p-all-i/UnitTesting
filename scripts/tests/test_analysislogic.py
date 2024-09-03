import pytest
from unittest.mock import Mock, MagicMock
import datetime
from assembly.analysis.analysis_logic import AnalysisLogic


@pytest.fixture
def mock_logger():
    """Fixture to create a mock logger object."""
    logger = Mock()
    logger.loop_logger.info = Mock()
    logger.loop_logger.error = Mock()
    return logger

@pytest.fixture
def mock_gp():
    """Fixture to create a mock GP object with interface objects."""
    mock_gp = Mock()
    # Create a mock interface object with a run method
    mock_interface = Mock()
    mock_interface.run = Mock(return_value=({"roi": "some_roi", "direction": "some_direction"}, 5))
    # Assume camera_id is 1 and iterator is 0 for testing
    mock_gp.interfaceObjs = {1: {0: mock_interface}}
    return mock_gp

@pytest.fixture
def analysis_logic_instance(mock_logger):
    """Fixture to create an instance of AnalysisLogic with a mock logger."""
    return AnalysisLogic(loggerObj=mock_logger)

def test_analysis_success(analysis_logic_instance, mock_gp):
    """Test the __call__ method for a successful image analysis."""
    # Mock input data
    input_data = {
        "image": MagicMock(),  # Mocked image data
        "beltId": 1,
        "groupId": 123,
        "iterator": 0,
        "configId": "config123",
        "groupLimit": 5,
        "extraInfo": "extra_info"
    }
    camera_id = 1

    # Call the AnalysisLogic instance
    result, object_count = analysis_logic_instance(mock_gp, input_data, camera_id)

    # Assertions
    assert result["roi"] == "some_roi"
    assert result["direction"] == "some_direction"
    assert result["cameraId"] == input_data["beltId"]
    assert result["groupId"] == input_data["groupId"]
    assert result["iterator"] == input_data["iterator"]
    assert result["configId"] == input_data["configId"]
    assert result["groupLimit"] == input_data["groupLimit"]
    assert result["extraInfo"] == input_data["extraInfo"]
    assert object_count == 5
    mock_gp.interfaceObjs[camera_id][int(input_data["iterator"])].run.assert_called_once_with(image=input_data["image"])
    analysis_logic_instance.loggerObj.loop_logger.info.assert_called_with("[INFO] Image Analysis Done again!")

def test_analysis_exception_handling(analysis_logic_instance, mock_gp, mock_logger):
    """Test the __call__ method for handling exceptions during image analysis."""
    # Mock input data
    input_data = {
        "image": MagicMock(),  # Mocked image data
        "beltId": 1,
        "groupId": 123,
        "iterator": 0,
        "configId": "config123",
        "groupLimit": 5,
        "extraInfo": "extra_info"
    }
    camera_id = 1

    # Make the interface run method raise an exception
    mock_gp.interfaceObjs[camera_id][int(input_data["iterator"])].run.side_effect = Exception("Test Exception")

    # Call the AnalysisLogic instance and check for exception handling
    with pytest.raises(SystemExit):  # Since exit(1) is called in case of an exception
        analysis_logic_instance(mock_gp, input_data, camera_id)

    # Assertions for error handling
    mock_logger.loop_logger.error.assert_any_call("[INFO] Error in Image Analysis again! Test Exception")
    assert mock_logger.loop_logger.error.call_count >= 3  # Ensure multiple error logs are made

if __name__ == "__main__":
    pytest.main()
