import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from collections import OrderedDict
import cv2
from assembly.components.mainProcessor import MainProcessor

@pytest.fixture
def setup_processor():
    """
    Fixture to set up MainProcessor with mocked dependencies.
    """
    # Mock the dependencies
    GP_mock = MagicMock()
    loggerObj_mock = MagicMock()
    analysisLogic_mock = MagicMock()
    output_sender_mock = MagicMock()
    outputprepObj_mock = MagicMock()
    visualisor_mock = MagicMock()
    save_dir_mock = "mock_save_dir"
    shared_queue_mock = MagicMock()
    transmitter_manager_mock = MagicMock()
    varientchangeServer_mock = MagicMock()

    # Properly configure the GP_mock if needed
    GP_mock.interfaceObjs = {
        "camera_1": [MagicMock()]
    }

    # Initialize the MainProcessor with mocks
    processor = MainProcessor(
        GP=GP_mock,
        loggerObj=loggerObj_mock,
        analysisLogic=analysisLogic_mock,
        output_sender=output_sender_mock,
        outputprepObj=outputprepObj_mock,
        visualisor=visualisor_mock,
        save_dir=save_dir_mock,
        shared_queue=shared_queue_mock,
        transmitter_manager=transmitter_manager_mock,
        varientchangeServer=varientchangeServer_mock
    )

    return processor, {
        'GP_mock': GP_mock,
        'loggerObj_mock': loggerObj_mock,
        'analysisLogic_mock': analysisLogic_mock,
        'output_sender_mock': output_sender_mock,
        'outputprepObj_mock': outputprepObj_mock,
        'visualisor_mock': visualisor_mock,
        'shared_queue_mock': shared_queue_mock,
        'transmitter_manager_mock': transmitter_manager_mock,
        'varientchangeServer_mock': varientchangeServer_mock
    }



def test_extract_frame_from_queue(setup_processor):
    processor, mocks = setup_processor

    # Mock the shared queue to return a frame
    mocks['shared_queue_mock'].empty.return_value = False
    mock_image = np.zeros((100, 100), dtype=np.uint8)
    frame_info = {
        "beltId": "camera_1",
        "image": mock_image
    }
    mocks['shared_queue_mock'].get.return_value = frame_info

    # Call the method
    img_master = processor.extract_frame_from_queue()

    # Verify the result
    assert "camera_1" in img_master
    assert isinstance(img_master["camera_1"], dict)
    assert np.array_equal(img_master["camera_1"]["image"], cv2.merge([mock_image, mock_image, mock_image]))



def test_analyse(setup_processor):
    processor, mocks = setup_processor

    # Mock analysis logic
    mocks['analysisLogic_mock'].return_value = ({"result": "some_result"}, 5)

    input_data = {
        "image": np.zeros((100, 100, 3), dtype=np.uint8)
    }

    # Call the method
    output_res, object_count = processor.analyse(input_data=input_data, camera_id="camera_1")

    # Verify the results
    assert output_res == {"result": "some_result"}
    assert object_count == 5
    mocks['analysisLogic_mock'].assert_called_once_with(GP=mocks['GP_mock'], input_data=input_data, camera_id="camera_1")


def test_run(setup_processor):
    """
    Test the run method of MainProcessor.
    """
    processor, mocks = setup_processor

    # Mock shared_queue to return frames
    mocks['shared_queue_mock'].empty.side_effect = [False, True]  # Process once then stop
    mock_image = np.zeros((100, 100), dtype=np.uint8)
    frame_info = {
        "beltId": "camera_1",
        "image": mock_image,
        "frame_count": 1,
        "groupId": "group_1",
        "iterator": 0
    }
    mocks['shared_queue_mock'].get.return_value = frame_info

    # Mock analysis logic to return some fake results
    mocks['analysisLogic_mock'].return_value = ({"roi": [(0, 0, 50, 50)], "direction": "horizontal", "result": "some_result"}, 1)

    # Mock output preparation
    mocks['outputprepObj_mock'].run.return_value = {"final_result": "some_output"}
    mocks['outputprepObj_mock'].final_prep.return_value = {"final_result": "some_output"}

    # Mock visualisation to return an image
    mocks['visualisor_mock'].draw.return_value = mock_image

    # Mock variant change handling to prevent test interference
    mocks['varientchangeServer_mock'].read.return_value = None

    # Run the processor for a short duration to test
    with patch('time.sleep', return_value=None):  # Speed up the loop
        with patch.object(processor, 'stop', side_effect=processor.stop) as mock_stop:
            processor.run()

    # Verify calls and outputs
    assert mock_stop.called
    # mocks['loggerObj_mock'].loop_logger.info.assert_any_call('Analysis Started!!!')  # Check log message
    mocks['output_sender_mock'].send.assert_called_once()  # Ensure sending function was called
    mocks['visualisor_mock'].draw.assert_called_once()  # Ensure visualisation function was called

# Additional test functions can be added here for other methods
