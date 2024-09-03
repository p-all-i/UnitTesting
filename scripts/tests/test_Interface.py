import pytest
from unittest.mock import MagicMock, patch

# Import the Interface class and dependencies to be mocked
from assembly.interfaces.interface import Interface

# Sample input data for testing
group_info_mock = {
    "steps": ["tracker", "cropping"],
    "tracker": {
        "model_id": "tracker_model_id",
        "roi": {
            "line": [(0, 0), (100, 100)],
            "direction": "horizontal"
        }
    },
    "cropping": {
        "roi_id_1": {
            "ground_truth": {},
            "model_1": [
                {
                    "model_id": "model_1_id",
                    "threshold": {
                        "score_thresh": {
                            "class_1": 0.7
                        }
                    },
                    "type": "detection"
                }
            ],
            "model_2": [],
            "roi": [10, 10, 50, 50]
        },
        "roi_id_2": {
            "ground_truth": {},
            "model_1": [
                {
                    "model_id": "model_2_id",
                    "threshold": {
                        "score_thresh": {
                            "class_2": 0.8
                        }
                    },
                    "type": "detection"
                }
            ],
            "model_2": [],
            "roi": [60, 60, 100, 100]
        }
    }
}

ModelDict_mock = {
    "tracker_model_id": MagicMock(),
    "model_1_id": MagicMock(),
    "model_2_id": MagicMock()
}

TrackerDict_mock = {
    "camera1": MagicMock()
}

@patch('assembly.interfaces.interface.TrackerInterface', autospec=True)
@patch('assembly.interfaces.interface.inferenceInterface', autospec=True)
@patch('assembly.interfaces.interface.AssemblyInterface', autospec=True)
def test_interface(mock_assembly, mock_inference, mock_tracker):
    # Mock the behaviors
    mock_tracker_instance = mock_tracker.return_value
    mock_tracker_instance.run.return_value = ({1: [199, 313, 760, 918]}, 1) # Mock the tracker output

    mock_inference_instance = mock_inference.return_value
    mock_inference_instance.run.return_value = {'result': 'some_output'}  # Mock inference output

    mock_assembly_instance = mock_assembly.return_value
    mock_assembly_instance.return_value = (["bboxes"], ["classes"], ["scores"])  # Mock tracker model output

    # Instantiate the Interface class
    interface = Interface(group_info=group_info_mock, ModelDict=ModelDict_mock, TrackerDict=TrackerDict_mock, camera_id="camera1")

    # 1. Check if AssemblyInterface and TrackerInterface are called when tracker is present in config
    mock_assembly.assert_called_once_with(model=ModelDict_mock["tracker_model_id"], threshold=0.75)
    mock_tracker.assert_called_once_with(tracker=TrackerDict_mock["camera1"], roi=group_info_mock["tracker"]["roi"]["line"], dir=group_info_mock["tracker"]["roi"]["direction"])

    # 2. Check inferenceInterface is called correct number of times
    assert mock_inference.call_count == len(group_info_mock["cropping"])

    # Mock an image for testing
    mock_image = MagicMock()

    # 3. Test the run method
    result, object_count = interface.run(mock_image)

    # a) Check self.tracker_model(image=image) mock
    mock_assembly_instance.assert_called_once_with(image=mock_image)

    # b) Check self.tracker.run(images, bboxes, classes, scores) mock
    mock_tracker_instance.run.assert_called_once_with([mock_image], ["bboxes"], ["classes"], ["scores"])

    # c) Verify tracker key contains expected result
    assert result["tracker"] == [[199, 313, 760, 918]]

    # d) Verify processer.run(image=image, roi=roi) is called correct number of times
    assert mock_inference_instance.run.call_count == len(group_info_mock["cropping"])

    # Verify final output
    assert 'tracker' in result
    assert 'cropping' in result
    assert object_count == 1

# Run the test
if __name__ == "__main__":
    pytest.main()
