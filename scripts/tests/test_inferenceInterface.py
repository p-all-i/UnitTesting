import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from assembly.interfaces.inferenceInterface import inferenceInterface

# Sample input data for testing
interface_info_mock = {
    "roi": [0, 0, 1, 1],
    "model_1": [
        {
            "model_id": "model_1_id",
            "threshold": {
                "score_thresh": 0.7
            },
            "type": "detection"
        }
    ],
    "model_2": [
        {
            "model_id": "model_2_id",
            "threshold": {
                "score_thresh": 0.8
            },
            "class_name": "4f1a9b88-2d13-4db8-94b4-87d3e7991mo2"
        }
    ]
}

ModelDict_mock = {
    "model_1_id": MagicMock(),
    "model_2_id": MagicMock()
}

@patch('assembly.interfaces.inferenceInterface.AssemblyInterface', autospec=True)
@patch('assembly.interfaces.inferenceInterface.ClassificationInterface', autospec=True)
def test_inference_interface_initialization(mock_classification, mock_assembly):
    # Initialize inferenceInterface with mocks
    interface = inferenceInterface(interface_info=interface_info_mock, ModelDict=ModelDict_mock)

    # Check if attributes are set correctly
    assert interface.interface_info == interface_info_mock
    assert interface.ModelDict == ModelDict_mock
    assert interface.model1_type == "detection"

def test_cropping():
    # Initialize inferenceInterface with mocks
    interface = inferenceInterface(interface_info=interface_info_mock, ModelDict=ModelDict_mock)
    
    # Create a mock image and define a ROI
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = [10, 10, 50, 50]
    
    # Perform cropping
    cropped_image = interface.cropping(image=image, roi=roi)
    
    # Validate cropped image shape
    assert cropped_image.shape == (40, 40, 3)

@patch('assembly.interfaces.inferenceInterface.AssemblyInterface', autospec=True)
@patch('assembly.interfaces.inferenceInterface.ClassificationInterface', autospec=True)
def test_run_detection(mock_classification, mock_assembly):
    # Step 1: Mock the detection model output
    mock_detection_model = mock_assembly.return_value
    mock_detection_model.return_value = (
        [np.array([76, 0, 104, 228]), np.array([3, 27, 38, 228]), np.array([57, 36, 99, 228])],
        ['4f1a9b88-2d13-4db8-94b4-87d3e7991mo2', '4f1a9b88-2d13-4db8-94b4-87d3e7991mo2', '4f1a9b88-2d13-4db8-94b4-87d3e7991mo2'],
        [0.94711053, 0.7712379, 0.768784]
    )

    # Step 2: Mock the classification model output
    mock_classification_instance = mock_classification.return_value
    mock_classification_instance.return_value = ("class_name_1", 0.85)  # Mock classification output

    # Step 3: Initialize inferenceInterface with mocks
    interface = inferenceInterface(interface_info=interface_info_mock, ModelDict=ModelDict_mock)

    # Step 4: Mock an image for testing
    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Step 5: Run inference
    result = interface.run(mock_image)

    # Step 6: Assertions to validate the inference run
    assert "detection" in result
    assert "classification" in result
    assert len(result["detection"]) > 0
    assert len(result["detection"]["4f1a9b88-2d13-4db8-94b4-87d3e7991mo2"]) == 3

    # Ensure that the result dictionary contains the expected structure and values
    detection_result = result["detection"]["4f1a9b88-2d13-4db8-94b4-87d3e7991mo2"][0]
    assert detection_result["box"] == [76, 0, 104, 228]
    assert detection_result["det_score"] == 0.94711053

    # Check if class_name and class_score keys exist before asserting
    assert "class_name" in detection_result
    assert detection_result["class_name"] == "class_name_1"
    assert detection_result["class_score"] == 0.85
def test_retrace():
    # Initialize inferenceInterface with mocks
    interface = inferenceInterface(interface_info=interface_info_mock, ModelDict=ModelDict_mock)
    
    # Define top-left corner and bounding boxes
    top_left_corner = (10, 20)
    bounding_boxes = [[5, 5, 15, 25], [0, 0, 10, 10]]
    
    # Retrace bounding boxes
    retraced_boxes = interface.retrace(top_left_corner=top_left_corner, bounding_boxes=bounding_boxes)
    
    # Validate retraced boxes
    assert retraced_boxes == [[15, 25, 25, 45], [10, 20, 20, 30]]

def test_check_roi():
    # Initialize inferenceInterface with mocks
    interface = inferenceInterface(interface_info=interface_info_mock, ModelDict=ModelDict_mock)
    
    # Mock image shape
    image_shape = (100, 100, 3)
    roi = [0.1, 0.1, 0.5, 0.5]
    
    # Check ROI conver
