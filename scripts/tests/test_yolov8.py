import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from ultralytics import YOLO  # Ensure you have ultralytics installed or mock appropriately
from assembly.models.detection.Yolov8_model import YoloV8 # Adjust the import path as needed


@pytest.fixture
def yolov8_instance():
    with patch('assembly.models.detection.Yolov8_model.YOLO') as mock_yolo:
        # Mock the YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        instance = YoloV8(
            model_weights="dummy_weights.pth",
            classes=["class1", "class2"],
            iou_thres=0.3,
            device="cpu",
            imgsz=(640, 640)
        )

        # Mock the model attributes
        mock_model.names = ["class1", "class2"]

        return instance


def test_initialization(yolov8_instance):
    assert yolov8_instance.model_weights == "dummy_weights.pth"
    assert yolov8_instance.classes == ["class1", "class2"]
    assert yolov8_instance.iou_thresh == 0.3
    assert yolov8_instance.device == "cpu"
    assert yolov8_instance.imgsz == (640, 640)
    assert isinstance(yolov8_instance.model, MagicMock)


def test_preprocess(yolov8_instance):
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    output = yolov8_instance.preProcess(dummy_image)

    assert np.array_equal(output, dummy_image)  # Should return the input image unchanged


@patch('assembly.models.detection.Yolov8_model.YoloV8.preProcess')
def test_forward(mock_preprocess, yolov8_instance):
    mock_preprocess.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Mock model prediction output
    mock_output = MagicMock()
    mock_output.boxes.xyxy = torch.tensor([[10, 10, 100, 100]])
    mock_output.boxes.conf = torch.tensor([0.9])
    mock_output.boxes.cls = torch.tensor([0])
    yolov8_instance.model.predict.return_value = [mock_output]

    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    pred_output = yolov8_instance.forward(dummy_image)

    assert pred_output is not None

def test_postprocess(yolov8_instance):
    mock_pred_output = MagicMock()
    mock_pred_output.boxes.xyxy = torch.tensor([[10, 10, 50, 50]])
    mock_pred_output.boxes.conf = torch.tensor([0.99])
    mock_pred_output.boxes.cls = torch.tensor([0])

    boxes, class_names, scores = yolov8_instance.postProcess(mock_pred_output)

    assert isinstance(boxes, list)
    assert len(boxes) == 1
    assert len(class_names) == 1
    assert len(scores) == 1


# @patch('assembly.models.detection.YoloV8_model.YoloV8.to')
# def test_to(mock_to, yolov8_instance):
#     yolov8_instance.to(device="cuda")
#     mock_to.assert_called_with(device="cuda")

#     yolov8_instance.to(device="cpu")
#     mock_to.assert_called_with(device="cpu")


def test_warm_up(yolov8_instance):
    with patch.object(yolov8_instance, 'forward', return_value=None) as mock_forward:
        yolov8_instance.warm_up()
        assert mock_forward.call_count == 2  # Warm up calls forward twice
