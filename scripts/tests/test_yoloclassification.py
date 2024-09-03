import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from ultralytics import YOLO
from assembly.models.classification.yolov8Classification import YoloV8Classification

@pytest.fixture
def yolov8_classification_instance():
    # Mock the YOLO class and its methods
    with patch('assembly.models.classification.yolov8Classification.YOLO') as mock_yolo:
        mock_model = MagicMock(spec=YOLO)
        mock_yolo.return_value = mock_model


        # # Mock the actual loading process
        # with patch('ultralytics.models.yolo.model.torch.load') as mock_torch_load:
        #     mock_torch_load.return_value = MagicMock()
        
        # Initialize instance of YoloV8Classification
        instance = YoloV8Classification(
            model_weights="dummy_weights.pt",
            classes=["class1", "class2", "class3"],
            score_thresh=0.7,
            device="cpu"
        )

        return instance

def test_initialization(yolov8_classification_instance):
    assert yolov8_classification_instance.model_weights == "dummy_weights.pt"
    assert yolov8_classification_instance.classes == ["class1", "class2", "class3"]
    assert yolov8_classification_instance.imgsz == (64, 64)
    assert yolov8_classification_instance.score_thresh == 0.7
    assert yolov8_classification_instance.device == "cpu"
    assert isinstance(yolov8_classification_instance.model, MagicMock)

def test_preprocess(yolov8_classification_instance):
    dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
    processed_image = yolov8_classification_instance.preProcess(dummy_image)
    assert np.array_equal(processed_image, dummy_image)

@patch('assembly.models.classification.yolov8Classification.YoloV8Classification.preProcess')
def test_forward(mock_preprocess, yolov8_classification_instance):
    # Mock the preprocess method
    mock_preprocess.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # Mock the model's predict method
    mock_pred_output = MagicMock()
    mock_pred_output.probs.top5 = [0]
    mock_pred_output.probs.top5conf.cpu.return_value = torch.tensor([0.9])
    yolov8_classification_instance.model.predict.return_value = [mock_pred_output]
    
    dummy_input = np.zeros((64, 64, 3), dtype=np.uint8)
    output = yolov8_classification_instance.forward(dummy_input)

    assert output is not None
    

def test_postprocess(yolov8_classification_instance):
    mock_pred_output = MagicMock()
    mock_pred_output.probs.top5 = [0]
    mock_pred_output.probs.top5conf.cpu.return_value = torch.tensor([0.9])
    
    class_name, class_conf = yolov8_classification_instance.postProcess(mock_pred_output)

    assert class_name == "class1"
    assert class_conf == 0.8999999761581421 # 0.9

def test_call(yolov8_classification_instance):
    with patch.object(yolov8_classification_instance, 'preProcess', return_value=np.zeros((64, 64, 3), dtype=np.uint8)) as mock_preprocess, \
         patch.object(yolov8_classification_instance, 'forward', return_value=MagicMock()) as mock_forward, \
         patch.object(yolov8_classification_instance, 'postProcess', return_value=("class1", 0.9)) as mock_postprocess:
        
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        class_name, class_conf = yolov8_classification_instance(dummy_image)

        assert class_name == "class1"
        assert class_conf == 0.9
        mock_preprocess.assert_called_once()
        mock_forward.assert_called_once()
        mock_postprocess.assert_called_once()

# def test_to(yolov8_classification_instance):
#     with patch.object(yolov8_classification_instance.model, 'cpu') as mock_cpu, \
#          patch.object(yolov8_classification_instance.model, 'cuda') as mock_cuda:
        
#         yolov8_classification_instance.to(device="cuda")
#         mock_cuda.assert_called_once()

#         yolov8_classification_instance.to(device="cpu")
#         mock_cpu.assert_called_once()
