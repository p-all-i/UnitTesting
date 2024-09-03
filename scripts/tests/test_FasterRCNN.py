import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.meta_arch import GeneralizedRCNN  # Adjusted import
# Import your FasterRCNN class
from assembly.models.detection.Frcnn_model import FasterRCNN  # Update import path as needed

@pytest.fixture
def faster_rcnn_instance():
    # Mock the configuration object and its methods
    with patch('assembly.models.detection.Frcnn_model.build_model') as mock_build_model, \
         patch('assembly.models.detection.Frcnn_model.DetectionCheckpointer'), \
         patch('assembly.models.detection.Frcnn_model.ResizeShortestEdge'), \
         patch('assembly.models.detection.Frcnn_model.model_zoo.get_config_file') as mock_get_config_file,\
         patch('assembly.models.detection.Frcnn_model.FasterRCNN.preProcess', return_value={'np_image': MagicMock()}), \
         patch('assembly.models.detection.Frcnn_model.get_cfg') as mock_get_cfg:

        # Mock the model as GeneralizedRCNN
        mock_model = MagicMock(spec=GeneralizedRCNN)
        mock_build_model.return_value = mock_model

        # Mock the configuration
        mock_cfg = MagicMock()
        mock_get_cfg.return_value = mock_cfg

        # Set up the mock configuration values
        mock_cfg.INPUT.FORMAT = "RGB"
        mock_cfg.merge_from_file = MagicMock()

        # Mock the model zoo to return a dummy config path
        mock_get_config_file.return_value = "dummy_path.yaml"

        instance = FasterRCNN(
            model_weights="dummy_weights.pth",
            config_path="dummy_config.yaml",
            classes=["class1", "class2"],
            device="cpu"
        )

        # Manually initialize `np_image` if needed
        instance.np_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Example initialization

    return instance

def test_initialization(faster_rcnn_instance):
    assert isinstance(faster_rcnn_instance.cfg, MagicMock)
    assert faster_rcnn_instance.classes == ["class1", "class2"]
    assert isinstance(faster_rcnn_instance.model, GeneralizedRCNN)

def test_preprocess(faster_rcnn_instance):
    # Mock np_image to bypass the RGB to BGR conversion
    faster_rcnn_instance.np_image = np.zeros((100, 100, 3), dtype=np.uint8)

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    inputs = faster_rcnn_instance.preProcess(dummy_image)

    assert isinstance(inputs, dict)
    assert "image" in inputs
    assert "height" in inputs
    assert "width" in inputs

@patch('assembly.models.detection.Frcnn_model.torch.no_grad')
@patch('assembly.models.detection.Frcnn_model.FasterRCNN.preProcess')
def test_forward(mock_no_grad, faster_rcnn_instance):
    # Mock the model to return a valid output
    faster_rcnn_instance.np_image = MagicMock(name='np_image')
    mock_output = {"instances": MagicMock()}
    faster_rcnn_instance.model = MagicMock()
    faster_rcnn_instance.model.return_value = [mock_output]

    dummy_input = {"image": torch.zeros((3, 100, 100)), "height": 100, "width": 100}
    output = faster_rcnn_instance.forward(dummy_input)

    assert output is not None

def test_postprocess(faster_rcnn_instance):
    mock_pred_output = {
        "instances": MagicMock(
            pred_boxes=MagicMock(tensor=torch.tensor([[10, 10, 50, 50]])),
            pred_classes=torch.tensor([0]),
            scores=torch.tensor([0.99])
        )
    }
    boxes, class_names, scores = faster_rcnn_instance.postProcess(mock_pred_output)

    assert isinstance(boxes, list)
    assert len(class_names) == 1
    assert len(scores) == 1

# def test_to(faster_rcnn_instance):
#     with patch.object(faster_rcnn_instance.model, 'to') as mock_to:
#         faster_rcnn_instance.to(device="cuda")
#         mock_to.assert_called_with(torch.device('cuda'))

#         faster_rcnn_instance.to(device="cpu")
#         mock_to.assert_called_with(torch.device('cpu'))

def test_warm_up(faster_rcnn_instance):
    # Mock the forward pass
    with patch.object(faster_rcnn_instance, 'forward', return_value=None) as mock_forward:
        faster_rcnn_instance.warm_up()
        assert mock_forward.call_count == 2  # Because warm_up calls forward twice
