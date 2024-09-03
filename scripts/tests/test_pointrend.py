import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from detectron2.modeling.meta_arch import GeneralizedRCNN
from assembly.models.detection.Pointrend import PointRend  # Update import path as needed

@pytest.fixture
def point_rend_instance():
    # Mock the configuration object and its methods
    with patch('assembly.models.detection.Pointrend.build_model') as mock_build_model, \
         patch('assembly.models.detection.Pointrend.DetectionCheckpointer'), \
         patch('assembly.models.detection.Pointrend.ResizeShortestEdge'), \
         patch('assembly.models.detection.Pointrend.model_zoo.get_config_file') as mock_get_config_file, \
         patch('assembly.models.detection.Pointrend.PointRend.preProcess', return_value={'np_image': MagicMock()}), \
         patch('assembly.models.detection.Pointrend.get_cfg') as mock_get_cfg:

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

        instance = PointRend(
            model_weights="dummy_weights.pth",
            config_path="dummy_config.yaml",
            classes=["class1", "class2"],
            device="cpu"
        )

        # Manually initialize `np_image` if needed
        instance.np_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Example initialization

    return instance

def test_initialization(point_rend_instance):
    assert isinstance(point_rend_instance.cfg, MagicMock)
    assert point_rend_instance.classes == ["class1", "class2"]
    assert isinstance(point_rend_instance.model, GeneralizedRCNN)

def test_preprocess(point_rend_instance):
    # Mock np_image to bypass the RGB to BGR conversion
    point_rend_instance.np_image = np.zeros((100, 100, 3), dtype=np.uint8)

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    inputs = point_rend_instance.preProcess(dummy_image)

    assert isinstance(inputs, dict)
    assert "image" in inputs
    assert "height" in inputs
    assert "width" in inputs

@patch('assembly.models.detection.Pointrend.torch.no_grad')
@patch('assembly.models.detection.Pointrend.PointRend.preProcess')
def test_forward(mock_preprocess, mock_no_grad, point_rend_instance):
    # Mock the model to return a valid output
    point_rend_instance.np_image = MagicMock(name='np_image')
    mock_output = {"instances": MagicMock()}
    point_rend_instance.model = MagicMock()
    point_rend_instance.model.return_value = [mock_output]

    dummy_input = {"image": torch.zeros((3, 100, 100)), "height": 100, "width": 100}
    output = point_rend_instance.forward(dummy_input)

    assert output is not None
    mock_no_grad.assert_called_once()

def test_postprocess(point_rend_instance):
    mock_pred_output = {
        "instances": MagicMock(
            pred_masks=torch.zeros((1, 100, 100)),
            pred_classes=torch.tensor([0]),
            pred_boxes=MagicMock(tensor=torch.tensor([[10, 10, 50, 50]])),
            scores=torch.tensor([0.99])
        )
    }
    boxes, class_names, scores = point_rend_instance.postProcess(mock_pred_output)

    assert isinstance(boxes, np.ndarray)
    assert len(class_names) == 1
    assert len(scores) == 1

def test_warm_up(point_rend_instance):
    # Mock the forward pass
    with patch.object(point_rend_instance, 'forward', return_value=None) as mock_forward:
        point_rend_instance.warm_up()
        assert mock_forward.call_count == 4  # Because warm_up calls forward four times
