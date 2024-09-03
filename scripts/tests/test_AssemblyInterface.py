import pytest
from unittest.mock import MagicMock, patch

# Import the AssemblyInterface class
from assembly.interfaces.assemblyInterface import AssemblyInterface

def test_initialization():
    mock_model = MagicMock()
    threshold = 0.5
    logger = MagicMock()

    interface = AssemblyInterface(model=mock_model, threshold=threshold, loggerObj=logger)

    assert interface.model == mock_model
    assert interface.conf_thresh == threshold
    assert interface.loggerObj == logger

def test_preprocess():
    mock_model = MagicMock()
    mock_image = MagicMock()
    mock_model.preProcess.return_value = "preprocessed_image"

    interface = AssemblyInterface(model=mock_model, threshold=0.5)
    result = interface.preProcess(mock_image)

    mock_model.preProcess.assert_called_once_with(mock_image)
    assert result == "preprocessed_image"

def test_forward():
    mock_model = MagicMock()
    mock_input = MagicMock()
    mock_model.forward.return_value = "pred_output"

    interface = AssemblyInterface(model=mock_model, threshold=0.5)
    result = interface.forward(mock_input)

    mock_model.forward.assert_called_once_with(mock_input)
    assert result == "pred_output"

def test_postprocess():
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_model.postProcess.return_value = (["bbox1"], ["class1"], [0.9])

    interface = AssemblyInterface(model=mock_model, threshold=0.5)
    bboxes, classes, scores = interface.postProcess(mock_output)

    mock_model.postProcess.assert_called_once_with(mock_output)
    assert bboxes == ["bbox1"]
    assert classes == ["class1"]
    assert scores == [0.9]

@patch.object(AssemblyInterface, 'applyConfThresh', return_value=(["bbox1"], ["class1"], [0.9]))
@patch.object(AssemblyInterface, 'preProcess', return_value="preprocessed_image")
@patch.object(AssemblyInterface, 'forward', return_value="pred_output")
@patch.object(AssemblyInterface, 'postProcess', return_value=(["bbox1"], ["class1"], [0.9]))
def test_call(mock_postProcess, mock_forward, mock_preProcess, mock_applyConfThresh):
    mock_model = MagicMock()
    interface = AssemblyInterface(model=mock_model, threshold=0.5)
    mock_image = MagicMock()

    # Run the __call__ method
    bboxes, classes, scores = interface(mock_image)

    # Update the assertions to use keyword arguments
    mock_preProcess.assert_called_once_with(image=mock_image)
    mock_forward.assert_called_once_with(input="preprocessed_image")
    mock_postProcess.assert_called_once_with(pred_output="pred_output")
    mock_applyConfThresh.assert_called_once_with(bboxes=["bbox1"], classes=["class1"], confs=[0.9])

    # Validate the output
    assert bboxes == ["bbox1"]
    assert classes == ["class1"]
    assert scores == [0.9]

# # Run the tests
# if __name__ == "__main__":
#     pytest.main()

def test_apply_conf_thresh():
    interface = AssemblyInterface(model=MagicMock(), threshold=0.75)
    bboxes = ["bbox1", "bbox2"]
    classes = ["class1", "class2"]
    confs = [0.8, 0.6]

    result_bboxes, result_classes, result_confs = interface.applyConfThresh(bboxes, classes, confs)

    assert result_bboxes == ["bbox1"]
    assert result_classes == ["class1"]
    assert result_confs == [0.8]

def test_apply_classwise_conf_thresh():
    interface = AssemblyInterface(model=MagicMock(), threshold={"class1": 0.7, "class2": 0.8})
    bboxes = ["bbox1", "bbox2", "bbox3"]
    classes = ["class1", "class2", "class1"]
    confs = [0.75, 0.85, 0.65]

    result_bboxes, result_classes, result_confs = interface.applyClasswiseConfThresh(bboxes, classes, confs)

    assert result_bboxes == ["bbox1", "bbox2"]
    assert result_classes == ["class1", "class2"]
    assert result_confs == [0.75, 0.85]

def test_create_res_json():
    mock_model = MagicMock()
    mock_model.classes = {0: "class1", 1: "class2"}

    interface = AssemblyInterface(model=mock_model, threshold=0.5)
    bboxes = ["bbox1", "bbox2"]
    classes = [0, 1]
    confs = [0.8, 0.6]

    result_json = interface.createResJson(bboxes, classes, confs)

    expected_json = [
        {"partName": "class1", "count": 1, "boxes": ["bbox1"]},
        {"partName": "class2", "count": 1, "boxes": ["bbox2"]}
    ]

    assert result_json == expected_json

# Run the tests
if __name__ == "__main__":
    pytest.main()
