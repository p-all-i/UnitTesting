import pytest
from unittest.mock import Mock, patch, mock_open
from assembly.models.ModelManager import modelManager
from assembly.model_utils.initialize import getConfigData
from assembly.model_utils.initiate_loggers import InitLoggers
import os

@pytest.fixture
def setup():
    # Initialize environment variables and logger
    MODELS_DIR = os.getenv("MODEL_WEIGHTS_DIR")
    LOGS_DIR = os.getenv("LOGS_DIR")
    max_bytes = int(os.getenv("MAXBYTES_LOGGER"))
    backup_count = int(os.getenv("BACKUPCOUNT_LOGGER"))
    loggerObj = InitLoggers(max_bytes, backup_count, save_path=LOGS_DIR)

    # Fetch the configuration data from the backend
    url1 = os.getenv('initial_data_endpoint')
    CONFIG_JSON = getConfigData().getData(loggerObj=loggerObj, url=url1)
    CONFIG_JSON = getConfigData().update_config_classes(CONFIG_JSON, loggerObj)

    # Adjusting to match the actual structure of the CONFIG_JSON
    model_params = {'8ef7af83-c241-4a11-9553-52d31403c49e': {'model_path': '8ef7af83-c241-4a11-9553-52d31403c49e/best.pth', 'model_properties': {'config': '8ef7af83-c241-4a11-9553-52d31403c49e/args.yaml', 'id': '8ef7af83-c241-4a11-9553-52d31403c49e', 'modelKey': 2, 'params': {'classes': ['6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0', 'fb338f4e-5774-4412-9334-92b5738f7dbf']}, 'threshold': 0.07, 'type': 'assembly', 'uuid_class_map': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 'Tag', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0': 'Label', 'fb338f4e-5774-4412-9334-92b5738f7dbf': 'Bag'}, 'weights': '8ef7af83-c241-4a11-9553-52d31403c49e/best.pth'}}, '36eceb6e-7176-41de-b424-220b05afe4d2': {'model_path': '36eceb6e-7176-41de-b424-220b05afe4d2/best.pth', 'model_properties': {'config': '36eceb6e-7176-41de-b424-220b05afe4d2/args.yaml', 'id': '36eceb6e-7176-41de-b424-220b05afe4d2', 'modelKey': 2, 'params': {'classes': ['6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0', 'fb338f4e-5774-4412-9334-92b5738f7dbf']}, 'threshold': 0.07, 'type': 'assembly', 'uuid_class_map': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 'Tag', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0': 'Label', 'fb338f4e-5774-4412-9334-92b5738f7dbf': 'Bag'}, 'weights': '36eceb6e-7176-41de-b424-220b05afe4d2/best.pth'}}, '767e8e64-9128-4c79-bacd-24ed298e9817': {'model_path': '767e8e64-9128-4c79-bacd-24ed298e9817/best.pth', 'model_properties': {'config': '767e8e64-9128-4c79-bacd-24ed298e9817/args.yaml', 'id': '767e8e64-9128-4c79-bacd-24ed298e9817', 'modelKey': 2, 'params': {'classes': ['6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0', 'fb338f4e-5774-4412-9334-92b5738f7dbf']}, 'threshold': 0.07, 'type': 'assembly', 'uuid_class_map': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 'Tag', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0': 'Label', 'fb338f4e-5774-4412-9334-92b5738f7dbf': 'Bag'}, 'weights': '767e8e64-9128-4c79-bacd-24ed298e9817/best.pth'}}}
    
    # Create a model manager instance using the actual configuration data
    model_manager = modelManager(
        model_params=model_params,
        loggerObj=loggerObj,
        MODELS_DIR=MODELS_DIR
    )

    return model_manager, model_params, loggerObj, MODELS_DIR

def test_initialization(setup):
    model_manager, model_params, loggerObj, MODELS_DIR = setup
    # Test that the modelManager initializes correctly
    assert model_manager.model_params == model_params
    assert model_manager.loggerObj == loggerObj
    assert model_manager.MODELS_DIR == MODELS_DIR

@patch('assembly.models.ModelManager.FasterRCNN')
@patch('assembly.models.ModelManager.YoloV8')
@patch('assembly.models.ModelManager.os.path.exists', return_value=True)
@patch('assembly.models.ModelManager.open', new_callable=mock_open, read_data=b'\x00' * 16)
def test_load_models_with_different_types(mock_open_file, mock_path_exists, MockYoloV8, MockFasterRCNN, setup):
    model_manager, model_params, loggerObj, MODELS_DIR = setup

    # Mocking active models list from the actual config
    active_models = list(model_params.keys())
    print("my active model should be---->", active_models)
    
    # Create mock instances of models
    mock_faster_rcnn_instance = MockFasterRCNN.return_value
    mock_yolov8_instance = MockYoloV8.return_value
    
    # Create a mock model_dict to compare against
    expected_model_dict = {'8ef7af83-c241-4a11-9553-52d31403c49e': "<assembly.models.detection.Pointrend.PointRend object at 0x7f394a41ceb0>", '36eceb6e-7176-41de-b424-220b05afe4d2': "<assembly.models.detection.Pointrend.PointRend object at 0x7f394a41cee0>", '767e8e64-9128-4c79-bacd-24ed298e9817': "<assembly.models.detection.Pointrend.PointRend object at 0x7f394a40c7c0>"}
    # Load models
    actual_model_dict = model_manager.load_models(model_dict={}, active_models=active_models)

    # Compare the actual model_dict with the expected mock model_dict
    assert len(actual_model_dict) == len(expected_model_dict)
    for key in expected_model_dict:
        assert key in actual_model_dict
        assert isinstance(actual_model_dict[key], type(expected_model_dict[key]))

def test_load_models_without_active_models(setup):
    model_manager, model_params, loggerObj, MODELS_DIR = setup

    # Mocking active models list as empty
    active_models = []

    # Dictionary to hold the loaded models
    model_dict = {}

    # Load models with no active models
    model_dict = model_manager.load_models(model_dict=model_dict, active_models=active_models)

    # Check that no models were loaded
    assert len(model_dict) == 0
