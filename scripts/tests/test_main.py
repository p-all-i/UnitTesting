import pytest
from unittest.mock import Mock, patch, MagicMock
import datetime
import sys

import os, sys, requests, json, time
import threading
from dotenv import load_dotenv
# Load environment variables from .env file
# load_dotenv("./.env")
# time.sleep(300000)
import datetime, logging, traceback, sys, torch, cv2
from assembly.model_utils.initialize import  GlobalParameters, extractFrameVCO, select_device, DumpModels, getConfigData
from assembly.model_utils.initiate_loggers import InitLoggers 
from assembly.model_utils.Visualisor import VisualizeResults
from assembly.components.FileVideoStream import NodeCommServer, varientchange_server
from assembly.analysis.analysis_logic import AnalysisLogic
from assembly.interfaces.InterfaceCreation import InterfaceCreation
from assembly.input_validation.validation import validateInput
from assembly.components.mainProcessor import MainProcessor
from assembly.components.prepOutput import OutputPrep

from assembly.components.transmitterManager import TransmitterManager
# from configupdate import ConfigUpdater
import multiprocessing as mp


# Fixtures to mock environment variables and external dependencies

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to mock environment variables."""
    monkeypatch.setenv("MAXBYTES_LOGGER", "100000")
    monkeypatch.setenv("BACKUPCOUNT_LOGGER", "5")
    monkeypatch.setenv("EXCHANGE_PUBLISH", "test_exchange")
    monkeypatch.setenv("QUEUE_PUBLISH", "test_queue")
    monkeypatch.setenv("initial_data_endpoint", "http://test.endpoint")
    monkeypatch.setenv("DOCKER_ID", "test_docker_id")
    monkeypatch.setenv("pika_host", "localhost")
    monkeypatch.setenv("redis_host", "localhost")
    monkeypatch.setenv("SAVE_DIR", "/tmp")
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", "/models")
    monkeypatch.setenv("LOGS_DIR", "/logs")
    monkeypatch.setenv("IMAGESERVER_HOST", "localhost")
    monkeypatch.setenv("IMAGESERVER_PORT", "8000")
    monkeypatch.setenv("CONSUME_QUEUE", "test_queue")

@pytest.fixture
def mock_logger():
    """Fixture to create a mock logger object."""
    logger = Mock()
    logger.logger.info = Mock()
    logger.logger.error = Mock()
    logger.logger.exception = Mock()
    logger.transmitter_logging = Mock(return_value=Mock())
    return logger

@pytest.fixture
def mock_gp():
    """Fixture to create a mock GlobalParameters object."""
    return Mock()

@pytest.fixture
def mock_analysis_logic(mock_logger):
    """Fixture to create a mock AnalysisLogic object."""
    return AnalysisLogic(loggerObj=mock_logger)

@pytest.fixture
def mock_output_prep():
    """Fixture to create a mock OutputPrep object."""
    return Mock()

@pytest.fixture
def mock_main_processor(mock_gp, mock_logger, mock_analysis_logic, mock_output_prep):
    """Fixture to create a mock MainProcessor object."""
    return MainProcessor(
        GP=mock_gp,
        loggerObj=mock_logger,
        analysisLogic=mock_analysis_logic,
        output_sender=Mock(),
        outputprepObj=mock_output_prep,
        visualisor=Mock(),
        save_dir='/tmp',
        shared_queue=Mock(),
        transmitter_manager=Mock(),
        varientchangeServer=Mock()
    )

# Mocking external dependencies like requests, Redis, etc.

@pytest.fixture
def mock_requests_get(monkeypatch):
    """Fixture to mock requests.get."""
    def mock_get(*args, **kwargs):
        response = Mock()
        response.json.return_value = {"mock_key": "mock_value"}
        return response

    monkeypatch.setattr(requests, "get", mock_get)

@pytest.fixture
def mock_get_config_data(monkeypatch):
    """Fixture to mock getConfigData."""
    mock_get_data_instance = Mock()
    mock_get_data_instance.getData.return_value = {"modelsInfo": [{"params": {}, "uuid_class_map": {}}]}
    mock_get_data_instance.update_config_classes.return_value = {"modelsInfo": [{"params": {}, "uuid_class_map": {}}]}
    monkeypatch.setattr(getConfigData, "getData", mock_get_data_instance.getData)
    monkeypatch.setattr(getConfigData, "update_config_classes", mock_get_data_instance.update_config_classes)
    return mock_get_data_instance

def test_initialization(mock_env_vars, mock_logger, mock_requests_get, mock_get_config_data):
    """Test the initialization logic."""
    # Initialize your classes with mocks
    loggerObj = InitLoggers(max_bytes=100000, backup_count=5, save_path='/logs')
    analysisLogic = AnalysisLogic(loggerObj=loggerObj)
    outputprepObj = OutputPrep()
    
    CONFIG_JSON = getConfigData().getData(loggerObj=loggerObj, url="http://test.endpoint?dockerId=test_docker_id")
    CONFIG_JSON = getConfigData().update_config_classes(CONFIG_JSON, loggerObj)

    # Assertions
    assert isinstance(analysisLogic, AnalysisLogic)
    assert isinstance(outputprepObj, OutputPrep)
    assert CONFIG_JSON is not None
    assert 'modelsInfo' in CONFIG_JSON
# @patch('your_module.MainProcessor.handle_variant_change', return_value=(None, False))
# def test_main_processor_run(mock_handle_variant_change, mock_main_processor):
#     """Test the run method of MainProcessor."""
#     # mock_main_processor.run()

#     # Assertions to ensure run method behavior
#     assert mock_main_processor.GP is not None
#     assert mock_main_processor.loggerObj is not None
#     mock_main_processor.analysisLogic.assert_called()
#     mock_main_processor.output_sender.assert_called()
#     mock_main_processor.outputprepObj.assert_called()
#     mock_main_processor.Visualisor.assert_called()

if __name__ == "__main__":
    pytest.main()
