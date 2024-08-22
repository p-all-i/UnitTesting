import unittest
from unittest.mock import Mock
from assembly.model_utils.initialize import  GlobalParameters, getConfigData
from assembly.model_utils.initiate_loggers import InitLoggers 
import os 
url1 = os.getenv('initial_data_endpoint')
LOGS_DIR = os.getenv("LOGS_DIR")
max_bytes = int(os.getenv("MAXBYTES_LOGGER"))
backup_count = int(os.getenv("BACKUPCOUNT_LOGGER"))
loggerObj = InitLoggers(max_bytes, backup_count, save_path=LOGS_DIR)
CONFIG_JSON = getConfigData().getData(loggerObj=loggerObj, url= url1)
#-------------------------------------------------------------------------------
CONFIG_JSON = getConfigData().update_config_classes(CONFIG_JSON, loggerObj)

MODELS_DIR = os.getenv("MODEL_WEIGHTS_DIR")
class TestGlobalParameters(unittest.TestCase):
    def setUp(self):
        # Mock the dependencies
        self.device = 'cuda'
        self.config_data = CONFIG_JSON
        # {
        #     "cameraInfo": {...},  # Add relevant test data
        #     "trackersInfo": [...],
        #     "modelsInfo": [...]
        # }
        self.loggerObj = Mock()
        self.EXCHANGE_PUBLISH = "test_exchange"
        self.QUEUE_PUBLISH = "test_queue"
        self.MODELS_DIR = MODELS_DIR

        # Instantiate GlobalParameters
        self.gp = GlobalParameters(self.device, self.config_data, self.loggerObj, self.EXCHANGE_PUBLISH, self.QUEUE_PUBLISH, self.MODELS_DIR)

    def test_initialization(self):
        self.assertIsNotNone(self.gp.ModelDict)
        self.assertIsNotNone(self.gp.TrackerDict)

    def test_load_models(self):
        self.gp.loadModels()
        self.assertGreater(len(self.gp.ModelDict), 0)

    def test_load_trackers(self):
        self.gp.loadTrackers()
        
        # Check if trackersInfo is present in the config data
        if 'trackersInfo' in self.config_data and len(self.config_data['trackersInfo']) > 0:
            # If trackersInfo exists, assert that TrackerDict is populated
            self.assertGreater(len(self.gp.TrackerDict), 0)
        else:
            # If trackersInfo does not exist, TrackerDict should be empty
            self.assertEqual(len(self.gp.TrackerDict), 0)

if __name__ == "__main__":
    unittest.main()
