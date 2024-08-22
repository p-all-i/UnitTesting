import unittest
from unittest.mock import Mock, patch, mock_open
from assembly.models.ModelManager import modelManager
from assembly.model_utils.initialize import getConfigData
from assembly.model_utils.initiate_loggers import InitLoggers
import os

class TestModelManager(unittest.TestCase):

    def setUp(self):
        # Initialize environment variables and logger
        self.MODELS_DIR = os.getenv("MODEL_WEIGHTS_DIR")
        self.LOGS_DIR = os.getenv("LOGS_DIR")
        max_bytes = int(os.getenv("MAXBYTES_LOGGER"))
        backup_count = int(os.getenv("BACKUPCOUNT_LOGGER"))
        self.loggerObj = InitLoggers(max_bytes, backup_count, save_path=self.LOGS_DIR)

        # Fetch the configuration data from the backend
        url1 = os.getenv('initial_data_endpoint')
        CONFIG_JSON = getConfigData().getData(loggerObj=self.loggerObj, url=url1)
        CONFIG_JSON = getConfigData().update_config_classes(CONFIG_JSON, self.loggerObj)

        # Adjusting to match the actual structure of the CONFIG_JSON
        self.model_params = {'8ef7af83-c241-4a11-9553-52d31403c49e': {'model_path': '8ef7af83-c241-4a11-9553-52d31403c49e/best.pth', 'model_properties': {'config': '8ef7af83-c241-4a11-9553-52d31403c49e/args.yaml', 'id': '8ef7af83-c241-4a11-9553-52d31403c49e', 'modelKey': 2, 'params': {'classes': ['6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0', 'fb338f4e-5774-4412-9334-92b5738f7dbf']}, 'threshold': 0.07, 'type': 'assembly', 'uuid_class_map': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 'Tag', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0': 'Label', 'fb338f4e-5774-4412-9334-92b5738f7dbf': 'Bag'}, 'weights': '8ef7af83-c241-4a11-9553-52d31403c49e/best.pth'}}, '36eceb6e-7176-41de-b424-220b05afe4d2': {'model_path': '36eceb6e-7176-41de-b424-220b05afe4d2/best.pth', 'model_properties': {'config': '36eceb6e-7176-41de-b424-220b05afe4d2/args.yaml', 'id': '36eceb6e-7176-41de-b424-220b05afe4d2', 'modelKey': 2, 'params': {'classes': ['6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0', 'fb338f4e-5774-4412-9334-92b5738f7dbf']}, 'threshold': 0.07, 'type': 'assembly', 'uuid_class_map': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 'Tag', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0': 'Label', 'fb338f4e-5774-4412-9334-92b5738f7dbf': 'Bag'}, 'weights': '36eceb6e-7176-41de-b424-220b05afe4d2/best.pth'}}, '767e8e64-9128-4c79-bacd-24ed298e9817': {'model_path': '767e8e64-9128-4c79-bacd-24ed298e9817/best.pth', 'model_properties': {'config': '767e8e64-9128-4c79-bacd-24ed298e9817/args.yaml', 'id': '767e8e64-9128-4c79-bacd-24ed298e9817', 'modelKey': 2, 'params': {'classes': ['6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0', 'fb338f4e-5774-4412-9334-92b5738f7dbf']}, 'threshold': 0.07, 'type': 'assembly', 'uuid_class_map': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 'Tag', 'a509c9d1-9492-4ea2-908a-7c9aa129bff0': 'Label', 'fb338f4e-5774-4412-9334-92b5738f7dbf': 'Bag'}, 'weights': '767e8e64-9128-4c79-bacd-24ed298e9817/best.pth'}}}
        
        # Create a model manager instance using the actual configuration data
        self.model_manager = modelManager(
            model_params=self.model_params,
            loggerObj=self.loggerObj,
            MODELS_DIR=self.MODELS_DIR
        )

    def test_initialization(self):
        # Test that the modelManager initializes correctly
        self.assertEqual(self.model_manager.model_params, self.model_params)
        self.assertEqual(self.model_manager.loggerObj, self.loggerObj)
        self.assertEqual(self.model_manager.MODELS_DIR, self.MODELS_DIR)

    @patch('assembly.models.ModelManager.FasterRCNN')
    @patch('assembly.models.ModelManager.YoloV8')
    # @patch('assembly.models.ModelManager.PointRend')
    @patch('assembly.models.ModelManager.os.path.exists', return_value=True)
    @patch('assembly.models.ModelManager.open', new_callable=mock_open, read_data=b'\x00' * 16)


    
    def test_load_models_with_different_types(self, mock_open_file, mock_path_exists, MockYoloV8, MockFasterRCNN ):#, MockPointRend):
        # Mocking active models list from the actual config
        active_models = list(self.model_params.keys())
        print("my activat model whould be---->", active_models)
        
        # Create mock instances of models
        mock_faster_rcnn_instance = MockFasterRCNN.return_value
        mock_yolov8_instance = MockYoloV8.return_value
        # mock_pointrend_instance = MockPointRend.return_value
        
        # Create a mock model_dict to compare against
        expected_model_dict = {'8ef7af83-c241-4a11-9553-52d31403c49e': "<assembly.models.detection.Pointrend.PointRend object at 0x7f394a41ceb0>", '36eceb6e-7176-41de-b424-220b05afe4d2': "<assembly.models.detection.Pointrend.PointRend object at 0x7f394a41cee0>", '767e8e64-9128-4c79-bacd-24ed298e9817': "<assembly.models.detection.Pointrend.PointRend object at 0x7f394a40c7c0>"}


        # Load models
        actual_model_dict = self.model_manager.load_models(model_dict={}, active_models=active_models)

        # Compare the actual model_dict with the expected mock model_dict
        self.assertEqual(len(actual_model_dict), len(expected_model_dict))
        for key in expected_model_dict:
            self.assertIn(key, actual_model_dict)
            # self.assertIsInstance(actual_model_dict[key], type(expected_model_dict[key]))


    def test_load_models_without_active_models(self):
        # Mocking active models list as empty
        active_models = []

        # Dictionary to hold the loaded models
        model_dict = {}

        # Load models with no active models
        model_dict = self.model_manager.load_models(model_dict=model_dict, active_models=active_models)

        # Check that no models were loaded
        self.assertEqual(len(model_dict), 0)

if __name__ == '__main__':
    unittest.main()





