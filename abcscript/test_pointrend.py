# tests/test_faster_rcnn.py
import unittest
import torch
import numpy as np
from assembly.models.detection import Pointrend
from assembly.models.ModelManager import modelManager

class TestPointRend(unittest.TestCase):
    def setUp(self):
        """
        Set up the testing environment. This method is called before each test.
        """
        print("coming to the test")
        
        model_weights,config_path,classes = modelManager.WC_forUT
        self.device = "cuda"  # Use CPU for testing to avoid GPU dependencies
        self.model = Pointrend(model_weights=model_weights, config_path=config_path, classes=classes, device=self.device)

    def test_model_initialization(self):
        """
        Test if the model is initialized properly.
        """
        self.assertIsNotNone(self.model.model, "Model should be initialized.")
        self.assertTrue(self.model.model.training is False, "Model should be in evaluation mode.")

    def test_model_inference(self):
        """
        Test if the model can perform inference on a dummy image.
        """
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)  # Create a black dummy image of size 800x800
        boxes, class_names, scores = self.model(dummy_image)

        # Check that the model returns lists
        self.assertIsInstance(boxes, list, "Boxes should be a list.")
        print("boxes passed")
        self.assertIsInstance(class_names, list, "Class names should be a list.")
        print("classnames passed")
        self.assertIsInstance(scores, list, "Scores should be a list.")

        # You can add more specific assertions here depending on your expected output

    def test_model_warm_up(self):
        """
        Test the warm-up functionality of the model.
        """
        self.model.warm_up()  # Should pass without errors
        self.assertTrue(True, "Warm-up completed without errors.")



    def test_preprocess(self):
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)
        processed_input = self.pointrend.preProcess(dummy_image)
        self.assertIn("image", processed_input)
        self.assertEqual(processed_input["image"].shape, (3, 800, 800))

    def test_forward_pass(self):
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)
        processed_input = self.pointrend.preProcess(dummy_image)
        predictions = self.pointrend.forward(processed_input)
        self.assertIsNotNone(predictions)

    def test_postprocess(self):
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)
        processed_input = self.pointrend.preProcess(dummy_image)
        predictions = self.pointrend.forward(processed_input)
        boxes, classes, scores = self.pointrend.postProcess(predictions)
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(classes, list)
        self.assertIsInstance(scores, np.ndarray)

if __name__ == '__main__':
    unittest.main()

