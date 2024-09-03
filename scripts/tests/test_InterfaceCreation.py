import pytest
from unittest.mock import MagicMock, patch
from collections import OrderedDict
from assembly.interfaces.InterfaceCreation import InterfaceCreation  # Replace with the actual import path
from assembly.interfaces.interface import Interface
# Mocking the Interface class to prevent actual initialization
@pytest.fixture
def mock_interface():
    with patch('assembly.interfaces.InterfaceCreation.Interface', autospec=True) as mock:
        yield mock

# Mocking the GP object with necessary attributes
@pytest.fixture
def mock_gp():
    gp = MagicMock()
    
    # Add all required model keys to the mocked ModelDict
    gp.ModelDict = {
        "model1": "mock_model_1",
        "767e8e64-9128-4c79-bacd-24ed298e9817": "mock_model_2",
        "36eceb6e-7176-41de-b424-220b05afe4d2": "mock_model_3",
        # Add any additional models here as needed
    }
    
    gp.TrackerDict = {
        "tracker1": "mock_tracker"
        # Add any additional trackers here as needed
    }
    
    # Using a single relevant camera configuration
    gp.cameraParams = {
        "camera1": [
            {
                'cropping': {
                    '611c5f9a-18f0-4ef5-812a-f23d5ff8e5cf': {
                        'ground_truth': {
                            '6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': {'count': 1, 'operator': 0}
                        },
                        'model_1': [
                            {'model_id': '767e8e64-9128-4c79-bacd-24ed298e9817', 'threshold': {'score_thresh': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 0.7}}, 'type': 'detection'}
                        ],
                        'model_2': [],
                        'roi': [0.7428, 0.4214, 0.9299, 0.9146]
                    }
                },
                'order': 0,
                'steps': ['cropping']
            }
        ]
    }
    
    gp.interfaceObjs = OrderedDict()
    return gp

# Mocking the logger object
@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.logger = MagicMock()
    return logger

def test_create_interfaces(mock_gp, mock_logger, mock_interface):
    # Calling the create method
    InterfaceCreation.create(mock_gp, mock_logger)
    
    # Assert the number of interface objects created
    assert len(mock_gp.interfaceObjs) == 1  # Only one camera
    assert len(mock_gp.interfaceObjs["camera1"]) == 1  # One configuration for this camera
    
    # Check that Interface is called with correct parameters
    mock_interface.assert_called_once_with(
        group_info={
            'cropping': {
                '611c5f9a-18f0-4ef5-812a-f23d5ff8e5cf': {
                    'ground_truth': {
                        '6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': {'count': 1, 'operator': 0}
                    },
                    'model_1': [
                        {'model_id': '767e8e64-9128-4c79-bacd-24ed298e9817', 'threshold': {'score_thresh': {'6cbd1e61-76d0-4c84-b2ff-532a1ebae7d9': 0.7}}, 'type': 'detection'}
                    ],
                    'model_2': [],
                    'roi': [0.7428, 0.4214, 0.9299, 0.9146]
                }
            },
            'order': 0,
            'steps': ['cropping']
        },
        ModelDict=mock_gp.ModelDict,
        TrackerDict=mock_gp.TrackerDict,
        camera_id="camera1"
    )
    
    # Assert the logging messages were called correctly
    mock_logger.logger.info.assert_called_once_with("Created a Interface for Camera id: camera1 & image id: 0")
