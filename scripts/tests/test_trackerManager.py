import pytest
from unittest.mock import Mock, patch

from assembly.models.tracking.centroidTracker import CentroidTracker
from assembly.models.tracking.ocSort_tracker import ocTracker
from assembly.models.TrackerManager import trackerManager

@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def tracker_manager(mock_logger):
    return trackerManager(mock_logger)

@patch('assembly.models.TrackerManager.CentroidTracker', autospec=True)
@patch('assembly.models.TrackerManager.ocTracker', autospec=True)
def test_load_trackers(MockOcTracker, MockCentroidTracker, tracker_manager):
    trackerDict = {}
    active_trackers = {
        'camera1': {'type': 'centroid', 'maxDistance': 50, 'line': [(0, 0), (100, 100)], 'direction': 'up'},
        'camera2': {'type': 'ocsort', 'line': [(0, 0), (200, 200)], 'direction': 'down'}
    }

    # Call load_trackers and verify correct objects are created
    trackerDict = tracker_manager.load_trackers(trackerDict, active_trackers)

    # Check if CentroidTracker was instantiated with correct parameters
    MockCentroidTracker.assert_called_once_with(maxDistance=50, ROI=[(0, 0), (100, 100)], movement_direction='up')
    # Check if ocTracker was instantiated with correct parameters
    MockOcTracker.assert_called_once_with(movement_direction='down', ROI=[(0, 0), (200, 200)])

   # Check if the trackerDict is updated correctly
    assert 'camera1' in trackerDict
    assert 'camera2' in trackerDict
    assert isinstance(trackerDict['camera1'], type(MockCentroidTracker.return_value))
    assert isinstance(trackerDict['camera2'], type(MockOcTracker.return_value))

    # Verify logger calls, ensuring correct messages for creation are logged
    tracker_manager.loggerObj.logger.info.assert_any_call("Tracker object of type CENTROID TRACKER created for camera1")
    tracker_manager.loggerObj.logger.info.assert_any_call("Tracker object of type OCSORT created for camera2")


@patch('assembly.models.tracking.centroidTracker.CentroidTracker', autospec=True)
@patch('assembly.models.tracking.ocSort_tracker.ocTracker', autospec=True)
def test_load_trackers_already_loaded(MockOcTracker, MockCentroidTracker, tracker_manager):
    trackerDict = {'camera1': Mock()}
    active_trackers = {
        'camera1': {'type': 'centroid', 'maxDistance': 50, 'line': [(0, 0), (100, 100)], 'direction': 'up'},
    }

    # Call load_trackers when tracker is already in trackerDict
    trackerDict = tracker_manager.load_trackers(trackerDict, active_trackers)

    # Check that no new trackers are added and existing is preserved
    assert len(trackerDict) == 1
    MockCentroidTracker.assert_not_called()  # Ensure no new tracker is created
    tracker_manager.loggerObj.logger.info.assert_called_once_with("[INFO] trackerManager started")  # Adjust to only check for the initial log message

def test_load_trackers_invalid_type(tracker_manager):
    trackerDict = {}
    active_trackers = {
        'camera3': {'type': 'unknown', 'maxDistance': 50, 'line': [(0, 0), (100, 100)], 'direction': 'up'},
    }

    # Check exception for invalid tracker type
    with pytest.raises(Exception) as exc_info:
        tracker_manager.load_trackers(trackerDict, active_trackers)
    
    assert "[INFO] Tracker not implemented" in str(exc_info.value)
    tracker_manager.loggerObj.logger.exception.assert_called_once_with("[INFO] Tried loading a Tracker that is not Implemented")
