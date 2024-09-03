import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from assembly.models.tracking.ocsort.ocsort import OCSort, KalmanBoxTracker
import filterpy
from filterpy.kalman import KalmanFilter
@pytest.fixture
def oc_sort():
    return OCSort(det_thresh=0.5, movement_direction="left2right", ROI=300, max_age=5, min_hits=1, iou_threshold=0.3)

@pytest.fixture
def mock_kalman_box_tracker():
    with patch('assembly.models.tracking.ocsort.ocsort.KalmanBoxTracker', autospec=True) as mock_tracker:
        instance = mock_tracker.return_value
        # Mock attributes and methods required by KalmanBoxTracker
        instance.last_observation = np.array([-1, -1, -1, -1, -1])
        instance.predict.return_value = [np.array([10, 10, 20, 20])]
        instance.update.return_value = None
        instance.time_since_update = 0
        instance.hit_streak = 0
        instance.hits = 0
        instance.age = 0
        instance.get_state.return_value = np.array([[10, 10, 20, 20]])
        # Mock the `id` and `cls` attributes
        instance.id = 0
        instance.cls = 1
        
        yield mock_tracker

def test_initialization(oc_sort):
    assert oc_sort.det_thresh == 0.5
    assert oc_sort.movement_direction == "left2right"
    assert oc_sort.ROI == 300
    assert oc_sort.max_age == 5
    assert oc_sort.min_hits == 1
    assert oc_sort.iou_threshold == 0.3
    assert oc_sort.trackers == []
    assert oc_sort.frame_count == 0

def test_update_empty_detections(oc_sort):
    dets = np.empty((0, 5))
    confs = np.empty((0, ))
    classes = np.empty((0, ))
    result = oc_sort.update(dets, confs, classes)
    assert result.shape == (0, 5)

def test_update_with_detections(oc_sort, mock_kalman_box_tracker):
    dets = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    confs = np.array([0.6, 0.7])
    classes = np.array([1, 2])
    
    result = oc_sort.update(dets, confs, classes)
    
    assert mock_kalman_box_tracker.call_count == 2  # Two new trackers should be initialized
    assert result.shape[1] == 6  # Format should include ID and class
    assert oc_sort.trackers  # Ensure trackers list is not empty after update

def test_predict_tracker(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    prediction = tracker.predict()
    assert prediction.shape == (1, 4)

def test_update_tracker(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    tracker.update(bbox, cls=2)
    assert tracker.cls == 2
    assert tracker.time_since_update == 0

def test_kalman_box_tracker_initialization():
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    assert tracker.id == 0
    assert tracker.cls == 1
    assert tracker.time_since_update == 0

def test_kalman_box_tracker_update(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    tracker.update(bbox, cls=2)
    assert tracker.cls == 2
    assert tracker.time_since_update == 0

def test_kalman_box_tracker_predict(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    prediction = tracker.predict()
    assert prediction.shape == (1, 4)

def test_update_trackers_with_high_confidence(oc_sort, mock_kalman_box_tracker):
    dets = np.array([[100, 100, 200, 200, 0.8], [150, 150, 250, 250, 0.9]])  # Ensure cX < ROI for left2right
    confs = np.array([0.8, 0.9])
    classes = np.array([1, 2])
    
    result = oc_sort.update(dets, confs, classes)
    
    assert len(oc_sort.trackers) == 2  # Expect two trackers to be initialized
    assert result.shape[1] == 6  # Ensure result includes [x1, y1, x2, y2, ID, class]
    assert oc_sort.frame_count == 1  # After update, frame count should be 1
    
def test_update_trackers_with_low_confidence(oc_sort, mock_kalman_box_tracker):
    dets = np.array([[10, 10, 20, 20, 0.4], [30, 30, 40, 40, 0.3]])
    confs = np.array([0.4, 0.3])
    classes = np.array([1, 2])
    
    result = oc_sort.update(dets, confs, classes)
    
    assert len(oc_sort.trackers) == 0  # No trackers should be initialized
    assert result.shape == (0, 5)  # Result should be empty

def test_update_trackers_with_mixed_confidence(oc_sort, mock_kalman_box_tracker):
    dets = np.array([[10, 10, 20, 20, 0.4], [30, 30, 40, 40, 0.8]])
    confs = np.array([0.4, 0.8])
    classes = np.array([1, 2])
    
    result = oc_sort.update(dets, confs, classes)
    
    assert len(oc_sort.trackers) == 1  # One tracker should be initialized for the high confidence detection
    assert result.shape[1] == 6  # Format should include ID and class

def test_kalman_filter_prediction(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    pred_bbox = tracker.predict()
    assert pred_bbox is not None
    assert pred_bbox.shape == (1, 4)

def test_kalman_filter_update(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    tracker.update(bbox, cls=1)
    assert tracker.time_since_update == 0
    assert tracker.hits == 1

def test_kalman_filter_get_state(mock_kalman_box_tracker):
    bbox = np.array([10, 10, 20, 20])
    tracker = KalmanBoxTracker(bbox=bbox, cls=1, id=0)
    state = tracker.get_state()
    assert state.shape == (1, 4)

