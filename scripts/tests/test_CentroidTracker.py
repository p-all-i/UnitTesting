import pytest
import numpy as np
from collections import OrderedDict
from assembly.models.tracking.centroidTracker import CentroidTracker

@pytest.fixture
def centroid_tracker():
    return CentroidTracker(ROI=100, maxDistance=50, movement_direction='left2right')

def test_initialization(centroid_tracker):
    assert centroid_tracker.roi == 100
    assert centroid_tracker.maxDistance == 50
    assert centroid_tracker.direction == 'left2right'
    assert centroid_tracker.objects == OrderedDict()
    assert centroid_tracker.bboxes == {}
    assert centroid_tracker.prev_boxes == []

def test_register(centroid_tracker):
    centroid = (10, 10)
    bbox = (5, 5, 15, 15)
    centroid_tracker.register(centroid, bbox)
    
    assert len(centroid_tracker.objects) == 1
    assert len(centroid_tracker.bboxes) == 1
    assert centroid_tracker.objects[0] == centroid
    assert centroid_tracker.bboxes[0] == bbox

def test_deregister(centroid_tracker):
    centroid = (10, 10)
    bbox = (5, 5, 15, 15)
    centroid_tracker.register(centroid, bbox)
    assert len(centroid_tracker.objects) == 1
    
    centroid_tracker.deregister(0)
    assert len(centroid_tracker.objects) == 0
    assert len(centroid_tracker.bboxes) == 0

def test_centroid_creation(centroid_tracker):
    coords = [(5, 5, 15, 15), (20, 20, 30, 30)]
    expected_centroids = np.array([[10, 10], [25, 25]])
    
    centroids = centroid_tracker.centroid_creation(coords)
    np.testing.assert_array_equal(centroids, expected_centroids)

def test_sorted_boxes(centroid_tracker):
    bboxes = [(20, 10, 30, 20), (5, 5, 15, 15)]
    sorted_bboxes = centroid_tracker.sorted_boxes(bboxes)
    expected_bboxes = [(5, 5, 15, 15), (20, 10, 30, 20)]
    
    assert sorted_bboxes == expected_bboxes

def test_run_register(centroid_tracker):
    dets = [(5, 5, 15, 15), (20, 20, 30, 30)]
    confs = [0.9, 0.8]
    classes = [1, 1]
    result = centroid_tracker.run(dets, confs, classes)
    
    assert len(result) == 2
    assert 0 in result
    assert 1 in result

def test_run_deregister(centroid_tracker):
    # First, register some objects
    dets = [(5, 5, 15, 15), (20, 20, 30, 30)]
    confs = [0.9, 0.8]
    classes = [1, 1]
    centroid_tracker.run(dets, confs, classes)
    
    # Now, pass an empty detection list to deregister them
    result = centroid_tracker.run([], [], [])
    
    assert len(result) == 0
    assert centroid_tracker.objects == {}
    assert centroid_tracker.bboxes == {}
    assert centroid_tracker.prev_boxes == []

def test_run_match_and_update(centroid_tracker):
    dets = [(5, 5, 15, 15), (20, 20, 30, 30)]
    confs = [0.9, 0.8]
    classes = [1, 1]
    centroid_tracker.run(dets, confs, classes)
    
    # Now update with new detections
    new_dets = [(7, 7, 17, 17), (22, 22, 32, 32)]
    result = centroid_tracker.run(new_dets, confs, classes)
    
    assert len(result) == 2
    assert 0 in result
    assert 1 in result
    assert result[0] == (7, 7, 17, 17)
    assert result[1] == (22, 22, 32, 32)

