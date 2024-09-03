import pytest
from unittest.mock import MagicMock
import torch, cv2, operator
import os, sys
import numpy as np
from collections import deque
# Import the TrackerInterface class to be tested
from assembly.interfaces.trackerInterface import TrackerInterface

def test_tracker_initialization():
    mock_tracker = MagicMock()
    roi = (50, 50, 150, 150)
    direction = "up2down"
    tracker_interface = TrackerInterface(tracker=mock_tracker, roi=roi, dir=direction)
    
    assert tracker_interface.tracker == mock_tracker
    assert tracker_interface.roi == roi
    assert tracker_interface.dir == direction
    assert tracker_interface.operator == operator.ge
    assert tracker_interface.object_count == 0
    assert isinstance(tracker_interface.queue, deque)

def test_update_method():
    # Create a mock tracker with a run method that returns a tuple (simulating the output format)
    mock_tracker = MagicMock()
    mock_tracker.run.return_value = ([(100, 100, 200, 200)],)  # Simulating a tuple returned

    # Initialize TrackerInterface with the mock tracker
    tracker_interface = TrackerInterface(tracker=mock_tracker, roi=50, dir="up2down")

    # Set up test inputs for detections, confidence scores, and classes
    dets = [(100, 100, 200, 200)]
    confs = [0.8]
    classes = [1]

    # Call the update method
    result = tracker_interface.update(dets=dets, confs=confs, classes=classes)
    
    # Verify that the tracker's run method was called with the correct parameters
    mock_tracker.run.assert_called_once_with(dets, confs, classes)

    # Verify that the result is correctly converted to a dictionary
    assert result == {0:[(100, 100, 200, 200)]}

def test_run_method():
    mock_tracker = MagicMock()
    mock_tracker.run.return_value = {0: (100, 100, 200, 200)}
    tracker_interface = TrackerInterface(tracker=mock_tracker, roi=150, dir="up2down")
    
    frame = MagicMock()
    dets = [(100, 100, 200, 200)]
    confs = [0.8]
    classes = [1]
    
    result, object_count = tracker_interface.run(frame, dets, confs, classes)
    assert result == {0: (100, 100, 200, 200)}
    assert object_count == 1

def test_checkRoiCrossed_method():
    tracker_interface = TrackerInterface(tracker=None, roi=150, dir="up2down")
    bbox = [100, 200, 200, 300]  # Should cross the ROI at y=150
    
    crossed = tracker_interface.checkRoiCrossed(bbox)
    assert crossed is True

def test_objAnalysed_method():
    tracker_interface = TrackerInterface(tracker=None, roi=150, dir="up2down")
    tracker_interface.queue.append(0)
    
    analyzed = tracker_interface.objAnalysed(0)
    not_analyzed = tracker_interface.objAnalysed(1)
    
    assert analyzed is True
    assert not_analyzed is False

def test_draw_roi_method():
    tracker_interface = TrackerInterface(tracker=None, roi=0.5, dir="up2down")
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    frame_with_roi = tracker_interface.draw_roi(frame)
    # We should visually inspect the frame or analyze the pixel values in a more advanced test
    assert frame_with_roi is not None

# Run the test
if __name__ == "__main__":
    pytest.main()
