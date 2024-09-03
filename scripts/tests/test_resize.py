import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import cv2
from PIL import Image
# from assembly.models.detection.Augmentation import ResizeShortestEdge, ResizeTransform  # Update import path as needed
from assembly.models.detection.resizing import ResizeShortestEdge, ResizeTransform  # Update import path as needed

@pytest.fixture
def resize_shortest_edge_instance():
    # Create an instance of ResizeShortestEdge with mock parameters
    return ResizeShortestEdge(short_edge_length=[400, 500], max_size=1000, sample_style="range", interp=Image.BILINEAR)


def test_resize_shortest_edge_initialization(resize_shortest_edge_instance):
    assert resize_shortest_edge_instance.short_edge_length == [400, 500]
    assert resize_shortest_edge_instance.max_size == 1000
    assert resize_shortest_edge_instance.is_range is True
    assert resize_shortest_edge_instance.interp == Image.BILINEAR


def test_resize_shortest_edge_get_transform(resize_shortest_edge_instance):
    dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)  # H, W, C
    transform = resize_shortest_edge_instance.get_transform(dummy_image)
    
    assert isinstance(transform, ResizeTransform)
    assert transform.new_h <= resize_shortest_edge_instance.max_size
    assert transform.new_w <= resize_shortest_edge_instance.max_size


@pytest.fixture
def resize_transform_instance():
    # Create an instance of ResizeTransform with mock parameters
    return ResizeTransform(h=600, w=800, new_h=400, new_w=500, interp=cv2.INTER_LINEAR)


def test_resize_transform_initialization(resize_transform_instance):
    assert resize_transform_instance.h == 600
    assert resize_transform_instance.w == 800
    assert resize_transform_instance.new_h == 400
    assert resize_transform_instance.new_w == 500
    assert resize_transform_instance.interp == cv2.INTER_LINEAR


def test_resize_transform_apply_image(resize_transform_instance):
    dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
    resized_image = resize_transform_instance.apply_image(dummy_image)

    assert resized_image.shape == (400, 500, 3)
    assert resized_image.dtype == np.uint8


def test_resize_transform_apply_coords(resize_transform_instance):
    dummy_coords = np.array([[100, 150], [200, 250]], dtype=np.float32)
    resized_coords = resize_transform_instance.apply_coords(dummy_coords)

    assert resized_coords.shape == dummy_coords.shape
    