import pytest
from unittest.mock import MagicMock
import numpy as np
import os
from assembly.components.prepOutput import OutputPrep

@pytest.fixture
def setup_output_prep():
    """
    Fixture to set up OutputPrep with mocked dependencies.
    """
    # Initialize the OutputPrep instance
    output_prep = OutputPrep()

    # Mock the interfaceObj
    interfaceObj_mock = MagicMock()
    interfaceObj_mock.ground_truth = {
        'roi_id_1': {
            'class_1': {'count': 1, 'operator': 0},
            'class_2': {'count': 2, 'operator': 1}
        }
    }

    return output_prep, interfaceObj_mock

def test_get_GT(setup_output_prep):
    """
    Test the get_GT method of OutputPrep.
    """
    output_prep, interfaceObj_mock = setup_output_prep

    # Call the get_GT method
    ground_truth = output_prep.get_GT('roi_id_1', interfaceObj_mock)

    # Verify the ground truth is returned correctly
    assert ground_truth == interfaceObj_mock.ground_truth['roi_id_1']

def test_compare_count(setup_output_prep):
    """
    Test the compare_count method of OutputPrep.
    """
    output_prep, _ = setup_output_prep

    # Test equality operator
    assert output_prep.compare_count(1, 1, 0) == True
    assert output_prep.compare_count(1, 2, 0) == False

    # Test greater-than operator
    assert output_prep.compare_count(2, 1, 1) == True
    assert output_prep.compare_count(1, 2, 1) == False

    # Test less-than operator
    assert output_prep.compare_count(1, 2, 2) == True
    assert output_prep.compare_count(2, 1, 2) == False

def test_prep_roi(setup_output_prep):
    """
    Test the prep_roi method of OutputPrep.
    """
    output_prep, interfaceObj_mock = setup_output_prep

    res_obj = {
        "detection": {
            "class_1": [{"box": [0, 0, 50, 50]}],
            "class_2": [{"box": [60, 60, 100, 100], "class_name": "class_2_negative"}]
        },
        "classification": {}
    }

    GT = interfaceObj_mock.ground_truth['roi_id_1']
    final_res = {}

    # Call the prep_roi method
    updated_final_res = output_prep.prep_roi(res_obj=res_obj, GT=GT, final_res=final_res)

    # Check that the updated final results are correct
    assert updated_final_res['class_1']['count'] == 1
    assert updated_final_res['class_1']['pass'] == True
    assert updated_final_res['class_2']['count'] == 0
    assert updated_final_res['class_2']['pass'] == True

def test_run(setup_output_prep):
    """
    Test the run method of OutputPrep.
    """
    output_prep, interfaceObj_mock = setup_output_prep

    res = {
        "tracker": {},
        "cropping": [
            {"roi_id_1": {"detection": {}, "classification": {}}}
        ],
        "cameraId": "camera_1",
        "configId": "config_1",
        "groupId": "group_1",
        "iterator": 0,
        "groupLimit": 10,
        "extraInfo": {}
    }

    # Mock environment variables for SAVE_DIR and POST_IMAGE
    os.environ["SAVE_DIR"] = "mock_save_dir"
    os.environ["POST_IMAGE"] = "1"

    # Call the run method
    main_res = output_prep.run(res=res, interfaceObj=interfaceObj_mock)

    # Verify the structure of main_res
    assert "tracker" in main_res
    assert "isPathUsed" in main_res
    assert "result" in main_res
    assert "cameraId" in main_res
    assert main_res["cameraId"] == "camera_1"
    assert main_res["isPathUsed"] == True

def test_final_prep(setup_output_prep):
    """
    Test the final_prep method of OutputPrep.
    """
    output_prep, _ = setup_output_prep

    result = {
        "result": [
            {
                "class_1": {"count": 1, "pass": True, "boxes": [], "fail_boxes": []},
                "class_2": {"count": 0, "pass": True, "boxes": [], "fail_boxes": []}
            }
        ]
    }

    # Call the final_prep method
    new_result = output_prep.final_prep(result=result)

    # Check that boxes and fail_boxes have been removed
    assert "boxes" not in new_result["result"][0]["info"]["class_1"]
    assert "fail_boxes" not in new_result["result"][0]["info"]["class_1"]
    assert new_result["result"][0]["pass"] == True

# Add more test functions for other methods if needed
