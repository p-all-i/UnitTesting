from assembly.interfaces.assemblyInterface import AssemblyInterface
from collections import OrderedDict
from assembly.interfaces.interface import Interface
import datetime

    
# Class that create different Interfaces Depending on the Task 
class InterfaceCreation:
    """
    This class defines a static method for creating Interface objects for multiple cameras/belts.
    
    Note:
        - This class is designed to work in conjunction with a Global Parameters.
        - The 'GP' object contains all the necessary parameters and dictionaries
          for creating the Interface objects.
    """
    
    @staticmethod
    def create(GP, loggerObj):
        """
        Static method to create Interface objects and store them in the GP object.
        
        Args:
            GP (object): The object containing all necessary parameters and dictionaries.
                         - GP.ModelDict: Dictionary of pre-loaded models.
                         - GP.TrackerDict: Dictionary of pre-loaded trackers.
                         - GP.cameraParams: Dictionary containing camera parameters.

        Returns:
            object: The modified GP object containing the created Interface objects.
        """  
        # Retrieve required dictionaries and parameters from the GP object
        ModelDict = GP.ModelDict
        TrackerDict = GP.TrackerDict
        cameraParams = GP.cameraParams
        print(f"camera param is {cameraParams}")
        # Initialize an ordered dictionary to store the Interface objects
        GP.interfaceObjs = OrderedDict()

        # Loop through each camera and its parameters to create Interface objects
        for camera_id, camera_params in cameraParams.items():
            GP.interfaceObjs[camera_id] = list()
            for camera_param in camera_params:
                # Create an Interface object for each set of camera parameters
                interfaceObj = Interface(group_info=camera_param, ModelDict=ModelDict, TrackerDict=TrackerDict, camera_id=camera_id)
                # Append the created Interface object to the list for the current camera
                GP.interfaceObjs[camera_id].append(interfaceObj)
                print(f"[INFO] {datetime.datetime.now()} Created a Interface for Camera id: {camera_id} & image id: {len(GP.interfaceObjs[camera_id])-1}")
                loggerObj.logger.info(f"Created a Interface for Camera id: {camera_id} & image id: {len(GP.interfaceObjs[camera_id])-1}")
        # Return the modified GP object containing the created Interface objects
        return GP



if __name__ == "__main__":
    sample_grp_info = {"camera1":[{
           "steps": [
            "tracker",
            "cropping"
           ],
           "tracker": {
            "model_id": "1234-5678-9012",
             "roi": {
               "direction": "up2down", # hard code meta data
               "coordinate": 50,
             },
           },
           "cropping": {
             "123-345-678": { # Roi ID
               "roi": [0, 0, 1000, 1000], # should be normalized to the annotation crop
               "model_1": [
                 {
                  "type": "classification/detection",
                  "model_id": "12-34-56-78",
                  "threshold": {"score_thresh":0.75}
                 },
               ],
               "model_2": [
                  {
                   "type": "classification",
                   "model_id": "12-34-56-78",
                   "threshold": {"score_thresh":0.75},
                    "class_name": "gear1"
                  },
                 {
                  "type": "classification",
                  "model_id": "12-34-56-78",
                  "threshold": {"score_thresh":0.75},
                   "class_name": "gear2"
                 },
                ],
             }
           },
      }, 
      {
           "steps": [
            "tracker",
            "cropping"
           ],
           "tracker": {
            "model_id": "1234-5678-9012",
             "roi": {
               "direction": "up2down", # hard code meta data
               "coordinate": 50,
             },
           },
           "cropping": {
             "123-345-678": { # Roi ID
               "roi": [0, 0, 1000, 1000], # should be normalized to the annotation crop
               "model_1": [
                 {
                  "type": "classification/detection",
                  "model_id": "12-34-56-78",
                  "threshold": {"score_thresh":0.75}
                 },
               ],
               "model_2": [
                  {
                   "type": "classification",
                   "model_id": "12-34-56-78",
                   "threshold": {"score_thresh":0.75},
                    "class_name": "gear1"
                  },
                 {
                  "type": "classification",
                  "model_id": "12-34-56-78",
                  "threshold": {"score_thresh":0.75},
                   "class_name": "gear2"
                 },
                ],
             }
           },
      }]}