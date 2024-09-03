import os, sys, time, datetime
import numpy as np
from collections import OrderedDict
from assembly.models.tracking.centroidTracker import CentroidTracker
from assembly.models.tracking.ocSort_tracker import ocTracker


# Class that is responsible for loading the Tracker objects
class trackerManager:
    """
    This class manages the loading of tracking objects based on their unique identifiers
    (camera IDs in this case).

    Attributes:
        loggerObj (object): An object for logging various steps and errors.
    """
    def __init__(self, loggerObj):
        """
        Initializes the trackerManager class.

        Args:
            loggerObj (object): The logging object.
        """
        self.loggerObj = loggerObj
        self.loggerObj.logger.info("[INFO] trackerManager started")
        print(f"[INFO] {datetime.datetime.now()} trackerManager started")

    # Method to load the trackers
    def load_trackers(self, trackerDict, active_trackers):
        """
        Load the tracker objects specified in the active_trackers list and populate the trackerDict.

        Args:
            trackerDict (dict): The dictionary to populate with loaded trackers, indexed by camera IDs.
            active_trackers (list): A list of camera IDs specifying which trackers should be active.

        Returns:
            dict: The updated trackerDict containing the loaded tracker objects.
        """
        # Iterating through the model lists in GP
        # Trackertypes: ocsort, centroid 
        for camera_id, trackerinfo in active_trackers.items():
            if camera_id in list(trackerDict.keys()):
                continue
            if trackerinfo["type"] == "centroid":
                trackerDict[camera_id] = CentroidTracker(maxDistance=trackerinfo["maxDistance"], ROI=trackerinfo["line"], movement_direction=trackerinfo["direction"])
                self.loggerObj.logger.info(f"Tracker object of type CENTROID TRACKER created for {camera_id}")
                print(f"[INFO] {datetime.datetime.now()} Tracker object created for {camera_id}")
            elif trackerinfo["type"] == "ocsort":
                print("is line good?", trackerinfo["line"])
                trackerDict[camera_id] = ocTracker(movement_direction=trackerinfo["direction"], ROI=trackerinfo["line"])
                self.loggerObj.logger.info(f"Tracker object of type OCSORT created for {camera_id}")
                print(f"[INFO] {datetime.datetime.now()} Tracker object created for {camera_id}")
            else:
                self.loggerObj.logger.exception("[INFO] Tried loading a Tracker that is not Implemented")
                print(f"[INFO] {datetime.datetime.now()} Tried loading a Tracker that is not Implemented")
                raise Exception("[INFO] Tracker not implemented")

        return trackerDict
    
