import cv2, datetime, traceback, sys, time 

class AnalysisLogic:
    """
    A class to perform image analysis using the necessary interface objects
    """

    def __init__(self, loggerObj):
        """
        Initializes the AnalysisLogic with a logger object.
        
        Args:
        - loggerObj (object): Logger object to log information and errors.
        """
        self.loggerObj = loggerObj

    def __call__(self, GP, input_data, camera_id):
        """
        Invokes the AnalysisLogic to perform image analysis.
        
        Args:
        - GP (object): The Global parameters object which contains interface objects 
                       for running the analysis.
        - input_data (dict): Dictionary containing image data and other related information.
        - camera_id (int/str): ID representing the camera from which the image originates.
        
        Returns:
        - dict: Dictionary containing analysis results, camera ID, group ID, and image ID.
        """
        frame_with_roi = input_data["image"].copy()  # Initialize frame with the original image
        object_count = None
        roi = None
        direction = None 
        try: 
            # Inferencing the image using the appropriate interface object             
            
            
            res_dict, object_count = GP.interfaceObjs[camera_id][int(input_data["iterator"])].run(image=input_data["image"]) 
            #interface.run 
            

            self.loggerObj.loop_logger.info("[INFO] Image Analysis Done again!")
            print(f"[INFO] {datetime.datetime.now()} Image Analysis Done again!")
            roi = res_dict.get("roi", None)
            direction = res_dict.get("direction", None)
            # Adding cameraId, groupId, and imageId to the results
            # Ensure res_dict is a dictionary
            if not isinstance(res_dict, dict):
                res_dict = {}
            res_dict["roi"] = roi
            res_dict["direction"] = direction
            res_dict["cameraId"] = input_data["beltId"]#camera_id
            res_dict["groupId"] = input_data["groupId"]
            res_dict["iterator"] = input_data["iterator"]
            res_dict["configId"] = input_data["configId"] 
            res_dict["groupLimit"] = input_data["groupLimit"]
            res_dict["extraInfo"] = input_data["extraInfo"]
            print("Analysis: ", input_data["groupId"], input_data["iterator"])
        except Exception as e:
            self.loggerObj.loop_logger.error(f"[INFO] Error in Image Analysis again! {e}")
            print(f"[ERROR] {datetime.datetime.now()} Error in Image Analysis {e}")
            traceback.print_exception(*sys.exc_info())
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # Print traceback to a string
            traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            # Log the traceback string
            print(f"[ERROR] {datetime.datetime.now()} {e}" )
            print("Reason: ",traceback_string)
            self.loggerObj.loop_logger.error("-----------------------------------")
            self.loggerObj.loop_logger.error("An error occurred:\n%s", traceback_string)
            self.loggerObj.loop_logger.error("-----------------------------------")
            exit(1)
    
        return res_dict, object_count

        
