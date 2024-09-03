import os
import numpy as np
import cv2

class VisualizeResults:
    """
    A utility class to visualize detection and classification results
    on images using bounding boxes and text annotations.
    """

    def __init__(self, uuid_class_map):
        """
        Initializes the VisualizeResults class with a UUID to class name map.
        
        Args:
        - uuid_class_map (dict): Dictionary mapping UUIDs to class names.
        """
        self.uuid_class_map = uuid_class_map
    
   
    def get_class_name(self, uuid):
        """
        Get the class name corresponding to the given class index.

        Args:
            class_index (int): The index of the class.

        Returns:
            str: The name of the class.
        """
        return self.uuid_class_map.get(uuid, "Unknown")

    def draw_box(self, frame, box, color=[0, 255, 0]):
        """
        Draws a bounding box on the given frame.
        
        Args:
        - frame (np.array): The image on which the box will be drawn.
        - box (tuple): Coordinates of the box in the format (x1, y1, x2, y2).
        - color (list, optional): Color of the bounding box. Defaults to green.

        Returns:
        - np.array: Image with the bounding box drawn.
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
        return frame
    
    def draw_roi(self, frame, roi, direction):
        """
        Draws the ROI on the given frame.

        Args:
            frame (np.array): The frame on which the ROI will be drawn.
            roi (int or tuple): The ROI coordinate or bounding box.
            direction (str): The direction of movement.

        Returns:
            np.array: The frame with the ROI drawn on it.
        """
        height, width = frame.shape[:2]
    
        if isinstance(roi, float):
            # Convert float roi to pixel coordinate
            if direction in ["up2down", "down2up"]:
                roi = int(roi * height)  # Scale by height for vertical line
            else:
                roi = int(roi * width)   # Scale by width for horizontal line

        if isinstance(roi, int):
            if direction in ["up2down", "down2up"]:
                cv2.line(frame, (0, roi), (frame.shape[1], roi), (0, 255, 0), 2)
            else:
                cv2.line(frame, (roi, 0), (roi, frame.shape[0]), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        return frame

    # Method to write the classname
    def write_name(self, frame, text, pt, color=[255, 255, 0]):
        """
        Writes text annotations on the given frame.
        
        Args:
        - frame (np.array): The image on which the text will be written.
        - text (str): Text to be written on the image.
        - pt (tuple): Position where the text will start.
        - color (list, optional): Color of the text. Defaults to yellow.

        Returns:
        - np.array: Image with the text written.
        """
        # writing on images
        cv2.putText(frame, text, (pt[0] - 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame

    def draw_result(self, image, result_info):
        """
        Draws the results on the image.
        
        Args:
        - image (np.array): The image on which the results will be drawn.
        - result_info (dict): Dictionary containing the results to be drawn.
        
        Returns:
        - np.array: Image with the results drawn.
        """
        for uuid, info in result_info.items():
            class_name = self.get_class_name(uuid)
            if class_name.endswith("absent"):
                continue
            
            color = [0, 255, 0] if info["pass"] else [0, 0, 255]
            for box in info["boxes"]:
                image = self.draw_box(frame=image, box=box, color=color)
                image = self.write_name(frame=image, text=class_name, pt=box[:2])

            for box in info["fail_boxes"]:
                image = self.draw_box(frame=image, box=box, color=[0, 0, 255])
                image = self.write_name(frame=image, text=f"{class_name}_absent", pt=box[:2])
        return image
    
    def print_object_count(self, image, count, position=(50, 50), color=[255, 255, 0]):
        cv2.putText(image, f"Object Count: {count}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return image
    
    # Method to resize the results image
    def resize_image(self, original_image, resize_ratio=4):
        """
        Resizes the image to a smaller size for display.
        
        Args:
        - original_image (np.array): The original image to be resized.
        - resize_ratio (int, optional): Ratio to resize the image. Defaults to 4.
        
        Returns:
        - np.array: The resized image.
        """
        height, width = original_image.shape[:2]
        # Calculate new dimensions: one-fourth of the original dimensions
        new_height, new_width = height // resize_ratio, width // resize_ratio
        # Resize the image
        resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def draw(self, image, results, object_count=None, roi=None, direction=None):
        """
        Draws the results on the image and prints the object count if available.
        
        Args:
        - image (np.array): The image on which the results will be drawn.
        - results (dict): The results to be drawn on the image.
        - object_count (int, optional): The object count to be printed. Defaults to None.
        
        Returns:
        - np.array: The image with the results and object count drawn.
        """
        for single_result in results["result"]:
            image = self.draw_result(image=image, result_info=single_result)
        
        if object_count is not None:
            image = self.print_object_count(image=image, count=object_count)
        
        if roi is not None and direction is not None:
            print(f"what is roi here {roi}")
            image = self.draw_roi(image, roi, direction)
        
        image = self.resize_image(original_image=image)
        return image