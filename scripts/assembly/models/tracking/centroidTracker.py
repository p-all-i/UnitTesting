
import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, ROI=550, maxDistance=340, movement_direction='left2right'):
        self.maxDistance = maxDistance
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.bboxes = {}
        self.prev_boxes = []
        self.direction = movement_direction
        self.roi = ROI
        print("coming to the centroidTracker", self.direction)

    def sorted_boxes(self, bboxes):
        if len(bboxes) == 0:
            return []
        direction_key_map = {
            'down2up': lambda bbox: bbox[1],
            'up2down': lambda bbox: -bbox[1],
            'left2right': lambda bbox: bbox[0],
            'right2left': lambda bbox: -bbox[0]
        }
        sorted_bboxes = sorted(bboxes, key=direction_key_map[self.direction])
        print("check direction in centroid", sorted_bboxes)
        return sorted_bboxes

    def register(self, centroid, bbox):
        direction_map = {
            'down2up': centroid[1] > self.roi - 50,
            'up2down': centroid[1] < self.roi + 50,
            'left2right': centroid[0] < self.roi + 50,
            'right2left': centroid[0] > self.roi - 50
        }
        if direction_map[self.direction]:
            self.objects[self.nextObjectID] = centroid
            self.bboxes[self.nextObjectID] = bbox
            self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]

    def centroid_creation(self, coords):
        inputCentroids = np.zeros((len(coords), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(coords):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        return inputCentroids

    def sort_bboxes_by_centroid(self, coords):
        def calculate_centroid(bbox):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            return (x_center, y_center)

        centroids = [calculate_centroid(bbox) for bbox in coords]
        print("what if it is centroids ", centroids)

        direction_key_map = {
            'down2up': lambda centroid: centroid[1],
            'up2down': lambda centroid: -centroid[1],
            'left2right': lambda centroid: centroid[0],
            'right2left': lambda centroid: -centroid[0]
        }


        sorted_data = sorted(zip(centroids, coords), key=lambda x: direction_key_map[self.direction](x[0]))
        sorted_centroids, sorted_bboxes = zip(*sorted_data)
        return list(sorted_centroids), list(sorted_bboxes)
    
    def match_check(self, new_centroid, old_centroid):
        if self.direction == "down2up": return new_centroid[1] < old_centroid[1]
        elif self.direction == "up2down": return new_centroid[1] > old_centroid[1]
        elif self.direction == "left2right": return new_centroid[0] > old_centroid[0]
        elif self.direction == "right2left": return new_centroid[1] > old_centroid[1]

    def match_centroids_modified(self, objectCentroids, inputCentroids, objectIDs):
        matches = []
        for i, new_centroid in enumerate(inputCentroids):
            print("Working with new box: ", i, new_centroid)
            closest_distance = np.inf
            closest_centroid_index = -1
            print(objectCentroids, objectIDs)
            for j, old_centroid in enumerate(objectCentroids):
                print("trying to maych with: ", j)
                matched_ids = [obj_id for obj_id, _ in matches]
                print("matched_ids: ", matched_ids)
                if objectIDs[j] in matched_ids:
                    continue
                
                # if new_centroid[1] < old_centroid[1]:
                if self.match_check(new_centroid, old_centroid):
                    direction_distance_map = {
                        'down2up': abs(new_centroid[1] - old_centroid[1]),
                        'up2down': abs(new_centroid[1] - old_centroid[1]),
                        'left2right': abs(new_centroid[0] - old_centroid[0]),
                        'right2left': abs(new_centroid[0] - old_centroid[0])
                    }
                    distance = direction_distance_map[self.direction]
                    print("distance: ", distance)

                    if distance < closest_distance and distance < self.maxDistance:
                        print("Condition satisfied: ", j, i)
                        closest_distance = distance
                        closest_centroid_index = j
                        break

            if closest_centroid_index != -1:
                objectID = objectIDs[closest_centroid_index]
                matches.append((objectID, i))

        return matches

    def check_same(self, input_boxes):
        if len(self.prev_boxes) != len(input_boxes):
            print("prev_boxessssss",len(self.prev_boxes))
            print("input_boxessssss",len(input_boxes))
            return False
        sorted_prev_boxes = self.sorted_boxes(bboxes=self.prev_boxes)
        current_boxes = self.sorted_boxes(bboxes=input_boxes)
        direction_diff_map = {
            'down2up': lambda prev, curr: abs(prev[1] - curr[1]),
            'up2down': lambda prev, curr: abs(prev[1] - curr[1]),
            'left2right': lambda prev, curr: abs(prev[0] - curr[0]),
            'right2left': lambda prev, curr: abs(prev[0] - curr[0])
        }
        for ind in range(len(sorted_prev_boxes)):
            if direction_diff_map[self.direction](sorted_prev_boxes[ind], current_boxes[ind]) > 10:
                return False
        return True

    # Change to dets : [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    def run(self, dets, confs, classes):
        same_status = self.check_same(input_boxes=dets)
        print("checking satus" , same_status)
        if same_status:
            self.objects = {}
            self.bboxes = {}
            self.prev_boxes = dets
            return self.bboxes

        if len(dets) != 0:
            inputCentroids, inputbboxes = self.sort_bboxes_by_centroid(dets)
            if len(self.objects) == 0:
                print("what is objects in centroid run", self.objects)
                for i, centroid in enumerate(inputCentroids):
                    self.register(centroid, inputbboxes[i])
                
                # return self.objects, self.bboxes
                return self.bboxes

            else:
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())
                matches = self.match_centroids_modified(objectCentroids, inputCentroids, objectIDs)
                unmatchedObjectIDs = set(objectIDs) - {o for o, _ in matches}
                unmatchedInputCentroids = set(range(len(inputCentroids))) - {i for _, i in matches}
                for objectID in unmatchedObjectIDs:
                    self.deregister(objectID)
                for i in unmatchedInputCentroids:
                    self.register(inputCentroids[i], inputbboxes[i])
                    # self.register(inputCentroids[i], inputbboxes[i])

                for match in matches:
                    self.objects[match[0]] = inputCentroids[match[1]]
                    self.bboxes[match[0]] = inputbboxes[match[1]]
                self.prev_boxes = dets
                print("is bonding box what i want ",self.bboxes)
                return self.bboxes
        else:
            self.objects = {}
            self.bboxes = {}
            self.prev_boxes = []
            print("is bonding box what i want ",self.bboxes)
            return self.bboxes
