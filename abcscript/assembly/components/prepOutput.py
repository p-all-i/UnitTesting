import os, cv2
import numpy as np


class OutputPrep:

    def get_GT(self, roi_id, interfaceObj):
        GT = interfaceObj.ground_truth[roi_id]
        return GT
    #{'ground_truth': {'inlet_pipe': 
        # {'count': 1, 'operator': 0}, 'GF6': {'count': 1, 'operator': 0}, 
        # 'pin': {'count': 1, 'operator': 0}}
        #

        # self.ground_truth[roi_id] = interface_info["ground_truth"]

    def compare_count(self, actual_count, predicted_count, operator):
        
        if operator==0:
            return actual_count==predicted_count
        elif operator==1:
            return actual_count>predicted_count
        elif operator==2:
            return actual_count<predicted_count


    def prep_roi(self, res_obj, GT, final_res):
        # print("[DEBUG] res_obj:", res_obj) 
        # print("[DEBUG] GT:", GT)
        if len(res_obj["detection"])!=0:
            for class_name, pred_res in res_obj["detection"].items():
                print("[DEBUG] Processing class_name:", class_name)
                count = 0
                boxes = []
                fail_boxes = []
                for pred in pred_res:
                    if pred.get("class_name", None) is None:
                        count+=1
                        boxes.append(pred["box"])
                    else:
                        if  pred["class_name"].split("_")[-1]=="positive":
                                count+=1
                                boxes.append(pred["box"])
                        else:
                            fail_boxes.append(pred["box"])
                
                # checking if the count matches with GT
                print("[DEBUG] GT[class_name]:", GT.get(class_name)) 
                passed = self.compare_count(actual_count=GT[class_name]["count"], predicted_count=count, operator=GT[class_name]["operator"])

                # adding the info to final_res
                if class_name in list(final_res.keys()):
                    final_res[class_name]["count"]+=count
                    final_res[class_name]["pass"] = passed if final_res[class_name]["pass"] else False
                    final_res[class_name]["boxes"].extend(boxes)
                    final_res[class_name]["fail_boxes"].extend(fail_boxes)
                else:
                    final_res[class_name] = dict()
                    final_res[class_name]["count"] = count
                    final_res[class_name]["pass"] = passed 
                    final_res[class_name]["boxes"] = boxes
                    final_res[class_name]["fail_boxes"] = fail_boxes
        
        for class_name, gt_info in GT.items():
            if class_name not in list(final_res.keys()):
                passed = self.compare_count(actual_count=gt_info["count"], predicted_count=0, operator=gt_info["operator"])
                final_res[class_name] = dict()
                final_res[class_name]["count"] = 0
                final_res[class_name]["pass"] = passed
                final_res[class_name]["boxes"] = []
                final_res[class_name]["fail_boxes"] = []

        if len(res_obj["classification"])!=0:
            if res_obj["classification"]["class_name"].split("_")[-1]=="positive":
                count=1
                passed = True
                class_name = res_obj["classification"]["class_name"].split("_")[0]  
                boxes = [res_obj["classification"]["box"]]
                fail_boxes = []
            else:
                count=0
                passed = False
                class_name = res_obj["classification"]["class_name"].split("_")[0]  
                boxes = []
                fail_boxes = [res_obj["classification"]["box"]]

            # adding the info to final_res
                if class_name in list(final_res.keys()):
                    final_res[class_name]["count"]+=count
                    final_res[class_name]["pass"] = passed if final_res[class_name]["pass"] else False
                    final_res[class_name]["boxes"].extend(boxes)
                    final_res[class_name]["fail_boxes"].extend(fail_boxes)
                else:
                    final_res[class_name] = dict()
                    final_res[class_name]["count"] = count
                    final_res[class_name]["pass"] = passed 
                    final_res[class_name]["boxes"] = boxes
                    final_res[class_name]["fail_boxes"] = fail_boxes

        return final_res

        

    def run(self, res, interfaceObj):
        final_res = [{}]*len(res["cropping"])
        for i, result_obj in enumerate(res["cropping"]):
            for roi_id, roi_result in result_obj.items():
                GT = self.get_GT(roi_id=roi_id, interfaceObj=interfaceObj)
                final_res[i] = self.prep_roi(res_obj=roi_result, GT=GT, final_res=final_res[i])


                send_path = os.getenv("SAVE_DIR")
                post_image = os.getenv("POST_IMAGE")


                posting = True
                if post_image == "1":
                    posting = True
                if post_image == "0":
                    posting = False


        main_res = {"tracker":res["tracker"],"isPathUsed": posting , "result":final_res, "cameraId": res["cameraId"],"configId": res["configId"], "groupId": res["groupId"], "iterator":  res["iterator"], "groupLimit":res["groupLimit"], "extraInfo": res["extraInfo"]}
        return main_res
    
    
    # def final_prep(self, result):
    #     new_result = []
    #     print("post result in prepoutput", result)
    #     for res in result["result"]:
    #         pass_lt = []
    #         for class_name in res.keys():
    #             pass_lt.append(res[class_name]["pass"])
    #             if res[class_name]["boxes"]:
    #                 del res[class_name]["boxes"]
    #             if res[class_name]["fail_boxes"]:
    #                 del res[class_name]["fail_boxes"]
    #         new_res = {"pass":all(pass_lt),
    #                    "info": res}
    #         new_result.append(new_res)
    #     result["result"] = new_result
    #     print("what is result after del", result)
    #     return result
    



    def final_prep(self, result):
        new_result = []
        for res in result["result"]:
            pass_lt = []
            new_res_info = {}
            for class_name, class_info in res.items():
                pass_lt.append(class_info["pass"])
                new_class_info = {k: v for k, v in class_info.items() if k not in ["boxes", "fail_boxes"]}
                new_res_info[class_name] = new_class_info
            new_res = {"pass": all(pass_lt), "info": new_res_info}
            new_result.append(new_res)
        result["result"] = new_result
        return result

