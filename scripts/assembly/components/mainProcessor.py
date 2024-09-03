import time, datetime, gc, traceback, sys, json, os, cv2
from assembly.model_utils.initialize import updateVariants, extractFrameVCO
import numpy as np
from collections import OrderedDict




# Main class that has the workflow        
class MainProcessor:
    def __init__(self, GP, loggerObj, analysisLogic, output_sender, outputprepObj, visualisor, save_dir, shared_queue, transmitter_manager, varientchangeServer):
        """
        Initialize the MainProcessor object.

        Args:
            GP: Global Parameters object that stores all the essential information.
        """
        self.GP = GP
        self.varient_change = False
        self.Varient_change_data = None
        self.loggerObj = loggerObj
        self.shared_queue = shared_queue 
        self.output_sender = output_sender
        self.analysisLogic = analysisLogic
        self.outputprepObj = outputprepObj
        self.visualisor = visualisor
        self.save_dir = save_dir
        # self.comm_server = comm_server
        os.makedirs(self.save_dir, exist_ok=True)
        self.transmitter_manager = transmitter_manager

        self.varientchangeServer = varientchangeServer

        self.st_s = time.time()



        self.running = True
        # self.imageSender = imageSender
    def stop(self):
        self.running = False



    def extract_frame_from_queue(self):
        """
        Extracts a frame from the shared queue and returns the image along with its metadata.
        
        Returns:
            dict: Dictionary containing the camera ID as the key, and a sub-dictionary 
                  with the image, group_id, and iterator as values.
        """
        if not self.shared_queue.empty():
            frame_info = self.shared_queue.get()
            img_master = OrderedDict()
            
            img_master[frame_info["beltId"]] = frame_info
            image = frame_info["image"]

            image = cv2.merge([image, image , image])
            # Convert to (H, W, C) format by adding a channel dimension
            # image = np.expand_dims(image, axis=-1)  # Shape becomes (2048, 3072, 1)

            # If needed, convert to (1, C, H, W) for batch processing
            # image = np.transpose(image, (2, 0, 1))  # Shape becomes (1, 1, 2048, 3072)
            frame_info["image"] = image
            
            



            return img_master
        
        return {}


    def convert_np_types(self,obj):
        if isinstance(obj, dict):
            return {k: self.convert_np_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_np_types(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def handle_variant_change(self):
        # print(f"[INFO] {datetime.datetime.now()} Received data for Variant change")
        # self.loggerObj.logger.info(f"Received data for Variant change")
        self.varient_change = False
        message = self.varientchangeServer.read()
        if message is not None: 
            print("coming here ")
            self.varient_change = True
            self.Varient_change_data = message

        return self.Varient_change_data, self.varient_change

    def varientChange(self):
        self.Varient_change_data, self.varient_change = self.handle_variant_change()
        if self.varient_change:
            self.transmitter_manager.stop_transmitters()
            self.loggerObj.logger.info(f"Stopping Transmitter before varient change")


            try:
                
                updateVariants(GP=self.GP, CONFIG_JSON=self.Varient_change_data, loggerObj=self.loggerObj)
                self.varient_change = False
                self.Varient_change_data = None
                print(f"[INFO] {datetime.datetime.now()} Variant Changed successfully")
                self.loggerObj.logger.info(f"Variant Changed successfully")
                self.transmitter_manager.start_transmitters()
                self.loggerObj.logger.info(f"Resuming Transmitter after varient change")
                # self.setup_interfaces()  # Reset interfaces on variant change
                
            except Exception as e:
                self.varient_change = False
                self.Varient_change_data = None
                traceback.print_exception(*sys.exc_info())
                print(f"[INFO] {datetime.datetime.now()} Varient Change Failed!!!")
                self.loggerObj.logger.info("Varient Change Failed!!!")
                print(f"[INFO] {datetime.datetime.now()} Error Occured while Changing variant")
                print(e)
                self.loggerObj.logger.error(f"Error Occured while Changing variant --- {e}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                print(f"[ERROR] {datetime.datetime.now()} {e}")
                self.loggerObj.logger.error("-----------------------------------")
                self.loggerObj.logger.error("An error occurred:\n%s", traceback_string)
                self.loggerObj.logger.error("-----------------------------------")

    def read_classes_file(self, classes_file):
        """
        Read classes from a text file.

        Args:
            classes_file (str): Path to the classes file.

        Returns:
            list: List of classes read from the file.
        """
        try:
            with open(classes_file, 'r') as f:
                classes = f.read().strip().split('\n')
            return classes
        except FileNotFoundError:
            print(f"Error: {classes_file} not found.")
            return []
        except IOError as e:
            print(f"Error reading {classes_file}: {e}")
            return []

    def analyse(self, input_data, camera_id):
        """
        Conducts inference on the given data from a specific camera.

        Args:
            input_data (dict): The data from the camera which includes the image.
            camera_id (str): The identifier for the camera.

        Returns:
            dict: The results from the analysis logic.
        """
        try:
            x = time.time()
            self.loggerObj.loop_logger.info(f'Analysis Started!!! {time.time()}')
            outputres, object_count = self.analysisLogic(GP=self.GP, input_data=input_data, camera_id=camera_id) 
            print("Time taken for single inference: ", time.time() - x, time.time())
            self.loggerObj.loop_logger.info(f"Time taken for single inference: {time.time() - x}, {time.time()}")
            return outputres, object_count 
        except ValueError as e:
            print(f"Error in analyse: {e}")
            raise
    
    
    def run(self):
        """
        Continuously processes incoming frames and manages exceptions.
        """
        st = time.time()
        # run_time = 300
        
        # Infinite loop to keep processing incoming frames
        while True: 
            # curr_time = time.time()
            # if curr_time - self.st_s > run_time:
            #     print("->time to stop<-")
            #     self.transmitter_manager.stop_transmitters()
            #     self.stop()
            #     break


            try:
                    # Check if there are no images in the queue every 5 seconds
                if int((time.time() - st)) == 5 and self.shared_queue is not None:
                    print(f"[INFO] {datetime.datetime.now()} No images present in Q!!!")
                    print("Active cameras: ",list(self.GP.interfaceObjs.keys()), [len(self.GP.interfaceObjs[key]) for key in list(self.GP.interfaceObjs.keys())])
                    self.loggerObj.loop_logger.info(f"No images present in Q!!!")
                    st = time.time()
                
                # Check for varinat change
                self.varientChange()
                
                # Extract frame
                st1 = time.time()

                img_master_dict = self.extract_frame_from_queue()
                extracted_batch_size = len(img_master_dict)
                if extracted_batch_size > 0:
                    print(f"Time taken for rading image from Q {time.time()-st1, {time.time()}}")
                    self.loggerObj.loop_logger.info(f"Time taken for reading image from Q {time.time()-st1, {time.time()}}")
               
                # If images are extracted, process them
                if extracted_batch_size>0: 
                    main_st = time.time()
                    st = time.time()
                    # Extracting data
 
#                    img_master_dict OrderedDict([(None, {'id': '142b6840-b1c2-4505-9a18-ccb809c3adbc', 'camera_ip': '192.168.69.131', 'image': array([[[13, 13, 13],
#         [14, 14, 14],
#         [10, 10, 10],





                    camera_id = list(img_master_dict.keys())[0]
                    
                    input_data = img_master_dict[camera_id]
                    
                    # Analysis
                    try:
                        output_res, object_count= self.analyse(input_data=input_data, camera_id=camera_id)
                        print("------------------------------------------------------------")
                        print("output_res:",output_res)
                        
                        
                        # cv2.imwrite(f'frame_with_roi.png', frame_with_roi)
                        print("------------------------------------------------------------")
                    except Exception as e:
                        traceback.print_exception(*sys.exc_info())
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        # Print traceback to a string
                        traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                        # Log the traceback string
                        print(f"[ERROR] {datetime.datetime.now()} {e}" )
                        self.loggerObj.logger.error("-----------------------------------")
                        self.loggerObj.logger.error("An error occurred:\n%s", traceback_string)
                        self.loggerObj.logger.error("-----------------------------------")
                        self.loggerObj.loop_logger.exception(f'Error Happened in the Main Analysis!!!')
                        print(f"[INFO] {datetime.datetime.now()} Error Happened in the Main Analysis!!! ")
                        sys.exit(1)

                    st2 = time.time()
                    
                    # output formation
                    main_result = self.outputprepObj.run(res=output_res, interfaceObj=self.GP.interfaceObjs[camera_id][int(input_data["iterator"])])
                    print(f"Time taken for output prep {time.time()-st2}, {time.time()}")
                    self.loggerObj.loop_logger.info(f"Time taken for output prep {time.time()-st2}, {time.time()}")
                    # timestamp = str(datetime.datetime.now()).split(".")[0].replace(" ", ",")
                    now = datetime.datetime.now()

# Format the timestamp to include milliseconds
                    timestamp = now.strftime('%Y-%m-%d,%H:%M:%S') + ':' + str(int(now.microsecond / 1000)).zfill(3)
                    main_result["timestamp"] = timestamp


                    image_name = f"{input_data['frame_count']}_{timestamp}_{input_data['groupId']}_{input_data['iterator']}.jpg"
                    
                    main_result["imagePath"] = image_name

              

                    print("------------------------------------------------------------")
                    print("main_result:",main_result)
                    print("------------------------------------------------------------")

                    st3 = time.time()
                    # Visualisation
                    frame_to_draw_roi = input_data["image"]
                    roi = output_res["roi"]
                    direction = output_res["direction"]

                    res_image = self.visualisor.draw(image=frame_to_draw_roi, results=main_result, object_count=object_count, roi=roi, direction=direction)
                    print(f"Time taken for Visualisation {time.time()-st3}, {time.time()}")
                    self.loggerObj.loop_logger.info(f"Time taken for Visualisation {time.time()-st3}, {time.time()}")

                    # Adding the timestamp
                    
                    # Final prep of output
                    main_result = self.outputprepObj.final_prep(result=main_result)

                    st4 = time.time()

            

                    

                    cv2.imwrite(os.path.join(self.save_dir, f"{input_data['frame_count']}_{timestamp}_{input_data['groupId']}_{input_data['iterator']}.jpg"), res_image)

                    print(f"Time taken for Writing image {time.time()-st4}, {time.time()}")
                    self.loggerObj.loop_logger.info(f"Time taken for Writing image {time.time()-st4}, {time.time()}")
                    
                    main_result_converted = self.convert_np_types(main_result)
                    
                    
                    
                    # print("checking the type of json", type(main_result))
                    st5 = time.time()
                    with open("OUTPUT_latest.json", "w") as outfile:
                        
                        json.dump(main_result_converted, outfile)
                    # Sending to backend


                    # if main_result_converted.get('result'):
                    #     # Write the output to a file
                    #     with open("OUTPUT_latest.json", "w") as outfile:
                    #         json.dump(main_result_converted, outfile)

                    #     # Sending to backend
                    #     self.output_sender.send(message=main_result_converted)
                    #     self.loggerObj.loop_logger.info(f"what is the message that i am sending {main_result_converted}")
                    # else:
                    #     self.loggerObj.loop_logger.info("Result is empty, not sending the message")



                    self.output_sender.send(message=main_result_converted)
                    self.loggerObj.loop_logger.info(f"what is the message that i am sending {main_result_converted}")

                    print(f"Time taken for RabbitMq pushing {time.time()-st5}, {time.time()}")
                    self.loggerObj.loop_logger.info(f"Time taken for RabbitMq pushing {time.time()-st5}, {time.time()}")

                    print("[INFO] Time taken for the whole process: From Getting frame to pushing res", time.time() - main_st, time.time())
                    self.loggerObj.loop_logger.info(f"[INFO] Time taken for the whole process: From Getting frame to pushing res { time.time() - main_st},{ time.time()}")
            




            except KeyboardInterrupt:
                print(f"[INFO] {datetime.datetime.now()} [INFO] KEYBORAD INTERAPT Called")
                gc.collect()
                print("\nEXITING...")
                break
            
            except Exception as e:
                self.loggerObj.logger.exception(f'Error at main while loop, {e}')
                print(f"\n [ERROR] {datetime.datetime.now()} Error at main while loop \n ")
                traceback.print_exception(*sys.exc_info())
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                print(f"[ERROR] {datetime.datetime.now()} {e}")
                self.loggerObj.logger.error("-----------------------------------")
                self.loggerObj.logger.error("An error occurred:\n%s", traceback_string)
                self.loggerObj.logger.error("-----------------------------------")

                
                self.stop()
                break

            

            
