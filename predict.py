from ultralytics import YOLO


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import easyocr




model = YOLO("models/yolov8n.pt")

classesNames = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

detect_class_list = [2,3,5,7]


def read_lp_text(path):
    license_plate_list = []
    image_path = cv2.imread(path)
    yolo_result = model(image_path)
    for r in yolo_result:
        detected_boxes = r.boxes

        for d_box in detected_boxes:
            d_box_conf = math.ceil((d_box.conf[0]*100))/100
            d_box_class = int(d_box.cls[0])

            if d_box_class in detect_class_list and d_box_conf >= 0.85:
                x, y, w, h = d_box.xyxy[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
            
                car_image = image_path[y:h, x:w]
                
                car_img_contiguous = np.ascontiguousarray(car_image)
                
                custom_lp_model = YOLO("models/best.pt")
                lp_result = custom_lp_model.predict(car_img_contiguous)

                
                lp_boxes = lp_result[0].boxes
                lp_box_conf_max = 0
                for box in lp_boxes:
                    if box.conf[0] >= lp_box_conf_max:
                        lp_box_conf_max = box.conf[0]
                        
                for lp_box in lp_boxes:
                    lp_box_conf = math.ceil((lp_box.conf[0]*100))/100
                    lp_box_conf_max = math.ceil((lp_box_conf_max*100))/100
                    if lp_box_conf == lp_box_conf_max and lp_box_conf >= 0.30:
                        x, y, w, h = lp_box.xyxy[0]
                        x, y, w, h= int(x), int(y), int(w), int(h)
                        license_plate_img = car_image[y:h, x:w]

                        gray_img = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
                        


                        blurred_img = cv2.GaussianBlur(gray_img,(1,1), cv2.BORDER_DEFAULT)
                        

                        canny = cv2.Canny(blurred_img, 125, 255)
                        
                        binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)
                        plt.imshow(binary_img) 
                        plt.axis('off') # Hide the axis 
                        plt.show()
                        number_reader = easyocr.Reader(['en']) 
                        license_num = number_reader.readtext(binary_img)
                        license_plate_list.append(license_num)
                   
                        
    cars_reg_details = []                          
    for lp_num in license_plate_list:
        if lp_num:
            car_reg_details =[]
            for bbox,num,prob in lp_num:
                cars_reg_details.append(num)
            
            cars_reg_details.append(car_reg_details)

    return cars_reg_details
  












