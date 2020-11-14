from imageai.Detection import ObjectDetection
import os
import cv2
from ObjetRecognition.DetectGender import *

current_dir = "D:\\Users\\97252\PycharmProjects\\tracking_project\\ObjetRecognition"
class ObjectDetector:
        def __init__(self):
            self.detector = ObjectDetection()
            self.detector.setModelTypeAsRetinaNet()
            self.detector.setModelPath(os.path.join(current_dir, "resnet50_coco_best_v2.0.1.h5"))
            self.detector.loadModel()
            self.gender_detector = GenderDetector()

        def update_detection(self, input_image):
            filename = "framex.jpg"
            cv2.imwrite(os.path.join(current_dir, filename), input_image)
            detections = self.detector.detectObjectsFromImage(os.path.join(current_dir, filename), output_image_path=os.path.join(current_dir , "imagenew.jpg"))
            for eachObject in detections:
                if eachObject["name"] == 'person':
                    dims = eachObject['box_points']
                    current_img = input_image[dims[1]:dims[3], dims[0]:dims[2]]
                    return self.gender_detector.get_predict(current_img)




