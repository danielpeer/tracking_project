import pathlib
import tarfile

from imageai.Detection import ObjectDetection
import os
import cv2
from ObjetRecognition.DetectGender import *

current_dir = "D:\\Users\\97252\PycharmProjects\\tracking_project\\ObjetRecognition"
class ObjectDetector:
        def __init__(self):
            self.detector = ObjectDetection()
            self.detector.setModelTypeAsRetinaNet()
            self.detector.setModelTypeAsYOLOv3()
            self.detector.setModelPath(os.path.join(current_dir, "yolo.h5"))
            self.detector.loadModel()
            self.gender_detector = GenderDetector()

        def get_target_detect(self, input_image):
            filename = "framex.jpg"
            cv2.imwrite(os.path.join(current_dir, filename), input_image)
            detections = self.detector.detectObjectsFromImage(os.path.join(current_dir, filename), output_image_path=os.path.join(current_dir , "imagenew.jpg"),minimum_percentage_probability=5)
            for eachObject in detections:
                if eachObject["name"] == 'person':
                    dims = eachObject['box_points']
                    current_img = input_image[dims[1]:dims[3], dims[0]:dims[2]]
                    return self.gender_detector.get_predict(current_img)

        def get_targets_on_the_sides(self, input_image):
            targets = []
            filename = "framex.jpg"
            cv2.imwrite(os.path.join(current_dir, filename), input_image)
            detections = self.detector.detectObjectsFromImage(os.path.join(current_dir, filename), output_image_path=os.path.join(current_dir , "imagenew.jpg"))
            for eachObject in detections:
                if eachObject["name"] == 'person':
                    dims = eachObject['box_points']
                    targets.append(dims)
            return targets
