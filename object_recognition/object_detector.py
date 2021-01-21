import pathlib
import tarfile

from imageai.Detection import ObjectDetection
import os
import cv2
from object_recognition.gender_detector import *

class ObjectDetector:
        def __init__(self):
            self.detector = ObjectDetection()
            self.detector.setModelTypeAsRetinaNet()
            self.detector.setModelTypeAsYOLOv3()
            self.detector.setModelPath(os.path.join(os.getcwd(), "..\\object_recognition\\yolo.h5"))
            self.detector.loadModel()
            self.gender_detector = GenderDetector()

        def get_target_detect(self, input_image):
            cv2.imwrite(".\\framex.jpg", input_image)
            detections = self.detector.detectObjectsFromImage(".\\framex.jpg", output_image_path= ".\\imagenew.jpg",minimum_percentage_probability=5)
            for eachObject in detections:
                if eachObject["name"] == 'person':
                    dims = eachObject['box_points']
                    current_img = input_image[dims[1]:dims[3], dims[0]:dims[2]]
                    return self.gender_detector.get_predict(current_img)

        def get_targets_on_the_sides(self, input_image):
            targets = []
            cv2.imwrite(".\\framex.jpg", input_image)
            detections = self.detector.detectObjectsFromImage(".\\framex.jpg", output_image_path=".\\imagenew.jpg")
            for eachObject in detections:
                if eachObject["name"] == 'person':
                    dims = eachObject['box_points']
                    targets.append(dims)
            return targets
