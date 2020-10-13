import os
import cv2
import glob
import lib.face_alignment as recognition_model

from lib.FaceDetector import FaceDetector
from lib.FaceLandmarks import LandmarkDetector
from lib.EmotionRecognition import EmotionRecognition


class EmoPipeline:
    def __init__(self, models = './models/'):
        self.models = models
        self.face_detector = FaceDetector('./models/FD/'+ 'RFB_320')
        self.landmark_detector = LandmarkDetector('./models/FL/'+ 'FL')
        self.emotion_recognition = EmotionRecognition('./models/EMO/' + 'EMO')

    def run(self, image):
        pred_scores, bbox = self.face_detector.detect(image, 0.9)
        landmarks, quality = self.landmark_detector.process_bboxes(image, bbox)
        faces = recognition_model.get_aligned_crops(image, landmarks)
        return {'Face_' + str(i): self.emotion_recognition.run(face) for i, face in enumerate(faces) if face is not None}


