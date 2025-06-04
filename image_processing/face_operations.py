import cv2
import numpy as np
from mtcnn import MTCNN
import dlib
import os

class FaceOperations:
    def __init__(self, shape_predictor_path):
        self.mtcnn_detector = MTCNN()
        self.dlib_face_detector = dlib.get_frontal_face_detector()
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor model not found at {shape_predictor_path}")
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect_and_crop_largest_face(self, image, margin_scale=0.3):
        """Detect the largest face using MTCNN and crop it."""
        faces = self.mtcnn_detector.detect_faces(image)
        if not faces:
            return None
        
        face_data = max(faces, key=lambda x: x['confidence']) # MTCNN specific
        x, y, w, h = face_data['box']
        
        margin_w = int(margin_scale * w)
        margin_h = int(margin_scale * h)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        return image[y1:y2, x1:x2]

    def align_face(self, image_crop):
        """Align a cropped face image."""
        gray_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        detected_faces_in_crop = self.dlib_face_detector(gray_crop, 1)
        
        if not detected_faces_in_crop:
            return image_crop
            
        face_rect = detected_faces_in_crop[0]
        landmarks = self.shape_predictor(image_crop, face_rect)
        
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_eye_center = left_eye_pts.mean(axis=0).astype(int)
        right_eye_center = right_eye_pts.mean(axis=0).astype(int)
        
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        (h_crop, w_crop) = image_crop.shape[:2]
        center_crop = (w_crop // 2, h_crop // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center_crop, angle, 1.0)
        aligned_face = cv2.warpAffine(image_crop, rotation_matrix, (w_crop, h_crop),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return aligned_face