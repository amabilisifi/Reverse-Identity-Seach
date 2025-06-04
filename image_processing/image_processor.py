import cv2
import numpy as np
from mtcnn import MTCNN
import dlib
import os # For checking file paths

class ImagePreprocessor:
    def __init__(self, sr_model_path="EDSR_x4.pb", shape_predictor_path="shape_predictor_68_face_landmarks.dat"):
        # Initialize MTCNN for face detection
        self.face_detector = MTCNN()

        # Initialize dlib's frontal face detector and shape predictor for alignment
        self.dlib_face_detector = dlib.get_frontal_face_detector()
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor model not found at {shape_predictor_path}")
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

        # Initialize Super-Resolution model
        if not os.path.exists(sr_model_path):
            raise FileNotFoundError(f"Super-resolution model not found at {sr_model_path}")
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(sr_model_path)
        # Assuming EDSR x4 model, set model name and scale accordingly
        model_name = os.path.basename(sr_model_path).split('_')[0].lower()
        scale_factor = int(os.path.basename(sr_model_path).split('x')[1].split('.')[0]) if 'x' in os.path.basename(sr_model_path) else 4
        self.sr.setModel(model_name, scale_factor)

    def enhance_image_quality(self, image):
        # Denoise the image
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def apply_super_resolution(self, image):
        return self.sr.upsample(image)

    def detect_and_crop_face(self, image, margin_scale=0.3):
        faces = self.face_detector.detect_faces(image)
        if not faces:
            return None
        
        # Get coordinates of the largest face based on confidence 
        face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = face['box']
        
        # Add margin around the face
        margin_w = int(margin_scale * w)
        margin_h = int(margin_scale * h) 
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        face_crop = image[y1:y2, x1:x2]
        return face_crop

    def align_face(self, image_crop):
        gray_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        detected_faces_in_crop = self.dlib_face_detector(gray_crop, 1)
        
        if not detected_faces_in_crop:
            print("No face detected in the cropped image.")
            return image_crop 
            
        face_rect = detected_faces_in_crop[0]
        
        # Get facial landmarks
        landmarks = self.shape_predictor(image_crop, face_rect) # Use original color crop for landmarks
        
        # Points for eye centers:
        # Left eye: points 36-41, Right eye: points 42-47
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_eye_center = left_eye_pts.mean(axis=0).astype(int)
        right_eye_center = right_eye_pts.mean(axis=0).astype(int)
        
        # Calculate angle for alignment
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center of the image crop for rotation
        (h_crop, w_crop) = image_crop.shape[:2]
        center_crop = (w_crop // 2, h_crop // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center_crop, angle, 1.0)
        
        # Perform affine transformation (rotation)
        aligned_face = cv2.warpAffine(image_crop, rotation_matrix, (w_crop, h_crop),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return aligned_face

    def enhance_contrast_clahe(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # clipLimit 2.0 or 3.0
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr

    def preprocess(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Step 1: Initial quality enhancement (denoise, sharpen)
        enhanced_quality_image = self.enhance_image_quality(image)
        # cv2.imwrite("debug_01_enhanced_quality.jpg", enhanced_quality_image)

        # Step 2: Super-resolution
        upscaled_image = self.apply_super_resolution(enhanced_quality_image)
        # cv2.imwrite("debug_02_upscaled.jpg", upscaled_image)
        
        # Step 3: Face detection and cropping
        face_crop = self.detect_and_crop_face(upscaled_image)
        if face_crop is None:
            print("No face detected in the image.")
            return None
        # cv2.imwrite("debug_03_face_crop.jpg", face_crop)
            
        # Step 4: Face alignment
        aligned_face = self.align_face(face_crop)
        # cv2.imwrite("debug_04_aligned_face.jpg", aligned_face)
        
        # Step 5: Final contrast enhancement on the aligned face
        final_processed_face = self.enhance_contrast_clahe(aligned_face)
        # cv2.imwrite("debug_05_final_processed_face.jpg", final_processed_face)
        
        print("Image preprocessing complete.")
        return final_processed_face


if __name__ == '__main__':
    if not os.path.exists("EDSR_x4.pb"):
        with open("EDSR_x4.pb", "w") as f: f.write("dummy edsr model")
        print("Created dummy EDSR_x4.pb. Replace with actual model.")
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        with open("shape_predictor_68_face_landmarks.dat", "w") as f: f.write("dummy landmarks model")
        print("Created dummy shape_predictor_68_face_landmarks.dat. Replace with actual model.")

    # Create a dummy image for testing
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("dummy_input.jpg", dummy_image)
    print("Created dummy_input.jpg.")

    try:
        preprocessor = ImagePreprocessor(
            sr_model_path="EDSR_x4.pb", # Ensure this model is downloaded
            shape_predictor_path="shape_predictor_68_face_landmarks.dat" # Ensure this model is downloaded
        )
        # Replace "dummy_input.jpg" 
        processed_image = preprocessor.preprocess("dummy_input.jpg") 

        if processed_image is not None:
            cv2.imshow("Processed Face", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite("output_processed_face.jpg", processed_image)
        else:
            print("Image preprocessing failed.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure model files are correctly placed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")