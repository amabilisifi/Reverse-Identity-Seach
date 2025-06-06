import cv2
import numpy as np
from typing import Union, Tuple, Optional, Dict, List
import logging
import concurrent.futures
import uuid

class BrightnessAdjuster:
    def __init__(self, 
                 target_brightness_range: Optional[Tuple[int, int]] = None,
                 brightness_thresholds: Optional[Dict[str, float]] = None):
        self.logger = logging.getLogger(__name__)
        
        # Target brightness range for face recognition (8-bit images)
        self.target_brightness_range = target_brightness_range or (110, 140)
        
        # Configurable brightness category thresholds
        self.brightness_thresholds = brightness_thresholds or {
            'very_dark': 60, 'dark': 90, 'slightly_dark': 120,
            'optimal': 180, 'bright': 220
        }
        
        self.linear_params = {'alpha': 1.0, 'beta': 0}
        self.gamma_params = {'gamma': 1.0, 'gain': 1.0}
        self.hist_params = {'target_mean': sum(self.target_brightness_range) / 2, 
                           'preserve_range': True}
        
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            self.logger.warning("Failed to load Haar cascade classifier for face detection")
    
    def preprocess_low_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast stretching if the image has low contrast."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            if np.std(gray) < 20:  # Low contrast threshold
                self.logger.info("Applying contrast stretching for low-contrast image")
                min_val, max_val = np.min(gray), np.max(gray)
                if max_val > min_val:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            return image
        except Exception as e:
            self.logger.error(f"Low-contrast preprocessing failed: {e}")
            return image
    
    def assess_brightness_level(self, image: np.ndarray) -> Dict:
        """Assess brightness characteristics of the image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            median_brightness = np.median(gray)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Categorize brightness
            t = self.brightness_thresholds
            category = (
                'very_dark' if mean_brightness < t['very_dark'] else
                'dark' if mean_brightness < t['dark'] else
                'slightly_dark' if mean_brightness < t['slightly_dark'] else
                'optimal' if mean_brightness < t['optimal'] else
                'bright' if mean_brightness < t['bright'] else
                'very_bright'
            )
            
            return {
                'mean_brightness': float(mean_brightness),
                'std_brightness': float(std_brightness),
                'median_brightness': float(median_brightness),
                'category': category,
                'histogram': hist.flatten()
            }
        except Exception as e:
            self.logger.error(f"Brightness assessment failed: {e}")
            return {'mean_brightness': 0, 'std_brightness': 0, 'median_brightness': 0, 
                    'category': 'error', 'histogram': np.zeros(256)}
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in the image using Haar cascade classifier."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces[0] if len(faces) > 0 else None  # Return first detected face (x, y, w, h)
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return None
    
    def linear_brightness_adjustment(self, image: np.ndarray, 
                                   custom_params: Optional[Dict] = None) -> np.ndarray:
        """linear brightness adjustment: output = alpha * input + beta."""
        params = custom_params or self.linear_params
        try:
            adjusted = cv2.convertScaleAbs(image, alpha=params['alpha'], beta=params['beta'])
            self.logger.info(f"Linear adjustment applied: alpha={params['alpha']}, beta={params['beta']}")
            return adjusted
        except Exception as e:
            self.logger.error(f"Linear adjustment failed: {e}")
            return image
    
    def gamma_brightness_adjustment(self, image: np.ndarray,
                                  custom_params: Optional[Dict] = None) -> np.ndarray:
        # for non-linear brightness adjustment.
        params = custom_params or self.gamma_params
        try:
            inv_gamma = 1.0 / params['gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 * params['gain']
                            for i in np.arange(0, 256)]).astype("uint8")
            adjusted = cv2.LUT(image, table)
            self.logger.info(f"Gamma adjustment applied: gamma={params['gamma']}, gain={params['gain']}")
            return adjusted
        except Exception as e:
            self.logger.error(f"Gamma adjustment failed: {e}")
            return image
    
    def histogram_based_adjustment(self, image: np.ndarray,
                                 custom_params: Optional[Dict] = None) -> np.ndarray:
        params = custom_params or self.hist_params
        try:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2].astype(np.float32)
                current_mean = np.mean(v_channel)
                target_mean = params['target_mean']
                adjustment = target_mean / current_mean if current_mean > 0 else 1.0
                v_adjusted = np.clip(v_channel * adjustment, 0, 255) if params['preserve_range'] else v_channel * adjustment
                hsv[:, :, 2] = v_adjusted.astype(np.uint8)
                adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            else:
                current_mean = np.mean(image.astype(np.float32))
                target_mean = params['target_mean']
                adjustment = target_mean / current_mean if current_mean > 0 else 1.0
                adjusted = np.clip(image.astype(np.float32) * adjustment, 0, 255).astype(np.uint8)
            self.logger.info(f"Histogram adjustment applied: target_mean={params['target_mean']}")
            return adjusted
        except Exception as e:
            self.logger.error(f"Histogram adjustment failed: {e}")
            return image
    
    def exposure_correction(self, image: np.ndarray, 
                          exposure_compensation: float = 0.0) -> np.ndarray:
        # Apply exposure correction similar to camera exposure compensation.
        try:
            multiplier = 2 ** exposure_compensation
            corrected = np.clip(image.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)
            self.logger.info(f"Exposure correction applied: {exposure_compensation} stops")
            return corrected
        except Exception as e:
            self.logger.error(f"Exposure correction failed: {e}")
            return image
    
    def adaptive_brightness_adjustment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        brightness_info = self.assess_brightness_level(image)
        mean_brightness = brightness_info['mean_brightness']
        category = brightness_info['category']
        method_info = {'original_brightness': mean_brightness, 'category': category}
        
        try:
            if category == 'very_dark':
                adjusted = self.gamma_brightness_adjustment(image, {'gamma': 0.5, 'gain': 1.5})
                method_info['method'] = 'gamma_brighten_strong'
            elif category == 'dark':
                adjusted = self.gamma_brightness_adjustment(image, {'gamma': 0.7, 'gain': 1.3})
                method_info['method'] = 'gamma_brighten_moderate'
            elif category == 'slightly_dark':
                adjusted = self.linear_brightness_adjustment(image, {'alpha': 1.1, 'beta': 15})
                method_info['method'] = 'linear_brighten_light'
            elif category == 'optimal':
                adjusted = self.histogram_based_adjustment(image, {'target_mean': np.mean(self.target_brightness_range)})
                method_info['method'] = 'histogram_optimize'
            elif category == 'bright':
                adjusted = self.gamma_brightness_adjustment(image, {'gamma': 1.3, 'gain': 0.9})
                method_info['method'] = 'gamma_darken_light'
            else:  # very_bright
                adjusted = self.gamma_brightness_adjustment(image, {'gamma': 1.5, 'gain': 0.7})
                method_info['method'] = 'gamma_darken_strong'
            
            final_brightness = self.assess_brightness_level(adjusted)['mean_brightness']
            method_info['final_brightness'] = final_brightness
            self.logger.info(f"Adaptive adjustment applied: {method_info}")
            return adjusted, method_info
        except Exception as e:
            self.logger.error(f"Adaptive adjustment failed: {e}")
            return image, method_info
    
    def region_based_adjustment(self, image: np.ndarray, 
                              face_region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Apply brightness adjustment focusing on face region."""
        try:
            if face_region is None:
                face_region = self.detect_face(image)
            if face_region is None:
                h, w = image.shape[:2]
                center_x, center_y = w // 2, h // 2
                face_w, face_h = w // 3, h // 3
                face_region = (max(0, center_x - face_w // 2), max(0, center_y - face_h // 2),
                              min(face_w, w - center_x), min(face_h, h - center_y))
            
            x, y, fw, fh = face_region
            face_roi = image[y:y+fh, x:x+fw] if len(image.shape) == 3 else image[y:y+fh, x:x+fw]
            face_brightness = self.assess_brightness_level(face_roi)
            
            if face_brightness['category'] in ['very_dark', 'dark']:
                adjusted = self.gamma_brightness_adjustment(image, {'gamma': 0.6, 'gain': 1.4})
            elif face_brightness['category'] in ['bright', 'very_bright']:
                adjusted = self.gamma_brightness_adjustment(image, {'gamma': 1.4, 'gain': 0.8})
            else:
                adjusted = self.histogram_based_adjustment(image, {'target_mean': np.mean(self.target_brightness_range)})
            
            self.logger.info(f"Region-based adjustment applied: face_region={face_region}")
            return adjusted
        except Exception as e:
            self.logger.error(f"Region-based adjustment failed: {e}")
            return image
    
    def multi_scale_brightness_adjustment(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-scale brightness adjustment combining global and local methods."""
        try:
            global_adjusted, _ = self.adaptive_brightness_adjustment(image)
            if len(image.shape) == 3:
                lab = cv2.cvtColor(global_adjusted, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                local_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                local_enhanced = clahe.apply(global_adjusted)
            blended = cv2.addWeighted(global_adjusted, 0.7, local_enhanced, 0.3, 0)
            self.logger.info("Multi-scale adjustment applied")
            return blended
        except Exception as e:
            self.logger.error(f"Multi-scale adjustment failed: {e}")
            return image
    
    def analyze_adjustment(self, original: np.ndarray, adjusted: np.ndarray) -> Dict:
        try:
            original_stats = self.assess_brightness_level(original)
            adjusted_stats = self.assess_brightness_level(adjusted)
            return {
                'original': original_stats,
                'adjusted': adjusted_stats,
                'improvements': {
                    'mean_brightness_change': adjusted_stats['mean_brightness'] - original_stats['mean_brightness'],
                    'std_brightness_change': adjusted_stats['std_brightness'] - original_stats['std_brightness'],
                    'within_target_range': (self.target_brightness_range[0] <= adjusted_stats['mean_brightness'] <= self.target_brightness_range[1])
                }
            }
        except Exception as e:
            self.logger.error(f"Adjustment analysis failed: {e}")
            return {'original': {}, 'adjusted': {}, 'improvements': {}}
    
    def validate_adjustment(self, adjusted: np.ndarray) -> bool:
        """Validate that the adjusted image contains a detectable face."""
        try:
            face_region = self.detect_face(adjusted)
            is_valid = face_region is not None
            self.logger.info(f"Adjustment validation: {'Valid' if is_valid else 'Invalid - no face detected'}")
            return is_valid
        except Exception as e:
            self.logger.error(f"Adjustment validation failed: {e}")
            return False
    
    def batch_adjust_brightness(self, images: List[np.ndarray], method: str = 'adaptive') -> List[np.ndarray]:
        """Process a batch of images for brightness adjustment."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda img: adjust_brightness(img, method, self), images))
            self.logger.info(f"Batch processed {len(images)} images with method: {method}")
            return results
        except Exception as e:
            self.logger.error(f"Batch adjustment failed: {e}")
            return images

def adjust_brightness(image: np.ndarray, method: str = 'adaptive', 
                    adjuster: Optional[BrightnessAdjuster] = None) -> np.ndarray:
    adjuster = adjuster or BrightnessAdjuster()
    
    try:
        # Preprocess for low contrast
        image = adjuster.preprocess_low_contrast(image)
        
        if method == 'adaptive':
            result, _ = adjuster.adaptive_brightness_adjustment(image)
        elif method == 'linear':
            result = adjuster.linear_brightness_adjustment(image)
        elif method == 'gamma':
            result = adjuster.gamma_brightness_adjustment(image)
        elif method == 'histogram':
            result = adjuster.histogram_based_adjustment(image)
        elif method == 'exposure':
            result = adjuster.exposure_correction(image, 0.5)
        elif method == 'multi_scale':
            result = adjuster.multi_scale_brightness_adjustment(image)
        else:
            adjuster.logger.warning(f"Unknown method: {method}, using adaptive")
            result, _ = adjuster.adaptive_brightness_adjustment(image)
        
        # Validate adjustment
        if not adjuster.validate_adjustment(result):
            adjuster.logger.warning("Adjustment validation failed, returning original image")
            return image
        
        return result
    except Exception as e:
        adjuster.logger.error(f"Brightness adjustment failed: {e}")
        return image

# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Test the brightness adjuster
#     adjuster = BrightnessAdjuster()
#     test_image = cv2.imread("test_image.jpg")
    
#     if test_image is not None:
#         methods = ['adaptive', 'linear', 'gamma', 'histogram', 'exposure', 'multi_scale']
#         for method in methods:
#             adjusted = adjust_brightness(test_image, method=method, adjuster=adjuster)
#             analysis = adjuster.analyze_adjustment(test_image, adjusted)
#             print(f"\nMethod: {method}")
#             print("Improvements:", analysis['improvements'])
            
#             # Test batch processing
#             batch_results = adjuster.batch_adjust_brightness([test_image] * 3, method=method)
#             print(f"Batch processed {len(batch_results)} images")