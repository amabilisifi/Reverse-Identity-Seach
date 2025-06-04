import cv2
import numpy as np
from typing import Union, Tuple, Optional
import logging

class ContrastEnhancer:
    def __init__(self):
        """Initialize the contrast enhancer with default parameters"""
        self.logger = logging.getLogger(__name__)
        
        # Default parameters for CLAHE
        self.clahe_params = {
            'clipLimit': 2.0,
            'tileGridSize': (8, 8)
        }
        
        # Default parameters for gamma correction
        self.gamma_params = {
            'gamma': 1.2,
            'gain': 1.0
        }
        
        # Default parameters for adaptive equalization
        self.adaptive_params = {
            'window_size': 64,
            'threshold': 0.1
        }
    
    def assess_contrast_level(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate RMS contrast
        mean_intensity = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
        
        return rms_contrast
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                # Convert to YUV color space
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                # Apply histogram equalization to Y channel
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                # Convert back to BGR
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                # Grayscale image
                result = cv2.equalizeHist(image)
            
            self.logger.info("Global histogram equalization applied")
            return result
            
        except Exception as e:
            self.logger.error(f"Histogram equalization failed: {e}")
            return image
    
    def clahe_enhancement(self, image: np.ndarray, 
                         custom_params: Optional[dict] = None) -> np.ndarray:
        params = custom_params if custom_params else self.clahe_params
        
        try:
            clahe = cv2.createCLAHE(
                clipLimit=params['clipLimit'],
                tileGridSize=params['tileGridSize']
            )
            
            if len(image.shape) == 3:
                # Convert to LAB color space for better results
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                # Apply CLAHE to L channel
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                # Convert back to BGR
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                result = clahe.apply(image)
            
            self.logger.info(f"CLAHE enhancement applied with params: {params}")
            return result
            
        except Exception as e:
            self.logger.error(f"CLAHE enhancement failed: {e}")
            return image
    
    def gamma_correction(self, image: np.ndarray, 
                        custom_params: Optional[dict] = None) -> np.ndarray:
        params = custom_params if custom_params else self.gamma_params
        
        try:
            # Normalize image to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            
            # Apply gamma correction
            gamma_corrected = params['gain'] * np.power(normalized, params['gamma'])
            
            # Convert back to [0, 255]
            result = np.clip(gamma_corrected * 255.0, 0, 255).astype(np.uint8)
            
            self.logger.info(f"Gamma correction applied with gamma: {params['gamma']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Gamma correction failed: {e}")
            return image
    
    def adaptive_histogram_equalization(self, image: np.ndarray,
                                      custom_params: Optional[dict] = None) -> np.ndarray:
        # Apply adaptive histogram equalization with local processing

        params = custom_params if custom_params else self.adaptive_params
        
        try:
            if len(image.shape) == 3:
                # Convert to grayscale for processing
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                process_image = gray
            else:
                process_image = image.copy()
            
            h, w = process_image.shape
            window_size = params['window_size']
            threshold = params['threshold']
            
            result = np.zeros_like(process_image)
            
            # Process image in overlapping windows
            step_size = window_size // 2
            
            for y in range(0, h, step_size):
                for x in range(0, w, step_size):
                    # Define window boundaries
                    y1 = max(0, y - window_size // 2)
                    y2 = min(h, y + window_size // 2)
                    x1 = max(0, x - window_size // 2)
                    x2 = min(w, x + window_size // 2)
                    
                    # Extract window
                    window = process_image[y1:y2, x1:x2]
                    
                    # Check if enhancement is needed
                    contrast = self.assess_contrast_level(window)
                    if contrast < threshold * 100:  # Threshold scaling
                        # Apply local histogram equalization
                        enhanced_window = cv2.equalizeHist(window)
                        result[y1:y2, x1:x2] = enhanced_window
                    else:
                        result[y1:y2, x1:x2] = window
            
            if len(image.shape) == 3:
                # Convert back to color
                result_color = image.copy()
                # Apply enhancement to each channel proportionally
                for i in range(3):
                    channel_ratio = result.astype(np.float32) / (process_image.astype(np.float32) + 1e-6)
                    result_color[:, :, i] = np.clip(
                        image[:, :, i].astype(np.float32) * channel_ratio, 0, 255
                    ).astype(np.uint8)
                result = result_color
            
            self.logger.info(f"Adaptive histogram equalization applied")
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive histogram equalization failed: {e}")
            return image
    
    def linear_contrast_stretching(self, image: np.ndarray, 
                                 percentile_low: float = 2, 
                                 percentile_high: float = 98) -> np.ndarray:
        # Apply linear contrast stretching using percentile-based clipping
        
        try:
            if len(image.shape) == 3:
                result = np.zeros_like(image)
                for i in range(3):
                    channel = image[:, :, i]
                    p_low = np.percentile(channel, percentile_low)
                    p_high = np.percentile(channel, percentile_high)
                    
                    # Stretch contrast
                    stretched = np.clip((channel - p_low) * (255.0 / (p_high - p_low)), 0, 255)
                    result[:, :, i] = stretched.astype(np.uint8)
            else:
                p_low = np.percentile(image, percentile_low)
                p_high = np.percentile(image, percentile_high)
                result = np.clip((image - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
            
            self.logger.info(f"Linear contrast stretching applied (percentiles: {percentile_low}-{percentile_high})")
            return result
            
        except Exception as e:
            self.logger.error(f"Linear contrast stretching failed: {e}")
            return image
    
    def adaptive_contrast_enhancement(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        contrast_level = self.assess_contrast_level(image)
        method_info = {'contrast_level': contrast_level}
        
        if contrast_level < 20:
            # Very low contrast - aggressive enhancement
            enhanced = self.clahe_enhancement(image, {'clipLimit': 3.0, 'tileGridSize': (8, 8)})
            method_info['method'] = 'aggressive_clahe'
            
        elif contrast_level < 40:
            # Low contrast - standard CLAHE
            enhanced = self.clahe_enhancement(image)
            method_info['method'] = 'standard_clahe'
            
        elif contrast_level < 60:
            # Medium contrast - light gamma correction
            enhanced = self.gamma_correction(image, {'gamma': 1.1, 'gain': 1.0})
            method_info['method'] = 'light_gamma'
            
        elif contrast_level < 80:
            # Good contrast - linear stretching
            enhanced = self.linear_contrast_stretching(image, 1, 99)
            method_info['method'] = 'linear_stretching'
            
        else:
            # High contrast - minimal enhancement
            enhanced = self.gamma_correction(image, {'gamma': 1.05, 'gain': 1.0})
            method_info['method'] = 'minimal_gamma'
        
        self.logger.info(f"Adaptive contrast enhancement applied: {method_info}")
        return enhanced, method_info
    
    def multi_scale_enhancement(self, image: np.ndarray) -> np.ndarray:
        try:
            # Apply different enhancement techniques and blend
            clahe_result = self.clahe_enhancement(image, {'clipLimit': 1.5, 'tileGridSize': (8, 8)})
            gamma_result = self.gamma_correction(image, {'gamma': 1.1, 'gain': 1.0})
            stretch_result = self.linear_contrast_stretching(image, 2, 98)
            
            # Weighted blending of results
            blended = (0.4 * clahe_result.astype(np.float32) + 
                      0.3 * gamma_result.astype(np.float32) + 
                      0.3 * stretch_result.astype(np.float32))
            
            result = np.clip(blended, 0, 255).astype(np.uint8)
            
            self.logger.info("Multi-scale contrast enhancement applied")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-scale enhancement failed: {e}")
            return image

def enhance_contrast(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    enhancer = ContrastEnhancer()
    
    if method == 'adaptive':
        result, _ = enhancer.adaptive_contrast_enhancement(image)
        return result
    elif method == 'clahe':
        return enhancer.clahe_enhancement(image)
    elif method == 'histogram':
        return enhancer.histogram_equalization(image)
    elif method == 'gamma':
        return enhancer.gamma_correction(image)
    elif method == 'linear':
        return enhancer.linear_contrast_stretching(image)
    elif method == 'multi_scale':
        return enhancer.multi_scale_enhancement(image)
    else:
        logging.warning(f"Unknown method: {method}, using adaptive")
        result, _ = enhancer.adaptive_contrast_enhancement(image)
        return result