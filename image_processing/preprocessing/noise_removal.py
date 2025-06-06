import cv2
import numpy as np
from typing import Union, Tuple, Optional
import logging

class NoiseRemover:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.bilateral_params = {
            'd': 9,           # Diameter of neighborhood
            'sigmaColor': 75,  # Filter sigma in color space
            'sigmaSpace': 75   # Filter sigma in coordinate space
        }
        
        self.nlm_params = {
            'h': 10,          # Filtering strength
            'templateWindowSize': 7,  # Template patch size
            'searchWindowSize': 21    # Search window size
        }
        
        self.gaussian_params = {
            'ksize': (5, 5),  # Kernel size
            'sigmaX': 1.0     # Standard deviation in X direction
        }
        
        self.median_params = {
            'ksize': 5        # Kernel size
        }
    
    def assess_noise_level(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate Laplacian variance as noise estimation
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def bilateral_filter_denoise(self, image: np.ndarray, 
                               custom_params: Optional[dict] = None) -> np.ndarray:
        params = custom_params if custom_params else self.bilateral_params
        
        try:
            denoised = cv2.bilateralFilter(
                image, 
                params['d'], 
                params['sigmaColor'], 
                params['sigmaSpace']
            )
            self.logger.info(f"Bilateral filtering applied with params: {params}")
            return denoised
            
        except Exception as e:
            self.logger.error(f"Bilateral filtering failed: {e}")
            return image
    
    def nlm_denoise(self, image: np.ndarray, 
                   custom_params: Optional[dict] = None) -> np.ndarray:
        # Non-local Means denoising
        params = custom_params if custom_params else self.nlm_params
        
        try:
            if len(image.shape) == 3:
                # Color image
                denoised = cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    params['h'],
                    params['h'],
                    params['templateWindowSize'],
                    params['searchWindowSize']
                )
            else:
                # Grayscale image
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    params['h'],
                    params['templateWindowSize'],
                    params['searchWindowSize']
                )
            
            self.logger.info(f"NLM denoising applied with params: {params}")
            return denoised
            
        except Exception as e:
            self.logger.error(f"NLM denoising failed: {e}")
            return image
    
    def gaussian_denoise(self, image: np.ndarray, 
                        custom_params: Optional[dict] = None) -> np.ndarray:
        params = custom_params if custom_params else self.gaussian_params
        
        try:
            denoised = cv2.GaussianBlur(
                image, 
                params['ksize'], 
                params['sigmaX']
            )
            self.logger.info(f"Gaussian denoising applied with params: {params}")
            return denoised
            
        except Exception as e:
            self.logger.error(f"Gaussian denoising failed: {e}")
            return image
    
    def median_denoise(self, image: np.ndarray, 
                      custom_params: Optional[dict] = None) -> np.ndarray:
        params = custom_params if custom_params else self.median_params
        
        try:
            denoised = cv2.medianBlur(image, params['ksize'])
            self.logger.info(f"Median denoising applied with ksize: {params['ksize']}")
            return denoised
            
        except Exception as e:
            self.logger.error(f"Median denoising failed: {e}")
            return image
    
    def adaptive_denoise(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        noise_level = self.assess_noise_level(image)
        method_info = {'noise_level': noise_level}
        
        if noise_level < 100:
            # Low noise - light Gaussian blur
            denoised = self.gaussian_denoise(image, {'ksize': (3, 3), 'sigmaX': 0.5})
            method_info['method'] = 'light_gaussian'
            
        elif noise_level < 500:
            # Medium noise - bilateral filter
            denoised = self.bilateral_filter_denoise(image)
            method_info['method'] = 'bilateral'
            
        elif noise_level < 1000:
            # High noise - NLM denoising
            denoised = self.nlm_denoise(image)
            method_info['method'] = 'nlm'
            
        else:
            # Very high noise - combined approach
            temp = self.median_denoise(image, {'ksize': 3})
            denoised = self.bilateral_filter_denoise(temp)
            method_info['method'] = 'combined_median_bilateral'
        
        self.logger.info(f"Adaptive denoising applied: {method_info}")
        return denoised, method_info
    
    def multi_stage_denoise(self, image: np.ndarray, 
                           stages: Optional[list] = None) -> np.ndarray:
        if stages is None:
            stages = ['median', 'bilateral']
        
        result = image.copy()
        
        for stage in stages:
            if stage == 'bilateral':
                result = self.bilateral_filter_denoise(result)
            elif stage == 'nlm':
                result = self.nlm_denoise(result)
            elif stage == 'gaussian':
                result = self.gaussian_denoise(result)
            elif stage == 'median':
                result = self.median_denoise(result)
            else:
                self.logger.warning(f"Unknown denoising stage: {stage}")
        
        self.logger.info(f"Multi-stage denoising completed with stages: {stages}")
        return result

def denoise_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    denoiser = NoiseRemover()
    
    if method == 'adaptive':
        result, _ = denoiser.adaptive_denoise(image)
        return result
    elif method == 'bilateral':
        return denoiser.bilateral_filter_denoise(image)
    elif method == 'nlm':
        return denoiser.nlm_denoise(image)
    elif method == 'gaussian':
        return denoiser.gaussian_denoise(image)
    elif method == 'median':
        return denoiser.median_denoise(image)
    else:
        logging.warning(f"Unknown method: {method}, using adaptive")
        result, _ = denoiser.adaptive_denoise(image)
        return result