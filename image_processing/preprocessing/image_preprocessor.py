# part1_preprocessing/image_preprocessor.py

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import logging

from .noise_removal import NoiseRemover
from .contrast_enhancement import ContrastEnhancer
from .brightness_adjustment import BrightnessAdjuster
from .utils import ImageUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize processing modules
        self.noise_remover = NoiseRemover(self.config.get('noise_removal', {}))
        self.contrast_enhancer = ContrastEnhancer(self.config.get('contrast_enhancement', {}))
        self.brightness_adjuster = BrightnessAdjuster(self.config.get('brightness_adjustment', {}))
        self.utils = ImageUtils()
        
        logger.info("ImagePreprocessor initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'noise_removal': {
                'method': 'bilateral',
                'bilateral_d': 9,
                'bilateral_sigma_color': 75,
                'bilateral_sigma_space': 75
            },
            'contrast_enhancement': {
                'method': 'clahe',
                'clahe_clip_limit': 3.0,
                'clahe_tile_grid_size': (8, 8)
            },
            'brightness_adjustment': {
                'method': 'auto',
                'target_brightness': 128,
                'alpha': 1.2,
                'beta': 10,
                'gamma': 1.0
            },
            'output_size': (224, 224),  # Standard size 
            'save_intermediate_steps': False
        }
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            logger.info(f"Image loaded successfully: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def preprocess_single_step(self, image: np.ndarray, step: str, **kwargs) -> np.ndarray:
        if step == 'noise_removal':
            method = kwargs.get('method', self.config['noise_removal']['method'])
            return self.noise_remover.remove_noise(image, method=method, **kwargs)
        
        elif step == 'contrast_enhancement':
            method = kwargs.get('method', self.config['contrast_enhancement']['method'])
            return self.contrast_enhancer.enhance_contrast(image, method=method, **kwargs)
        
        elif step == 'brightness_adjustment':
            method = kwargs.get('method', self.config['brightness_adjustment']['method'])
            return self.brightness_adjuster.adjust_brightness(image, method=method, **kwargs)
        
        else:
            logger.warning(f"Unknown preprocessing step: {step}")
            return image
    
    def preprocess_image(self, 
                        image: np.ndarray, 
                        steps: Optional[list] = None,
                        save_intermediate: bool = False,
                        output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        if steps is None:
            steps = ['noise_removal', 'contrast_enhancement', 'brightness_adjustment']
        
        results = {'original': image.copy()}
        current_image = image.copy()
        
        logger.info(f"Starting preprocessing pipeline with steps: {steps}")
        
        for step in steps:
            logger.info(f"Applying step: {step}")
            
            try:
                processed_image = self.preprocess_single_step(current_image, step)
                results[step] = processed_image.copy()
                current_image = processed_image
                
                # Save intermediate result if requested
                if save_intermediate and output_dir:
                    self._save_intermediate_result(processed_image, step, output_dir)
                    
            except Exception as e:
                logger.error(f"Error in preprocessing step {step}: {str(e)}")
                results[step] = current_image  # Use previous image if step fails
        
        # Resize to standard output size
        output_size = self.config.get('output_size', (224, 224))
        final_image = cv2.resize(current_image, output_size)
        results['final'] = final_image
        
        logger.info("Preprocessing pipeline completed successfully")
        return results
    
    def preprocess_image_from_path(self, 
                                  image_path: str,
                                  steps: Optional[list] = None,
                                  save_intermediate: bool = False,
                                  output_dir: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        image = self.load_image(image_path)
        if image is None:
            return None
        
        return self.preprocess_image(image, steps, save_intermediate, output_dir)
    
    #    Preprocess multiple images in batch.
    def batch_preprocess(self, 
                        image_paths: list,
                        steps: Optional[list] = None,
                        save_intermediate: bool = False,
                        output_dir: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        results = {}
        
        logger.info(f"Starting batch preprocessing for {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.preprocess_image_from_path(
                image_path, steps, save_intermediate, output_dir
            )
            
            if result is not None:
                results[image_path] = result
            else:
                logger.warning(f"Failed to process image: {image_path}")
        
        logger.info(f"Batch preprocessing completed. Successfully processed {len(results)} images")
        return results
    
    def _save_intermediate_result(self, image: np.ndarray, step: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{step}_result.jpg")
        cv2.imwrite(output_path, image)
        logger.debug(f"Saved intermediate result: {output_path}")
    
    def visualize_preprocessing_steps(self, 
                                    results: Dict[str, np.ndarray],
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (15, 10)):
        
        steps = ['original', 'noise_removal', 'contrast_enhancement', 'brightness_adjustment', 'final']
        available_steps = [step for step in steps if step in results]
        
        n_steps = len(available_steps)
        fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=figsize)
        axes = axes.flatten() if n_steps > 2 else [axes] if n_steps == 1 else axes
        
        for i, step in enumerate(available_steps):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                image_rgb = cv2.cvtColor(results[step], cv2.COLOR_BGR2RGB)
                axes[i].imshow(image_rgb)
                axes[i].set_title(step.replace('_', ' ').title())
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(available_steps), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        return self.utils.calculate_quality_metrics(image)
    
    def compare_before_after(self, 
                           original: np.ndarray, 
                           processed: np.ndarray) -> Dict[str, Any]:
        
        original_metrics = self.get_image_quality_metrics(original)
        processed_metrics = self.get_image_quality_metrics(processed)
        
        comparison = {
            'original_metrics': original_metrics,
            'processed_metrics': processed_metrics,
            'improvement': {}
        }
        
        # Calculate improvements
        for metric in original_metrics:
            if metric in processed_metrics:
                improvement = processed_metrics[metric] - original_metrics[metric]
                comparison['improvement'][metric] = improvement
        
        return comparison
    
    def update_config(self, new_config: Dict[str, Any]):
        self.config.update(new_config)
        
        # Reinitialize modules with new config
        self.noise_remover = NoiseRemover(self.config.get('noise_removal', {}))
        self.contrast_enhancer = ContrastEnhancer(self.config.get('contrast_enhancement', {}))
        self.brightness_adjuster = BrightnessAdjuster(self.config.get('brightness_adjustment', {}))
        
        logger.info("Configuration updated successfully")

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Example single image processing
    image_path = "sample_image.jpg"
    results = preprocessor.preprocess_image_from_path(
        image_path, 
        save_intermediate=True,
        output_dir="preprocessing_results"
    )
    
    if results:
        # Visualize results
        preprocessor.visualize_preprocessing_steps(results)
        
        # Get quality metrics
        comparison = preprocessor.compare_before_after(
            results['original'], 
            results['final']
        )
        print("Quality Comparison:", comparison)