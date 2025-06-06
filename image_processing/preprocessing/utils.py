import cv2
import numpy as np
import logging
from typing import Union, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from pathlib import Path

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing.log')
        ]
    )
    return logging.getLogger(__name__)

def load_image(image_path: Union[str, Path], 
               color_mode: str = 'BGR') -> Optional[np.ndarray]:
    try:
        if color_mode.upper() == 'GRAY':
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if color_mode.upper() == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None
            
        logging.info(f"Successfully loaded image: {image_path}, shape: {image.shape}")
        return image
        
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, 
               output_path: Union[str, Path], 
               color_mode: str = 'BGR') -> bool:
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert color mode if necessary
        if len(image.shape) == 3 and color_mode.upper() == 'RGB':
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image
        
        success = cv2.imwrite(str(output_path), image_to_save)
        
        if success:
            logging.info(f"Successfully saved image: {output_path}")
            return True
        else:
            logging.error(f"Failed to save image: {output_path}")
            return False
            
    except Exception as e:
        logging.error(f"Error saving image {output_path}: {e}")
        return False

def validate_image(image: np.ndarray) -> Dict[str, Union[bool, str, Tuple]]:
    validation_result = {
        'is_valid': True,
        'error_message': '',
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'is_color': len(image.shape) == 3,
        'channels': image.shape[2] if len(image.shape) == 3 else 1
    }
    
    # Check basic properties
    if image is None:
        validation_result['is_valid'] = False
        validation_result['error_message'] = 'Image is None'
        return validation_result
    
    if len(image.shape) not in [2, 3]:
        validation_result['is_valid'] = False
        validation_result['error_message'] = f'Invalid image dimensions: {image.shape}'
        return validation_result
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        validation_result['is_valid'] = False
        validation_result['error_message'] = f'Invalid number of channels: {image.shape[2]}'
        return validation_result
    
    # Check value range
    if image.dtype == np.uint8:
        if validation_result['min_value'] < 0 or validation_result['max_value'] > 255:
            validation_result['is_valid'] = False
            validation_result['error_message'] = 'Values out of range for uint8 [0, 255]'
    elif image.dtype in [np.float32, np.float64]:
        if validation_result['min_value'] < 0 or validation_result['max_value'] > 1:
            logging.warning('Float image values may be out of expected range [0, 1]')
    
    return validation_result

def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int], 
                interpolation: int = cv2.INTER_LINEAR,
                maintain_aspect_ratio: bool = False) -> np.ndarray:
    try:
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            # Create canvas and center the image
            canvas = np.zeros((target_h, target_w, image.shape[2] if len(image.shape) == 3 else 1), 
                            dtype=image.dtype)
            if len(image.shape) == 2:
                canvas = np.zeros((target_h, target_w), dtype=image.dtype)
            
            # Calculate position to center the image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            if len(image.shape) == 3:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            else:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=interpolation)
            
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return image

def calculate_image_statistics(image: np.ndarray) -> Dict[str, float]:
    stats = {}
    
    try:
        if len(image.shape) == 3:
            # Color image statistics
            for i, channel in enumerate(['Blue', 'Green', 'Red']):
                channel_data = image[:, :, i]
                stats[f'{channel}_mean'] = float(np.mean(channel_data))
                stats[f'{channel}_std'] = float(np.std(channel_data))
                stats[f'{channel}_min'] = float(np.min(channel_data))
                stats[f'{channel}_max'] = float(np.max(channel_data))
            
            # Convert to grayscale for overall statistics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Overall statistics
        stats['overall_mean'] = float(np.mean(gray))
        stats['overall_std'] = float(np.std(gray))
        stats['overall_min'] = float(np.min(gray))
        stats['overall_max'] = float(np.max(gray))
        stats['overall_median'] = float(np.median(gray))
        
        # Histogram statistics
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        stats['histogram_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # Contrast measures
        stats['rms_contrast'] = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))
        stats['michelson_contrast'] = float((np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-10))
        
        # Sharpness measure using Laplacian variance
        stats['laplacian_variance'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
    except Exception as e:
        logging.error(f"Error calculating image statistics: {e}")
        stats['error'] = str(e)
    
    return stats

def create_comparison_plot(original: np.ndarray, 
                         processed: np.ndarray, 
                         title: str = "Image Comparison",
                         save_path: Optional[str] = None) -> None:
    #   Create side-by-side comparison plot of original and processed images
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Display processed image
        if len(processed.shape) == 3:
            axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Processed')
        axes[1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Comparison plot saved: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logging.error(f"Error creating comparison plot: {e}")

def create_histogram_comparison(original: np.ndarray, 
                              processed: np.ndarray,
                              title: str = "Histogram Comparison",
                              save_path: Optional[str] = None) -> None:
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        # Original image
        axes[0, 0].imshow(orig_gray, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(proc_gray, cmap='gray')
        axes[0, 1].set_title('Processed Image')
        axes[0, 1].axis('off')
        
        # Original histogram
        hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        axes[1, 0].plot(hist_orig)
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        # Processed histogram
        hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
        axes[1, 1].plot(hist_proc)
        axes[1, 1].set_title('Processed Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Histogram comparison saved: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logging.error(f"Error creating histogram comparison: {e}")

def batch_process_images(input_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        processing_function,
                        file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
                        **kwargs) -> Dict[str, bool]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logging.info(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            try:
                # Load image
                image = load_image(image_file)
                if image is None:
                    results[str(image_file)] = False
                    continue
                
                # Process image
                processed_image = processing_function(image, **kwargs)
                
                # Save processed image
                output_file = output_path / image_file.name
                success = save_image(processed_image, output_file)
                results[str(image_file)] = success
                
                if success:
                    logging.info(f"Successfully processed: {image_file}")
                else:
                    logging.error(f"Failed to save processed image: {image_file}")
                
            except Exception as e:
                logging.error(f"Error processing {image_file}: {e}")
                results[str(image_file)] = False
        
        return results
    
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        return results