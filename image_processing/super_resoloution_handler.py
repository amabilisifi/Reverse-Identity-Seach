import cv2
import os

class SuperResolutionHandler:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Super-resolution model not found at {model_path}")
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        model_name = os.path.basename(model_path).split('_')[0].lower()
        scale_factor = int(os.path.basename(model_path).split('x')[1].split('.')[0]) if 'x' in os.path.basename(model_path) else 4
        self.sr.setModel(model_name, scale_factor)

    def upscale(self, image):
        """Upscale the image."""
        return self.sr.upsample(image)