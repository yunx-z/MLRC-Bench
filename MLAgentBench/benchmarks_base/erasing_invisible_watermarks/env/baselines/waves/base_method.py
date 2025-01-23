from abc import ABC, abstractmethod

class BaseWatermarkRemovalMethod(ABC):
    """Base class for watermark removal methods"""
    
    @abstractmethod
    def remove_watermark(self, image):
        """Remove watermark from image"""
        pass
    
    @abstractmethod
    def evaluate(self, images):
        """Evaluate method on a set of images"""
        pass 