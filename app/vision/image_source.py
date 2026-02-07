"""
Image Source Abstraction
Moved from root to vision module.
Provides unified interface for both static image files and live camera feeds.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Type of image source."""
    IMAGE_FILE = "image_file"
    CAMERA = "camera"
    UNKNOWN = "unknown"


class ImageSource(ABC):
    """Abstract base class for image sources with context manager support."""
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are released."""
        self.release()
        return False
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from source."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is available."""
        pass
    
    @abstractmethod
    def release(self):
        """Release resources."""
        pass
    
    @abstractmethod
    def get_source_info(self) -> str:
        """Get human-readable source information."""
        pass
    
    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Get source type enumeration."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict:
        """Get source metadata."""
        pass


class ImageFileSource(ImageSource):
    """Image source from static file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.image = None
        self._load_image()
    
    def _load_image(self):
        """Load image from file."""
        try:
            self.image = cv2.imread(self.filepath)
            if self.image is None:
                logger.error(f"Failed to load image: {self.filepath}")
            else:
                logger.info(f"Image loaded: {self.filepath} ({self.image.shape})")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            self.image = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Return the loaded image (creates copy for safety)."""
        if self.image is not None:
            return self.image.copy()
        return None
    
    def is_available(self) -> bool:
        """Check if image was loaded successfully."""
        return self.image is not None
    
    def release(self):
        """Release image memory."""
        if self.image is not None:
            logger.debug(f"Releasing image: {self.filepath}")
        self.image = None
    
    def get_source_info(self) -> str:
        """Get file path info."""
        return f"Image File: {self.filepath}"
    
    def get_source_type(self) -> SourceType:
        """Get source type."""
        return SourceType.IMAGE_FILE
    
    def get_metadata(self) -> Dict:
        """Get image metadata."""
        if self.image is not None:
            height, width = self.image.shape[:2]
            return {
                'source_type': self.get_source_type().value,
                'filepath': self.filepath,
                'width': width,
                'height': height,
                'channels': self.image.shape[2] if len(self.image.shape) > 2 else 1
            }
        return {
            'source_type': self.get_source_type().value,
            'filepath': self.filepath,
            'available': False
        }


class CameraSource(ImageSource):
    """Image source from camera device."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.frame_count = 0
        self._open_camera()
    
    def _open_camera(self):
        """Open camera device."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                self.cap = None
            else:
                logger.info(f"Camera opened: {self.camera_index}")
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            self.cap = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture frame from camera."""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return frame
        return None
    
    def is_available(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            logger.debug(f"Releasing camera: {self.camera_index}")
            self.cap.release()
            self.cap = None
    
    def get_source_info(self) -> str:
        """Get camera device info."""
        return f"Camera Device: {self.camera_index}"
    
    def get_source_type(self) -> SourceType:
        """Get source type."""
        return SourceType.CAMERA
    
    def get_metadata(self) -> Dict:
        """Get camera metadata."""
        metadata = {
            'source_type': self.get_source_type().value,
            'camera_index': self.camera_index,
            'frame_count': self.frame_count,
            'available': self.is_available()
        }
        
        if self.cap is not None and self.cap.isOpened():
            metadata.update({
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS)
            })
        
        return metadata


class ImageSourceFactory:
    """Factory for creating image sources with unified interface."""
    
    @staticmethod
    def create_image_source(filepath: str) -> ImageFileSource:
        """Create image file source."""
        return ImageFileSource(filepath)
    
    @staticmethod
    def create_camera_source(camera_index: int = 0) -> CameraSource:
        """Create camera source."""
        return CameraSource(camera_index)
    
    @staticmethod
    def create_source_from_config(config: Dict) -> Optional[ImageSource]:
        """Create image source from configuration dictionary."""
        source_type = config.get('type', '').lower()
        
        if source_type in ['image', 'file', 'image_file']:
            filepath = config.get('filepath')
            if filepath:
                return ImageFileSource(filepath)
        elif source_type == 'camera':
            camera_index = config.get('index', 0)
            return CameraSource(camera_index)
        
        logger.error(f"Invalid source config: {config}")
        return None
