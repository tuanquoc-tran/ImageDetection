"""
Wire Color Detection
Moved from wire_color_detector.py to vision module.
HSV-based color classification for wire inspection.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class WireColor(Enum):
    """Enumeration of detectable wire colors."""
    BLACK = "black"
    WHITE = "white"
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    BROWN = "brown"
    GRAY = "gray"
    UNKNOWN = "unknown"


class WireColorDetector:
    """
    Production-grade wire color detector using HSV color space.
    Optimized for industrial lighting conditions.
    """
    
    # HSV color ranges for each wire color (H: 0-180, S: 0-255, V: 0-255)
    COLOR_RANGES = {
        WireColor.BLACK: {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 50])
        },
        WireColor.WHITE: {
            'lower': np.array([0, 0, 200]),
            'upper': np.array([180, 30, 255])
        },
        WireColor.RED: {
            'lower': np.array([0, 100, 100]),
            'upper': np.array([10, 255, 255])
        },
        WireColor.BLUE: {
            'lower': np.array([100, 100, 100]),
            'upper': np.array([130, 255, 255])
        },
        WireColor.GREEN: {
            'lower': np.array([40, 100, 100]),
            'upper': np.array([80, 255, 255])
        },
        WireColor.YELLOW: {
            'lower': np.array([20, 100, 100]),
            'upper': np.array([30, 255, 255])
        },
        WireColor.ORANGE: {
            'lower': np.array([10, 100, 100]),
            'upper': np.array([20, 255, 255])
        },
        WireColor.BROWN: {
            'lower': np.array([10, 100, 50]),
            'upper': np.array([20, 255, 150])
        },
        WireColor.GRAY: {
            'lower': np.array([0, 0, 50]),
            'upper': np.array([180, 50, 200])
        }
    }
    
    def __init__(self):
        """Initialize color detector."""
        logger.debug("WireColorDetector initialized")
    
    def detect_color(self, wire_segment: np.ndarray) -> Tuple[WireColor, float]:
        """
        Detect dominant wire color in segment.
        
        Args:
            wire_segment: Wire segment image (BGR format)
            
        Returns:
            Tuple of (detected_color, confidence_score)
        """
        # Convert to HSV for robust color detection
        hsv = cv2.cvtColor(wire_segment, cv2.COLOR_BGR2HSV)
        
        best_color = WireColor.UNKNOWN
        best_confidence = 0.0
        
        # Test each color range
        for color, ranges in self.COLOR_RANGES.items():
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            pixel_count = cv2.countNonZero(mask)
            total_pixels = wire_segment.shape[0] * wire_segment.shape[1]
            confidence = pixel_count / total_pixels
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_color = color
        
        logger.debug(f"Detected color: {best_color.value} (confidence: {best_confidence:.3f})")
        
        return best_color, best_confidence
