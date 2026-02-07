"""
Wire Segmentation
Extracted from wire_color_detector.py for better modularity.
Splits wire ROI into individual wire segments for color detection.
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class WireSegmentExtractor:
    """
    Extracts individual wire segments from wire ROI.
    Divides wire region into equal segments based on number of wires.
    """
    
    def __init__(self, num_wires: int):
        """
        Initialize segment extractor.
        
        Args:
            num_wires: Number of wires to extract
        """
        self.num_wires = num_wires
        logger.debug(f"WireSegmentExtractor initialized for {num_wires} wires")
    
    def extract_segments(self, wire_roi: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual wire segments from wire ROI.
        
        Args:
            wire_roi: Wire region of interest
            
        Returns:
            List of wire segment images
        """
        height, width = wire_roi.shape[:2]
        segment_width = width // self.num_wires
        
        segments = []
        for i in range(self.num_wires):
            x_start = i * segment_width
            # Last segment takes remaining width to handle rounding
            x_end = (i + 1) * segment_width if i < self.num_wires - 1 else width
            
            segment = wire_roi[:, x_start:x_end]
            segments.append(segment)
            
            logger.debug(f"Segment {i}: x={x_start} to {x_end}, size={segment.shape}")
        
        return segments
    
    def get_segment_coordinates(self, roi_x: int, roi_y: int,
                                roi_width: int, roi_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Get absolute coordinates of wire segments for visualization.
        
        Args:
            roi_x, roi_y: ROI top-left corner
            roi_width, roi_height: ROI dimensions
            
        Returns:
            List of (x, y, width, height) tuples for each segment
        """
        segment_width = roi_width // self.num_wires
        
        coordinates = []
        for i in range(self.num_wires):
            x_start = roi_x + (i * segment_width)
            # Last segment takes remaining width
            w = segment_width if i < self.num_wires - 1 else (roi_x + roi_width - x_start)
            
            coordinates.append((x_start, roi_y, w, roi_height))
        
        return coordinates
