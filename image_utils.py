"""
Image processing utilities for AOI connector inspection
Handles ROI extraction, preprocessing, and basic image operations
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class ROIExtractor:
    """Handles extraction and management of Regions of Interest"""
    
    @staticmethod
    def extract_roi(image: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
        """
        Extract a rectangular ROI from an image
        
        Args:
            image: Input image
            roi: Dictionary with 'x', 'y', 'width', 'height'
            
        Returns:
            Extracted ROI as numpy array
        """
        x, y = roi['x'], roi['y']
        w, h = roi['width'], roi['height']
        
        # Ensure ROI is within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        return image[y:y+h, x:x+w].copy()
    
    @staticmethod
    def get_wire_slot_rois(config) -> List[Dict[str, int]]:
        """
        Generate ROIs for individual wire slots based on configuration
        
        Args:
            config: InspectionConfig object
            
        Returns:
            List of ROI dictionaries for each wire slot
        """
        rois = []
        for i in range(config.NUM_WIRES):
            roi = {
                'x': config.WIRE_SLOT_START_X + i * config.WIRE_SLOT_SPACING,
                'y': config.WIRE_SLOT_START_Y,
                'width': config.WIRE_SLOT_WIDTH,
                'height': config.WIRE_SLOT_HEIGHT
            }
            rois.append(roi)
        return rois
    
    @staticmethod
    def draw_roi(image: np.ndarray, roi: Dict[str, int], color: Tuple[int, int, int] = (0, 255, 0), 
                 thickness: int = 2, label: Optional[str] = None) -> np.ndarray:
        """
        Draw ROI rectangle on image for visualization
        
        Args:
            image: Input image
            roi: ROI dictionary
            color: BGR color for rectangle
            thickness: Line thickness
            label: Optional text label
            
        Returns:
            Image with ROI drawn
        """
        img_copy = image.copy()
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        
        if label:
            cv2.putText(img_copy, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img_copy


class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    @staticmethod
    def preprocess_for_wire_detection(image: np.ndarray, config) -> np.ndarray:
        """
        Preprocess image for wire presence detection
        
        Args:
            image: Input image (BGR or grayscale)
            config: InspectionConfig object
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        
        # Apply binary threshold
        _, binary = cv2.threshold(blurred, config.WIRE_PRESENCE_THRESHOLD, 
                                 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPHOLOGY_KERNEL_SIZE)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    @staticmethod
    def preprocess_for_edge_detection(image: np.ndarray, config) -> np.ndarray:
        """
        Preprocess image for edge-based analysis (insertion depth)
        
        Args:
            image: Input image
            config: InspectionConfig object
            
        Returns:
            Edge image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, config.CANNY_THRESHOLD_LOW, 
                         config.CANNY_THRESHOLD_HIGH)
        
        return edges
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE directly to grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced


class ColorAnalyzer:
    """Utilities for color-based wire identification"""
    
    @staticmethod
    def extract_dominant_color(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
        """
        Extract dominant color from an image region
        
        Args:
            image: Input BGR image
            mask: Optional binary mask
            
        Returns:
            Dominant color as (B, G, R) tuple
        """
        if mask is not None:
            pixels = image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            return (0, 0, 0)
        
        # Calculate mean color
        mean_color = np.mean(pixels, axis=0).astype(int)
        return tuple(mean_color)
    
    @staticmethod
    def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        Calculate Euclidean distance between two BGR colors
        
        Args:
            color1: First color (B, G, R)
            color2: Second color (B, G, R)
            
        Returns:
            Distance value
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))
    
    @staticmethod
    def match_wire_color(detected_color: Tuple[int, int, int], 
                        expected_colors: List[Tuple[str, List[int]]], 
                        tolerance: float) -> Tuple[Optional[str], float]:
        """
        Match detected color to expected wire colors
        
        Args:
            detected_color: Detected BGR color
            expected_colors: List of (name, [B, G, R]) tuples
            tolerance: Maximum distance for a match
            
        Returns:
            Tuple of (matched_color_name, confidence) or (None, 0.0)
        """
        min_distance = float('inf')
        best_match = None
        
        for color_name, expected_bgr in expected_colors:
            distance = ColorAnalyzer.color_distance(detected_color, tuple(expected_bgr))
            if distance < min_distance:
                min_distance = distance
                best_match = color_name
        
        # Calculate confidence (inverse of normalized distance)
        if min_distance <= tolerance:
            confidence = 1.0 - (min_distance / tolerance)
            return best_match, confidence
        else:
            return None, 0.0


class VisualizationHelper:
    """Helper functions for creating visualization outputs"""
    
    @staticmethod
    def create_result_overlay(image: np.ndarray, result, config) -> np.ndarray:
        """
        Create visualization overlay with inspection results
        
        Args:
            image: Original input image
            result: InspectionResult object
            config: InspectionConfig object
            
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        
        # Draw status banner
        status_color = (0, 255, 0) if result.status == "OK" else (0, 0, 255)
        cv2.rectangle(overlay, (10, 10), (300, 80), status_color, -1)
        cv2.putText(overlay, f"Status: {result.status}", (20, 45),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 2)
        cv2.putText(overlay, f"{result.processing_time_ms:.1f} ms", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw defect information
        if result.defects:
            y_offset = 100
            for defect in result.defects:
                cv2.putText(overlay, f"Defect: {defect}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        # Draw wire slot ROIs
        wire_rois = ROIExtractor.get_wire_slot_rois(config)
        for i, roi in enumerate(wire_rois):
            color = (0, 255, 0) if result.status == "OK" else (0, 165, 255)
            overlay = ROIExtractor.draw_roi(overlay, roi, color, 2, f"Wire {i+1}")
        
        # Draw connector ROI
        overlay = ROIExtractor.draw_roi(overlay, config.CONNECTOR_ROI, 
                                       (255, 0, 0), 2, "Connector")
        
        return overlay
    
    @staticmethod
    def create_debug_montage(images: List[Tuple[str, np.ndarray]], 
                            grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create a montage of debug images
        
        Args:
            images: List of (title, image) tuples
            grid_size: Optional (rows, cols) for grid layout
            
        Returns:
            Montage image
        """
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Determine grid size if not specified
        if grid_size is None:
            n = len(images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size
        
        # Get maximum dimensions
        max_h = max(img.shape[0] for _, img in images)
        max_w = max(img.shape[1] for _, img in images)
        
        # Create montage
        montage = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
        
        for idx, (title, img) in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Place image
            y_start = row * max_h
            x_start = col * max_w
            montage[y_start:y_start + img.shape[0], 
                   x_start:x_start + img.shape[1]] = img
            
            # Add title
            cv2.putText(montage, title, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return montage
