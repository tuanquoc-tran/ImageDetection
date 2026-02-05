"""
Defect detection modules for AOI connector inspection
Each inspector class handles a specific type of defect detection
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from image_utils import ROIExtractor, ImagePreprocessor, ColorAnalyzer


class WirePresenceInspector:
    """Detects missing wires in connector slots"""
    
    def __init__(self, config):
        self.config = config
        
    def inspect(self, image: np.ndarray) -> Tuple[bool, List[int], Dict]:
        """
        Check for missing wires in connector
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_ok, missing_wire_indices, details)
        """
        wire_rois = ROIExtractor.get_wire_slot_rois(self.config)
        missing_wires = []
        wire_presence = []
        
        for i, roi in enumerate(wire_rois):
            # Extract wire slot ROI
            wire_region = ROIExtractor.extract_roi(image, roi)
            
            # Preprocess for wire detection
            binary = ImagePreprocessor.preprocess_for_wire_detection(
                wire_region, self.config
            )
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any significant contour exists
            max_area = 0
            if contours:
                max_area = max(cv2.contourArea(c) for c in contours)
            
            is_present = max_area >= self.config.WIRE_PRESENCE_MIN_AREA
            wire_presence.append(is_present)
            
            if not is_present:
                missing_wires.append(i)
        
        details = {
            'wire_presence': wire_presence,
            'missing_wire_slots': missing_wires
        }
        
        is_ok = len(missing_wires) == 0
        return is_ok, missing_wires, details


class WireOrderInspector:
    """Detects incorrect wire order based on color"""
    
    def __init__(self, config):
        self.config = config
        
    def inspect(self, image: np.ndarray) -> Tuple[bool, List[int], Dict]:
        """
        Check if wires are in correct color order
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_ok, incorrect_positions, details)
        """
        wire_rois = ROIExtractor.get_wire_slot_rois(self.config)
        incorrect_positions = []
        detected_colors = []
        expected_colors = []
        confidences = []
        
        for i, roi in enumerate(wire_rois):
            # Extract wire slot ROI
            wire_region = ROIExtractor.extract_roi(image, roi)
            
            # Get binary mask for wire region
            binary = ImagePreprocessor.preprocess_for_wire_detection(
                wire_region, self.config
            )
            
            # Extract dominant color from wire region
            detected_color = ColorAnalyzer.extract_dominant_color(
                wire_region, binary
            )
            
            # Match to expected color
            expected_color_name, expected_bgr = self.config.WIRE_COLORS[i]
            matched_color, confidence = ColorAnalyzer.match_wire_color(
                detected_color,
                self.config.WIRE_COLORS,
                self.config.COLOR_MATCH_TOLERANCE
            )
            
            detected_colors.append(detected_color)
            expected_colors.append(expected_color_name)
            confidences.append(confidence)
            
            # Check if color matches expected position
            if (matched_color != expected_color_name or 
                confidence < self.config.COLOR_MATCH_MIN_CONFIDENCE):
                incorrect_positions.append(i)
        
        details = {
            'detected_colors': detected_colors,
            'expected_colors': expected_colors,
            'confidences': confidences,
            'incorrect_positions': incorrect_positions
        }
        
        is_ok = len(incorrect_positions) == 0
        return is_ok, incorrect_positions, details


class ConnectorPositionInspector:
    """Detects connector position offset"""
    
    def __init__(self, config):
        self.config = config
        self.reference_template = None
        
    def set_reference_template(self, template: np.ndarray):
        """
        Set reference template for position matching
        
        Args:
            template: Reference connector image
        """
        self.reference_template = template
        
    def inspect(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int], Dict]:
        """
        Check if connector is properly positioned
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_ok, (offset_x, offset_y), details)
        """
        # Extract connector ROI
        connector_roi = ROIExtractor.extract_roi(image, self.config.CONNECTOR_ROI)
        
        if self.reference_template is not None:
            # Use template matching for precise position detection
            offset_x, offset_y, match_score = self._template_match_position(
                connector_roi, self.reference_template
            )
        else:
            # Use centroid-based position detection
            offset_x, offset_y, match_score = self._centroid_position(
                connector_roi
            )
        
        # Check if offset exceeds threshold
        is_ok = (abs(offset_x) <= self.config.POSITION_MAX_OFFSET_X and 
                abs(offset_y) <= self.config.POSITION_MAX_OFFSET_Y)
        
        details = {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'match_score': match_score,
            'max_offset_x': self.config.POSITION_MAX_OFFSET_X,
            'max_offset_y': self.config.POSITION_MAX_OFFSET_Y
        }
        
        return is_ok, (offset_x, offset_y), details
    
    def _template_match_position(self, roi: np.ndarray, 
                                template: np.ndarray) -> Tuple[int, int, float]:
        """
        Detect position using template matching
        
        Args:
            roi: Connector ROI
            template: Reference template
            
        Returns:
            Tuple of (offset_x, offset_y, match_score)
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        
        # Perform template matching
        result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Calculate offset from expected center
        roi_center = (roi.shape[1] // 2, roi.shape[0] // 2)
        template_center = (template.shape[1] // 2, template.shape[0] // 2)
        matched_center = (max_loc[0] + template_center[0], 
                         max_loc[1] + template_center[1])
        
        offset_x = matched_center[0] - roi_center[0]
        offset_y = matched_center[1] - roi_center[1]
        
        return offset_x, offset_y, max_val
    
    def _centroid_position(self, roi: np.ndarray) -> Tuple[int, int, float]:
        """
        Detect position using centroid calculation
        
        Args:
            roi: Connector ROI
            
        Returns:
            Tuple of (offset_x, offset_y, confidence)
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Threshold and find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, 0.0
        
        # Find largest contour (connector body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return 0, 0, 0.0
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Calculate offset from ROI center
        roi_center = (roi.shape[1] // 2, roi.shape[0] // 2)
        offset_x = cx - roi_center[0]
        offset_y = cy - roi_center[1]
        
        # Confidence based on contour quality
        area = cv2.contourArea(largest_contour)
        bbox_area = roi.shape[0] * roi.shape[1]
        confidence = min(area / bbox_area, 1.0)
        
        return offset_x, offset_y, confidence


class WireInsertionInspector:
    """Detects insufficient wire insertion depth"""
    
    def __init__(self, config):
        self.config = config
        
    def inspect(self, image: np.ndarray) -> Tuple[bool, List[int], Dict]:
        """
        Check if wires are fully inserted
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_ok, poorly_inserted_indices, details)
        """
        wire_rois = ROIExtractor.get_wire_slot_rois(self.config)
        poorly_inserted = []
        insertion_depths = []
        
        for i, roi in enumerate(wire_rois):
            # Extract wire slot ROI
            wire_region = ROIExtractor.extract_roi(image, roi)
            
            # Measure insertion depth
            depth = self._measure_insertion_depth(wire_region)
            insertion_depths.append(depth)
            
            # Check if depth is sufficient
            min_depth = self.config.INSERTION_MIN_DEPTH
            tolerance = self.config.INSERTION_DEPTH_TOLERANCE
            
            if depth < (min_depth - tolerance):
                poorly_inserted.append(i)
        
        details = {
            'insertion_depths': insertion_depths,
            'min_required_depth': self.config.INSERTION_MIN_DEPTH,
            'poorly_inserted_slots': poorly_inserted
        }
        
        is_ok = len(poorly_inserted) == 0
        return is_ok, poorly_inserted, details
    
    def _measure_insertion_depth(self, wire_region: np.ndarray) -> float:
        """
        Measure wire insertion depth using edge detection
        
        Args:
            wire_region: ROI containing single wire slot
            
        Returns:
            Measured insertion depth in pixels
        """
        # Get edges
        edges = ImagePreprocessor.preprocess_for_edge_detection(
            wire_region, self.config
        )
        
        # Project edges vertically to find wire length
        vertical_projection = np.sum(edges, axis=1)
        
        # Find first and last significant edge rows
        threshold = np.max(vertical_projection) * 0.1
        significant_rows = np.where(vertical_projection > threshold)[0]
        
        if len(significant_rows) == 0:
            return 0.0
        
        # Calculate depth as span of significant edges
        depth = significant_rows[-1] - significant_rows[0]
        
        return float(depth)


class InspectorFactory:
    """Factory class to create and manage inspectors"""
    
    @staticmethod
    def create_all_inspectors(config):
        """
        Create all inspector instances
        
        Args:
            config: InspectionConfig object
            
        Returns:
            Dictionary of inspector instances
        """
        inspectors = {
            'wire_presence': WirePresenceInspector(config),
            'wire_order': WireOrderInspector(config),
            'connector_position': ConnectorPositionInspector(config),
            'wire_insertion': WireInsertionInspector(config)
        }
        return inspectors
    
    @staticmethod
    def load_reference_images(inspectors: Dict, config):
        """
        Load reference images for inspectors that need them
        
        Args:
            inspectors: Dictionary of inspector instances
            config: InspectionConfig object
        """
        # Load reference template for position inspection
        if config.REFERENCE_CONNECTOR_PATH:
            try:
                ref_image = cv2.imread(config.REFERENCE_CONNECTOR_PATH)
                if ref_image is not None:
                    ref_roi = ROIExtractor.extract_roi(ref_image, 
                                                       config.CONNECTOR_ROI)
                    inspectors['connector_position'].set_reference_template(ref_roi)
            except Exception as e:
                print(f"Warning: Could not load reference image: {e}")
