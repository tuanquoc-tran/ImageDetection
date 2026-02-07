"""
Template-Based ROI Localization
Moved from template_localization.py to vision module.
Automatically locates inspection ROI using template matching.
Position-independent inspection for shifted/rotated connectors.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemplateMatchResult:
    """Result of template matching operation."""
    found: bool
    match_x: int
    match_y: int
    confidence: float
    offset_x: int
    offset_y: int
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.found:
            return (f"Template found at ({self.match_x}, {self.match_y}), "
                   f"confidence: {self.confidence:.3f}, "
                   f"offset: ({self.offset_x}, {self.offset_y})")
        return "Template not found"


class TemplateLocalizer:
    """Template-based ROI localization for position-independent inspection."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize template localizer."""
        self.confidence_threshold = confidence_threshold
        self.template = None
        self.template_center_x = 0
        self.template_center_y = 0
        self.golden_roi_offset_x = 0
        self.golden_roi_offset_y = 0
        
        logger.info(f"TemplateLocalizer initialized (threshold: {confidence_threshold})")
    
    def set_golden_template(self, image: np.ndarray, 
                           template_x: int, template_y: int,
                           template_width: int, template_height: int,
                           roi_x: int, roi_y: int):
        """Set golden sample template and ROI relationship."""
        # Extract template from golden image
        self.template = image[
            template_y:template_y + template_height,
            template_x:template_x + template_width
        ].copy()
        
        # Store template center
        self.template_center_x = template_x + template_width // 2
        self.template_center_y = template_y + template_height // 2
        
        # Calculate offset from template center to ROI
        self.golden_roi_offset_x = roi_x - self.template_center_x
        self.golden_roi_offset_y = roi_y - self.template_center_y
        
        logger.info(
            f"Golden template set: size=({template_width}x{template_height}), "
            f"center=({self.template_center_x},{self.template_center_y}), "
            f"ROI offset=({self.golden_roi_offset_x},{self.golden_roi_offset_y})"
        )
    
    def locate_roi(self, image: np.ndarray, 
                   roi_width: int, roi_height: int) -> Tuple[Optional[Tuple[int, int, int, int]], TemplateMatchResult]:
        """Locate inspection ROI using template matching."""
        if self.template is None:
            logger.error("No golden template set")
            return None, TemplateMatchResult(
                found=False, match_x=0, match_y=0, 
                confidence=0.0, offset_x=0, offset_y=0
            )
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY) if len(self.template.shape) == 3 else self.template
        
        # Template matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        match_x, match_y = max_loc
        confidence = float(max_val)
        
        logger.debug(f"Template match: position=({match_x},{match_y}), confidence={confidence:.3f}")
        
        # Check confidence
        if confidence < self.confidence_threshold:
            logger.warning(f"Template match confidence {confidence:.3f} below threshold {self.confidence_threshold}")
            return None, TemplateMatchResult(
                found=False, match_x=match_x, match_y=match_y,
                confidence=confidence, offset_x=0, offset_y=0
            )
        
        # Calculate matched center
        template_h, template_w = gray_template.shape[:2]
        matched_center_x = match_x + template_w // 2
        matched_center_y = match_y + template_h // 2
        
        # Calculate offset
        offset_x = matched_center_x - self.template_center_x
        offset_y = matched_center_y - self.template_center_y
        
        # Calculate ROI
        roi_x = matched_center_x + self.golden_roi_offset_x
        roi_y = matched_center_y + self.golden_roi_offset_y
        
        # Validate bounds
        img_h, img_w = gray_image.shape[:2]
        if (roi_x < 0 or roi_y < 0 or
            roi_x + roi_width > img_w or
            roi_y + roi_height > img_h):
            logger.error(f"Computed ROI out of bounds")
            return None, TemplateMatchResult(
                found=False, match_x=match_x, match_y=match_y,
                confidence=confidence, offset_x=offset_x, offset_y=offset_y
            )
        
        logger.info(f"ROI located: ({roi_x},{roi_y},{roi_width}x{roi_height}), confidence: {confidence:.3f}")
        
        return (roi_x, roi_y, roi_width, roi_height), TemplateMatchResult(
            found=True, match_x=match_x, match_y=match_y,
            confidence=confidence, offset_x=offset_x, offset_y=offset_y
        )
    
    def visualize_match(self, image: np.ndarray, 
                       match_result: TemplateMatchResult,
                       roi_coords: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Visualize template match and ROI on image."""
        vis_image = image.copy()
        
        if not match_result.found:
            cv2.putText(vis_image, "Template Not Found", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis_image
        
        # Draw template location
        if self.template is not None:
            template_h, template_w = self.template.shape[:2]
            cv2.rectangle(vis_image,
                         (match_result.match_x, match_result.match_y),
                         (match_result.match_x + template_w, match_result.match_y + template_h),
                         (255, 0, 255), 2)
            
            cv2.putText(vis_image, f"Template: {match_result.confidence:.2f}",
                       (match_result.match_x, match_result.match_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw ROI
        if roi_coords:
            x, y, w, h = roi_coords
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(vis_image, f"Auto ROI (offset: {match_result.offset_x},{match_result.offset_y})",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
    
    def get_template_image(self) -> Optional[np.ndarray]:
        """Get stored template image."""
        return self.template.copy() if self.template is not None else None


class PositionIndependentInspector:
    """Wrapper for inspection with automatic ROI localization."""
    
    def __init__(self, wire_inspector: 'WireOrderInspector',
                 template_localizer: TemplateLocalizer):
        """Initialize position-independent inspector."""
        self.wire_inspector = wire_inspector
        self.localizer = template_localizer
    
    def inspect_with_localization(self, image: np.ndarray) -> Tuple['WireOrderResult', TemplateMatchResult]:
        """Perform inspection with automatic ROI localization."""
        import time
        start_time = time.perf_counter()
        
        # Get expected ROI dimensions
        golden_sample = self.wire_inspector.golden_sample
        if not golden_sample:
            logger.error("No golden sample configured")
            from .inspector import WireOrderResult
            return WireOrderResult(
                is_ok=False, detected_order=[], expected_order=[],
                defects=[{"error": "No golden sample"}],
                processing_time_ms=0.0
            ), TemplateMatchResult(False, 0, 0, 0.0, 0, 0)
        
        roi_cfg = golden_sample.roi_config
        
        # Locate ROI
        roi_coords, match_result = self.localizer.locate_roi(
            image, roi_cfg.wire_width, roi_cfg.wire_height
        )
        
        if not match_result.found or roi_coords is None:
            logger.error(f"ROI localization failed: {match_result.get_summary()}")
            from .inspector import WireOrderResult
            return WireOrderResult(
                is_ok=False, detected_order=[], expected_order=[],
                defects=[{"error": f"ROI localization failed (confidence: {match_result.confidence:.3f})"}],
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            ), match_result
        
        # Temporarily override ROI
        roi_x, roi_y, roi_w, roi_h = roi_coords
        original_roi = (roi_cfg.wire_x, roi_cfg.wire_y)
        roi_cfg.wire_x = roi_x
        roi_cfg.wire_y = roi_y
        
        try:
            result = self.wire_inspector.inspect(image)
        finally:
            roi_cfg.wire_x, roi_cfg.wire_y = original_roi
        
        # Add localization info
        result.source_metadata['template_match'] = {
            'confidence': match_result.confidence,
            'offset_x': match_result.offset_x,
            'offset_y': match_result.offset_y,
            'roi_x': roi_x,
            'roi_y': roi_y
        }
        
        return result, match_result
