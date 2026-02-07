"""
Wire Order Inspector
Moved from wire_order_inspector.py to vision module.
Compares detected wire color order against golden sample reference.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import logging
import time

from .golden_sample import GoldenSample
from .color_detector import WireColorDetector
from .wire_segmenter import WireSegmentExtractor
from .image_source import ImageSource

logger = logging.getLogger(__name__)


@dataclass
class WireOrderResult:
    """Result of wire order inspection."""
    is_ok: bool
    detected_order: List[str]
    expected_order: List[str]
    defects: List[Dict]
    processing_time_ms: float = 0.0
    source_metadata: Dict = field(default_factory=dict)
    
    def get_defect_message(self) -> str:
        """Generate human-readable defect message."""
        if self.is_ok:
            return "OK - Wire order matches"
        
        messages = []
        for defect in self.defects:
            if 'index' in defect:
                idx = defect['index']
                exp = defect['expected']
                det = defect['detected']
                messages.append(f"Position {idx+1}: Expected {exp}, Got {det}")
            elif 'error' in defect:
                messages.append(defect['error'])
        
        return "NG - " + "; ".join(messages) if messages else "NG - Unknown error"
    
    def get_summary(self) -> str:
        """Get comprehensive result summary."""
        source_type = self.source_metadata.get('source_type', 'unknown')
        lines = [
            f"Source: {source_type}",
            f"Status: {'OK' if self.is_ok else 'NG'}",
            f"Processing Time: {self.processing_time_ms:.2f} ms",
            f"Detected: {self.detected_order}",
            f"Expected: {self.expected_order}"
        ]
        
        if not self.is_ok:
            lines.append(f"Defects: {self.get_defect_message()}")
        
        return "\n".join(lines)


class WireOrderInspector:
    """Wire color order inspector with automatic ROI localization."""
    
    def __init__(self, golden_sample: Optional[GoldenSample] = None,
                 behavior_controller: Optional['BehaviorController'] = None):
        """Initialize inspector."""
        self.golden_sample = golden_sample
        self.color_detector = WireColorDetector()
        self.segment_extractor = None
        self.behavior_controller = behavior_controller
        
        if golden_sample:
            self._initialize_extractor()
            if behavior_controller:
                behavior_controller.set_golden_sample_loaded(True)
    
    def set_golden_sample(self, golden_sample: GoldenSample):
        """Set or update golden sample reference."""
        self.golden_sample = golden_sample
        self._initialize_extractor()
        
        if self.behavior_controller:
            self.behavior_controller.set_golden_sample_loaded(True)
    
    def _initialize_extractor(self):
        """Initialize segment extractor."""
        if self.golden_sample:
            self.segment_extractor = WireSegmentExtractor(self.golden_sample.num_wires)
    
    def inspect(self, image: np.ndarray, source_metadata: Dict = None) -> WireOrderResult:
        """Inspect wire order in image with automatic ROI cropping."""
        start_time = time.perf_counter()
        
        if source_metadata is None:
            source_metadata = {'source_type': 'direct_array'}
        
        if self.golden_sample is None:
            logger.error("No golden sample configured")
            return WireOrderResult(
                is_ok=False, detected_order=[], expected_order=[],
                defects=[{"error": "No golden sample"}],
                processing_time_ms=0.0, source_metadata=source_metadata
            )
        
        # Validate and extract ROI
        roi_cfg = self.golden_sample.roi_config
        img_h, img_w = image.shape[:2]
        
        if (roi_cfg.wire_x < 0 or roi_cfg.wire_y < 0 or
            roi_cfg.wire_x + roi_cfg.wire_width > img_w or
            roi_cfg.wire_y + roi_cfg.wire_height > img_h):
            logger.error(f"ROI out of bounds: image=({img_w}x{img_h})")
            return WireOrderResult(
                is_ok=False, detected_order=[], expected_order=[],
                defects=[{"error": "ROI out of image bounds"}],
                processing_time_ms=0.0, source_metadata=source_metadata
            )
        
        # Extract wire ROI
        wire_roi = image[
            roi_cfg.wire_y:roi_cfg.wire_y + roi_cfg.wire_height,
            roi_cfg.wire_x:roi_cfg.wire_x + roi_cfg.wire_width
        ]
        
        # Split and detect
        segments = self.segment_extractor.extract_segments(wire_roi)
        detected_colors = []
        
        for i, segment in enumerate(segments):
            color, confidence = self.color_detector.detect_color(segment)
            detected_colors.append(color.value)
        
        # Compare with expected
        expected_order = self.golden_sample.wire_color_order
        defects = []
        
        for i, (expected, detected) in enumerate(zip(expected_order, detected_colors)):
            if expected != detected:
                defects.append({'index': i, 'expected': expected, 'detected': detected})
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return WireOrderResult(
            is_ok=len(defects) == 0,
            detected_order=detected_colors,
            expected_order=expected_order,
            defects=defects,
            processing_time_ms=processing_time,
            source_metadata=source_metadata
        )
    
    def inspect_from_source(self, source: ImageSource) -> WireOrderResult:
        """Inspect from image source with state validation."""
        # ...existing validation code...
        
        if not source.is_available():
            logger.error(f"Source not available: {source.get_source_info()}")
            return WireOrderResult(
                is_ok=False, detected_order=[], expected_order=[],
                defects=[{"error": "Source not available"}],
                processing_time_ms=0.0, source_metadata=source.get_metadata()
            )
        
        frame = source.get_frame()
        if frame is None:
            logger.error(f"Failed to get frame from: {source.get_source_info()}")
            return WireOrderResult(
                is_ok=False, detected_order=[], expected_order=[],
                defects=[{"error": "Failed to get frame"}],
                processing_time_ms=0.0, source_metadata=source.get_metadata()
            )
        
        return self.inspect(frame, source_metadata=source.get_metadata())
    
    def get_wire_segments_coordinates(self) -> List[Tuple[int, int, int, int]]:
        """Get absolute coordinates of wire segments for visualization."""
        if not self.golden_sample or not self.segment_extractor:
            return []
        
        roi_cfg = self.golden_sample.roi_config
        return self.segment_extractor.get_segment_coordinates(
            roi_cfg.wire_x, roi_cfg.wire_y,
            roi_cfg.wire_width, roi_cfg.wire_height
        )


class InspectionPipeline:
    """High-level inspection pipeline with state management."""
    
    def __init__(self, inspector: WireOrderInspector, 
                 behavior_controller: Optional['BehaviorController'] = None):
        """Initialize pipeline."""
        self.inspector = inspector
        self.behavior_controller = behavior_controller
        
        if behavior_controller and not inspector.behavior_controller:
            inspector.behavior_controller = behavior_controller
