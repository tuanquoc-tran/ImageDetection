"""
Golden Sample Management
Moved from root to vision module.
Handles teaching mode, ROI storage, and wire order reference.
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from ..config.paths import get_config_dir

logger = logging.getLogger(__name__)


@dataclass
class ROIConfig:
    """ROI configuration for connector and wire areas."""
    connector_x: int
    connector_y: int
    connector_width: int
    connector_height: int
    wire_x: int
    wire_y: int
    wire_width: int
    wire_height: int


@dataclass
class GoldenSample:
    """Golden sample configuration and reference data."""
    sample_name: str
    creation_date: str
    num_wires: int
    roi_config: ROIConfig
    wire_color_order: List[str]
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['roi_config'] = asdict(self.roi_config)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GoldenSample':
        """Create from dictionary."""
        roi_data = data['roi_config']
        roi_config = ROIConfig(**roi_data)
        
        return cls(
            sample_name=data['sample_name'],
            creation_date=data['creation_date'],
            num_wires=data['num_wires'],
            roi_config=roi_config,
            wire_color_order=data['wire_color_order'],
            notes=data.get('notes', '')
        )


class GoldenSampleManager:
    """Manages golden sample storage and retrieval."""
    
    DEFAULT_CONFIG_FILE = "golden_sample.json"
    
    def __init__(self):
        """Initialize manager."""
        self.config_dir = get_config_dir()
    
    def save_golden_sample(self, golden_sample: GoldenSample, 
                          filename: str = None) -> bool:
        """Save golden sample to JSON file."""
        filename = filename or self.DEFAULT_CONFIG_FILE
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(golden_sample.to_dict(), f, indent=2)
            logger.info(f"Golden sample saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save golden sample: {e}")
            return False
    
    def load_golden_sample(self, filename: str = None) -> Optional[GoldenSample]:
        """Load golden sample from JSON file."""
        filename = filename or self.DEFAULT_CONFIG_FILE
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Golden sample file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            golden_sample = GoldenSample.from_dict(data)
            logger.info(f"Golden sample loaded from {filepath}")
            return golden_sample
        except Exception as e:
            logger.error(f"Failed to load golden sample: {e}")
            return None
    
    def list_golden_samples(self) -> List[str]:
        """List all available golden sample files."""
        try:
            files = [f.name for f in self.config_dir.glob('*.json')]
            return files
        except Exception as e:
            logger.error(f"Failed to list golden samples: {e}")
            return []


class TeachingMode:
    """Teaching mode for golden sample creation."""
    
    def __init__(self):
        """Initialize teaching mode."""
        # Import here to avoid circular dependency
        from .color_detector import WireColorDetector
        from .wire_segmenter import WireSegmentExtractor
        
        self.color_detector = WireColorDetector()
        self.WireSegmentExtractor = WireSegmentExtractor
    
    def create_golden_sample(self, 
                            image: 'np.ndarray',
                            sample_name: str,
                            roi_config: ROIConfig,
                            num_wires: int,
                            notes: str = "") -> GoldenSample:
        """Create golden sample from image and ROI configuration."""
        import cv2
        
        # Create segment extractor
        segment_extractor = self.WireSegmentExtractor(num_wires)
        
        # Extract wire ROI
        wire_roi = image[
            roi_config.wire_y:roi_config.wire_y + roi_config.wire_height,
            roi_config.wire_x:roi_config.wire_x + roi_config.wire_width
        ]
        
        # Split into segments
        segments = segment_extractor.extract_segments(wire_roi)
        
        # Detect color for each wire
        wire_colors = []
        for i, segment in enumerate(segments):
            color, confidence = self.color_detector.detect_color(segment)
            logger.info(f"Wire {i}: {color.value} (confidence: {confidence:.2f})")
            wire_colors.append(color.value)
        
        # Create golden sample
        golden_sample = GoldenSample(
            sample_name=sample_name,
            creation_date=datetime.now().isoformat(),
            num_wires=num_wires,
            roi_config=roi_config,
            wire_color_order=wire_colors,
            notes=notes
        )
        
        return golden_sample
