"""
Application Controller
High-level application flow management.
Coordinates between UI, vision system, and behavior controller.
"""

import logging
from typing import Optional
import numpy as np

from ..controller.behavior_controller import BehaviorController, SystemState
from ..vision.golden_sample import GoldenSampleManager, GoldenSample
from ..vision.inspector import WireOrderInspector, InspectionPipeline
from ..vision.roi_locator import TemplateLocalizer, PositionIndependentInspector
from ..vision.image_source import ImageSource
from ..config.paths import DataPaths

logger = logging.getLogger(__name__)


class ApplicationController:
    """
    Main application controller.
    Manages application state, coordinates vision system and UI.
    """
    
    def __init__(self):
        """Initialize application controller."""
        # Ensure data directories exist
        DataPaths.ensure_directories()
        
        # Core components
        self.behavior_controller = BehaviorController()
        self.golden_sample_manager = GoldenSampleManager()
        
        # Vision system components (initialized when golden sample loaded)
        self.wire_inspector: Optional[WireOrderInspector] = None
        self.inspection_pipeline: Optional[InspectionPipeline] = None
        self.template_localizer: Optional[TemplateLocalizer] = None
        self.position_independent_inspector: Optional[PositionIndependentInspector] = None
        
        # Current image source
        self.current_source: Optional[ImageSource] = None
        
        logger.info("ApplicationController initialized")
    
    def initialize_vision_system(self, golden_sample: GoldenSample):
        """
        Initialize vision inspection system with golden sample.
        
        Args:
            golden_sample: Golden sample configuration
        """
        self.wire_inspector = WireOrderInspector(
            golden_sample=golden_sample,
            behavior_controller=self.behavior_controller
        )
        
        self.inspection_pipeline = InspectionPipeline(
            inspector=self.wire_inspector,
            behavior_controller=self.behavior_controller
        )
        
        # Initialize template-based localization
        self.template_localizer = TemplateLocalizer(confidence_threshold=0.7)
        self.position_independent_inspector = PositionIndependentInspector(
            wire_inspector=self.wire_inspector,
            template_localizer=self.template_localizer
        )
        
        logger.info(f"Vision system initialized with golden sample: {golden_sample.sample_name}")
    
    def load_golden_sample(self) -> Optional[GoldenSample]:
        """
        Load golden sample from storage.
        
        Returns:
            GoldenSample if found, None otherwise
        """
        golden_sample = self.golden_sample_manager.load_golden_sample()
        if golden_sample:
            self.initialize_vision_system(golden_sample)
            logger.info(f"Loaded golden sample: {golden_sample.sample_name}")
        return golden_sample
    
    def save_golden_sample(self, golden_sample: GoldenSample) -> bool:
        """
        Save golden sample to storage.
        
        Args:
            golden_sample: Golden sample to save
            
        Returns:
            True if successful
        """
        success = self.golden_sample_manager.save_golden_sample(golden_sample)
        if success:
            self.initialize_vision_system(golden_sample)
            logger.info(f"Saved golden sample: {golden_sample.sample_name}")
        return success
    
    def set_image_source(self, source: ImageSource):
        """Set current image source (camera or file)."""
        if self.current_source:
            self.current_source.release()
        self.current_source = source
        self.behavior_controller.set_source_configured(source.is_available())
    
    def release_image_source(self):
        """Release current image source."""
        if self.current_source:
            self.current_source.release()
            self.current_source = None
        self.behavior_controller.set_source_configured(False)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get frame from current source."""
        if self.current_source and self.current_source.is_available():
            return self.current_source.get_frame()
        return None
    
    def is_ready_for_inspection(self) -> bool:
        """Check if system is ready for inspection."""
        return (self.wire_inspector is not None and
                self.behavior_controller.is_golden_sample_loaded and
                self.current_source is not None and
                self.current_source.is_available())
    
    def shutdown(self):
        """Shutdown application controller."""
        logger.info("Shutting down ApplicationController")
        self.release_image_source()
        self.behavior_controller.shutdown()
