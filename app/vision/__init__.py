"""
Vision Processing Module
Contains all image processing and inspection logic.
"""

from .image_source import ImageSource, ImageSourceFactory, ImageFileSource, CameraSource
from .inspector import WireOrderInspector, InspectionPipeline, WireOrderResult
from .golden_sample import GoldenSample, GoldenSampleManager, TeachingMode, ROIConfig
from .roi_locator import TemplateLocalizer, PositionIndependentInspector
from .color_detector import WireColorDetector, WireColor
from .wire_segmenter import WireSegmentExtractor

__all__ = [
    'ImageSource', 'ImageSourceFactory', 'ImageFileSource', 'CameraSource',
    'WireOrderInspector', 'InspectionPipeline', 'WireOrderResult',
    'GoldenSample', 'GoldenSampleManager', 'TeachingMode', 'ROIConfig',
    'TemplateLocalizer', 'PositionIndependentInspector',
    'WireColorDetector', 'WireColor',
    'WireSegmentExtractor'
]
