"""
ROI Editor Widget
Moved from roi_manager.py to ui module.
Interactive ROI selection and management for PyQt6.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtWidgets import QLabel, QFrame
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
import logging

logger = logging.getLogger(__name__)


class ROIMode(Enum):
    """ROI operation modes"""
    TEACHING = "teaching"
    INSPECTION = "inspection"
    VERIFICATION = "verification"


@dataclass
class ROI:
    """Region of Interest data structure"""
    x: int
    y: int
    width: int
    height: int
    name: str = ""
    mode: ROIMode = ROIMode.TEACHING
    color: Tuple[int, int, int] = (0, 255, 0)
    metadata: dict = field(default_factory=dict)
    
    def to_rect(self) -> QRect:
        """Convert to QRect"""
        return QRect(self.x, self.y, self.width, self.height)
    
    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract ROI region from image"""
        return image[self.y:self.y+self.height, self.x:self.x+self.width].copy()
    
    def is_valid(self, image_shape: Tuple[int, int]) -> bool:
        """Check if ROI is within image bounds"""
        h, w = image_shape[:2]
        return (self.x >= 0 and self.y >= 0 and 
                self.x + self.width <= w and 
                self.y + self.height <= h and
                self.width > 0 and self.height > 0)


class InteractiveROIEditor(QLabel):
    """Interactive ROI editor widget."""
    
    roi_changed = pyqtSignal(ROI)
    
    def __init__(self, multi_roi: bool = False):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.multi_roi = multi_roi
        self.rois: List[ROI] = []
        self.active_roi_index: Optional[int] = None
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.temp_rect = None
        
        # Display
        self.current_image = None
        self.scale_factor = 1.0
        
        self.setMouseTracking(True)
        
        logger.debug("InteractiveROIEditor initialized")
    
    def set_image(self, image: np.ndarray):
        """Update displayed image"""
        self.current_image = image.copy()
        self.update_display()
    
    def update_display(self):
        """Refresh display with current image and ROIs"""
        if self.current_image is None:
            return
        
        height, width = self.current_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(self.current_image.data, width, height, 
                       bytes_per_line, QImage.Format.Format_BGR888)
        
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), 
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        
        self.scale_factor = scaled_pixmap.width() / width
        
        # Draw ROIs
        painter = QPainter(scaled_pixmap)
        
        for idx, roi in enumerate(self.rois):
            is_active = idx == self.active_roi_index
            self._draw_roi(painter, roi, is_active)
        
        # Draw temporary rectangle
        if self.drawing and self.temp_rect:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine))
            scaled_rect = self._scale_rect(self.temp_rect)
            painter.drawRect(scaled_rect)
        
        painter.end()
        self.setPixmap(scaled_pixmap)
    
    def _draw_roi(self, painter: QPainter, roi: ROI, is_active: bool):
        """Draw single ROI"""
        scaled_rect = self._scale_rect(roi.to_rect())
        
        color = QColor(0, 255, 0) if is_active else QColor(100, 100, 100)
        pen_style = Qt.PenStyle.SolidLine if is_active else Qt.PenStyle.DashLine
        
        painter.setPen(QPen(color, 2 if is_active else 1, pen_style))
        painter.drawRect(scaled_rect)
    
    def _scale_rect(self, rect: QRect) -> QRect:
        """Scale rectangle to display coordinates"""
        return QRect(
            int(rect.x() * self.scale_factor),
            int(rect.y() * self.scale_factor),
            int(rect.width() * self.scale_factor),
            int(rect.height() * self.scale_factor)
        )
    
    def _unscale_point(self, point: QPoint) -> QPoint:
        """Convert display to image coordinates"""
        pixmap = self.pixmap()
        if pixmap:
            offset_x = (self.width() - pixmap.width()) // 2
            offset_y = (self.height() - pixmap.height()) // 2
            point = QPoint(point.x() - offset_x, point.y() - offset_y)
        
        return QPoint(int(point.x() / self.scale_factor),
                     int(point.y() / self.scale_factor))
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if self.current_image is None:
            return
        
        img_pos = self._unscale_point(event.pos())
        self.drawing = True
        self.start_point = img_pos
        self.temp_rect = QRect(img_pos, img_pos)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move"""
        if self.current_image is None:
            return
        
        img_pos = self._unscale_point(event.pos())
        
        # Clamp to bounds
        h, w = self.current_image.shape[:2]
        img_pos.setX(max(0, min(w, img_pos.x())))
        img_pos.setY(max(0, min(h, img_pos.y())))
        
        if self.drawing:
            self.temp_rect = QRect(self.start_point, img_pos).normalized()
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.drawing:
            self.drawing = False
            if self.temp_rect and self.temp_rect.width() > 5 and self.temp_rect.height() > 5:
                new_roi = ROI(
                    x=self.temp_rect.x(),
                    y=self.temp_rect.y(),
                    width=self.temp_rect.width(),
                    height=self.temp_rect.height()
                )
                
                if not self.multi_roi:
                    self.rois.clear()
                
                self.rois.append(new_roi)
                self.active_roi_index = len(self.rois) - 1
                
                self.roi_changed.emit(new_roi)
            
            self.temp_rect = None
            self.update_display()
    
    def add_roi(self, roi: ROI):
        """Add ROI programmatically"""
        if not self.multi_roi:
            self.rois.clear()
        
        self.rois.append(roi)
        self.active_roi_index = len(self.rois) - 1
        self.update_display()
    
    def get_active_roi(self) -> Optional[ROI]:
        """Get currently active ROI"""
        if self.active_roi_index is not None and self.active_roi_index < len(self.rois):
            return self.rois[self.active_roi_index]
        return None
    
    def clear_rois(self):
        """Clear all ROIs"""
        self.rois.clear()
        self.active_roi_index = None
        self.update_display()
