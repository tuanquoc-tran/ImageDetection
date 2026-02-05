"""
ROI Manager Module - Reusable ROI selection and management
Supports both sample teaching and inspection modes
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QObject
from PyQt6.QtWidgets import QLabel, QFrame
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush


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
    color: Tuple[int, int, int] = (0, 255, 0)  # RGB
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'name': self.name,
            'mode': self.mode.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ROI':
        """Create from dictionary"""
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            name=data.get('name', ''),
            mode=ROIMode(data.get('mode', 'teaching')),
            metadata=data.get('metadata', {})
        )
    
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


class ROICollection:
    """Manages multiple ROIs"""
    
    def __init__(self):
        self.rois: List[ROI] = []
        self.active_roi_index: Optional[int] = None
        
    def add(self, roi: ROI) -> int:
        """Add ROI and return its index"""
        self.rois.append(roi)
        return len(self.rois) - 1
    
    def remove(self, index: int) -> bool:
        """Remove ROI by index"""
        if 0 <= index < len(self.rois):
            self.rois.pop(index)
            if self.active_roi_index == index:
                self.active_roi_index = None
            elif self.active_roi_index is not None and self.active_roi_index > index:
                self.active_roi_index -= 1
            return True
        return False
    
    def get(self, index: int) -> Optional[ROI]:
        """Get ROI by index"""
        if 0 <= index < len(self.rois):
            return self.rois[index]
        return None
    
    def get_active(self) -> Optional[ROI]:
        """Get currently active ROI"""
        if self.active_roi_index is not None:
            return self.get(self.active_roi_index)
        return None
    
    def set_active(self, index: Optional[int]):
        """Set active ROI index"""
        self.active_roi_index = index
    
    def clear(self):
        """Remove all ROIs"""
        self.rois.clear()
        self.active_roi_index = None
    
    def count(self) -> int:
        """Get number of ROIs"""
        return len(self.rois)
    
    def get_by_name(self, name: str) -> Optional[ROI]:
        """Get ROI by name"""
        for roi in self.rois:
            if roi.name == name:
                return roi
        return None
    
    def get_by_mode(self, mode: ROIMode) -> List[ROI]:
        """Get all ROIs with specific mode"""
        return [roi for roi in self.rois if roi.mode == mode]


class InteractiveROIEditor(QLabel):
    """Interactive ROI editor widget - reusable for any mode"""
    
    roi_changed = pyqtSignal(ROI)
    roi_selected = pyqtSignal(int)  # ROI index
    
    def __init__(self, multi_roi: bool = False):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # ROI management
        self.multi_roi = multi_roi
        self.roi_collection = ROICollection()
        
        # Current drawing state
        self.drawing = False
        self.dragging = False
        self.resizing = False
        self.start_point = None
        self.resize_corner = None
        self.drag_offset = QPoint()
        self.temp_rect = None  # Temporary rectangle during drawing
        
        # Display state
        self.current_image = None
        self.scale_factor = 1.0
        
        # Visualization settings
        self.show_labels = True
        self.show_handles = True
        self.active_color = QColor(0, 255, 0)
        self.inactive_color = QColor(100, 100, 100)
        
        self.setMouseTracking(True)
        
    def set_image(self, image: np.ndarray):
        """Update displayed image"""
        self.current_image = image.copy()
        self.update_display()
        
    def update_display(self):
        """Refresh the display with current image and ROIs"""
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
        
        # Draw all ROIs
        painter = QPainter(scaled_pixmap)
        
        # Draw inactive ROIs first
        for idx, roi in enumerate(self.roi_collection.rois):
            if idx != self.roi_collection.active_roi_index:
                self._draw_roi(painter, roi, False)
        
        # Draw active ROI on top
        active_roi = self.roi_collection.get_active()
        if active_roi:
            self._draw_roi(painter, active_roi, True)
        
        # Draw temporary rectangle during drawing
        if self.drawing and self.temp_rect:
            painter.setPen(QPen(self.active_color, 2, Qt.PenStyle.DashLine))
            scaled_rect = self._scale_rect(self.temp_rect)
            painter.drawRect(scaled_rect)
        
        painter.end()
        self.setPixmap(scaled_pixmap)
        
    def _draw_roi(self, painter: QPainter, roi: ROI, is_active: bool):
        """Draw a single ROI"""
        scaled_rect = self._scale_rect(roi.to_rect())
        
        color = self.active_color if is_active else self.inactive_color
        pen_style = Qt.PenStyle.SolidLine if is_active else Qt.PenStyle.DashLine
        
        painter.setPen(QPen(color, 2 if is_active else 1, pen_style))
        painter.drawRect(scaled_rect)
        
        # Draw handles for active ROI
        if is_active and self.show_handles:
            handle_size = 8
            painter.setBrush(QBrush(color))
            corners = [
                scaled_rect.topLeft(),
                scaled_rect.topRight(),
                scaled_rect.bottomLeft(),
                scaled_rect.bottomRight()
            ]
            for corner in corners:
                painter.drawRect(corner.x() - handle_size//2, 
                               corner.y() - handle_size//2,
                               handle_size, handle_size)
        
        # Draw label
        if self.show_labels and roi.name:
            painter.setPen(QPen(color))
            painter.drawText(scaled_rect.topLeft() + QPoint(5, -5), roi.name)
    
    def _scale_rect(self, rect: QRect) -> QRect:
        """Scale rectangle to display coordinates"""
        return QRect(
            int(rect.x() * self.scale_factor),
            int(rect.y() * self.scale_factor),
            int(rect.width() * self.scale_factor),
            int(rect.height() * self.scale_factor)
        )
    
    def _unscale_point(self, point: QPoint) -> QPoint:
        """Convert display coordinates to image coordinates"""
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
        display_pos = event.pos()
        
        # Check if clicking on active ROI
        active_roi = self.roi_collection.get_active()
        if active_roi:
            # Check resize handles
            corner = self._get_corner_at_pos(active_roi, display_pos)
            if corner is not None:
                self.resizing = True
                self.resize_corner = corner
                return
            
            # Check if clicking inside ROI for dragging
            if self._is_point_in_roi(active_roi, display_pos):
                self.dragging = True
                self.drag_offset = QPoint(img_pos.x() - active_roi.x,
                                         img_pos.y() - active_roi.y)
                return
        
        # Check if clicking on other ROIs
        if self.multi_roi:
            for idx, roi in enumerate(self.roi_collection.rois):
                if self._is_point_in_roi(roi, display_pos):
                    self.roi_collection.set_active(idx)
                    self.roi_selected.emit(idx)
                    self.update_display()
                    return
        
        # Start drawing new ROI
        self.drawing = True
        self.start_point = img_pos
        self.temp_rect = QRect(img_pos, img_pos)
        
    def mouseMoveEvent(self, event):
        """Handle mouse move"""
        if self.current_image is None:
            return
        
        img_pos = self._unscale_point(event.pos())
        
        # Clamp to image bounds
        h, w = self.current_image.shape[:2]
        img_pos.setX(max(0, min(w, img_pos.x())))
        img_pos.setY(max(0, min(h, img_pos.y())))
        
        if self.drawing:
            self.temp_rect = QRect(self.start_point, img_pos).normalized()
            self.update_display()
        elif self.dragging:
            active_roi = self.roi_collection.get_active()
            if active_roi:
                new_x = img_pos.x() - self.drag_offset.x()
                new_y = img_pos.y() - self.drag_offset.y()
                active_roi.x = new_x
                active_roi.y = new_y
                self.update_display()
        elif self.resizing:
            active_roi = self.roi_collection.get_active()
            if active_roi:
                self._resize_roi(active_roi, img_pos)
                self.update_display()
                
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.drawing:
            self.drawing = False
            if self.temp_rect and self.temp_rect.width() > 5 and self.temp_rect.height() > 5:
                # Create new ROI
                new_roi = ROI(
                    x=self.temp_rect.x(),
                    y=self.temp_rect.y(),
                    width=self.temp_rect.width(),
                    height=self.temp_rect.height()
                )
                
                if self.multi_roi:
                    idx = self.roi_collection.add(new_roi)
                    self.roi_collection.set_active(idx)
                else:
                    # Single ROI mode - replace existing
                    self.roi_collection.clear()
                    idx = self.roi_collection.add(new_roi)
                    self.roi_collection.set_active(idx)
                
                self.roi_changed.emit(new_roi)
            self.temp_rect = None
            self.update_display()
            
        elif self.dragging or self.resizing:
            self.dragging = False
            self.resizing = False
            self.resize_corner = None
            
            active_roi = self.roi_collection.get_active()
            if active_roi:
                self.roi_changed.emit(active_roi)
                
    def _get_corner_at_pos(self, roi: ROI, pos: QPoint) -> Optional[str]:
        """Check if position is near a resize corner"""
        scaled_rect = self._scale_rect(roi.to_rect())
        handle_size = 12
        
        corners = {
            'tl': scaled_rect.topLeft(),
            'tr': scaled_rect.topRight(),
            'bl': scaled_rect.bottomLeft(),
            'br': scaled_rect.bottomRight()
        }
        
        for corner_name, corner_pos in corners.items():
            if (abs(pos.x() - corner_pos.x()) < handle_size and
                abs(pos.y() - corner_pos.y()) < handle_size):
                return corner_name
        return None
    
    def _is_point_in_roi(self, roi: ROI, pos: QPoint) -> bool:
        """Check if point is inside ROI"""
        scaled_rect = self._scale_rect(roi.to_rect())
        return scaled_rect.contains(pos)
    
    def _resize_roi(self, roi: ROI, pos: QPoint):
        """Resize ROI based on corner being dragged"""
        rect = roi.to_rect()
        
        if self.resize_corner == 'tl':
            rect.setTopLeft(pos)
        elif self.resize_corner == 'tr':
            rect.setTopRight(pos)
        elif self.resize_corner == 'bl':
            rect.setBottomLeft(pos)
        elif self.resize_corner == 'br':
            rect.setBottomRight(pos)
        
        rect = rect.normalized()
        roi.x = rect.x()
        roi.y = rect.y()
        roi.width = rect.width()
        roi.height = rect.height()
    
    def add_roi(self, roi: ROI) -> int:
        """Add ROI programmatically"""
        idx = self.roi_collection.add(roi)
        self.roi_collection.set_active(idx)
        self.update_display()
        return idx
    
    def get_active_roi(self) -> Optional[ROI]:
        """Get currently active ROI"""
        return self.roi_collection.get_active()
    
    def get_all_rois(self) -> List[ROI]:
        """Get all ROIs"""
        return self.roi_collection.rois.copy()
    
    def clear_rois(self):
        """Clear all ROIs"""
        self.roi_collection.clear()
        self.update_display()
    
    def set_mode_filter(self, mode: ROIMode):
        """Show only ROIs of specific mode"""
        # Can be extended to filter display
        pass


class ROIManager(QObject):
    """High-level ROI management with callbacks"""
    
    roi_created = pyqtSignal(ROI)
    roi_updated = pyqtSignal(ROI)
    roi_deleted = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.collection = ROICollection()
        self.callbacks = {
            'create': [],
            'update': [],
            'delete': []
        }
    
    def create_roi(self, x: int, y: int, width: int, height: int, 
                   name: str = "", mode: ROIMode = ROIMode.TEACHING,
                   metadata: Dict = None) -> ROI:
        """Create and register new ROI"""
        roi = ROI(x, y, width, height, name, mode, metadata=metadata or {})
        idx = self.collection.add(roi)
        self.roi_created.emit(roi)
        self._trigger_callbacks('create', roi)
        return roi
    
    def update_roi(self, index: int, roi: ROI):
        """Update existing ROI"""
        existing = self.collection.get(index)
        if existing:
            self.collection.rois[index] = roi
            self.roi_updated.emit(roi)
            self._trigger_callbacks('update', roi)
    
    def delete_roi(self, index: int):
        """Delete ROI"""
        if self.collection.remove(index):
            self.roi_deleted.emit(index)
            self._trigger_callbacks('delete', index)
    
    def on_create(self, callback: Callable):
        """Register callback for ROI creation"""
        self.callbacks['create'].append(callback)
    
    def on_update(self, callback: Callable):
        """Register callback for ROI update"""
        self.callbacks['update'].append(callback)
    
    def on_delete(self, callback: Callable):
        """Register callback for ROI deletion"""
        self.callbacks['delete'].append(callback)
    
    def _trigger_callbacks(self, event_type: str, data):
        """Trigger registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            callback(data)
    
    def export_rois(self) -> List[Dict]:
        """Export all ROIs to dictionaries"""
        return [roi.to_dict() for roi in self.collection.rois]
    
    def import_rois(self, rois_data: List[Dict]):
        """Import ROIs from dictionaries"""
        self.collection.clear()
        for data in rois_data:
            roi = ROI.from_dict(data)
            self.collection.add(roi)
