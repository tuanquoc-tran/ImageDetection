"""
Main Window - Industrial Vision Inspection GUI
Refactored from vision_gui.py with proper module organization.
Centralized UI controller with behavior-driven state management.
"""

import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QFileDialog, QMessageBox,
    QSpinBox, QComboBox, QGridLayout, QInputDialog, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QKeyEvent
import logging

# Import from organized modules
from ..controller.behavior_controller import BehaviorController, SystemState
from ..controller.app_controller import ApplicationController
from ..vision.golden_sample import ROIConfig, GoldenSample
from ..vision.inspector import WireOrderResult
from ..vision.image_source import ImageSourceFactory
from .roi_editor import InteractiveROIEditor, ROI

logger = logging.getLogger(__name__)


class InspectionMainWindow(QMainWindow):
    """
    Main GUI window for industrial vision inspection.
    Refactored with proper separation of concerns.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Vision Inspection System v1.0")
        
        # Start in maximized mode (recommended for industrial use)
        self.showMaximized()
        
        # Track fullscreen state
        self.is_fullscreen = False
        
        # Core controller
        self.app_controller = ApplicationController()
        self.behavior_controller = self.app_controller.behavior_controller
        
        # UI state
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        
        self.current_frame = None
        self.current_result: WireOrderResult = None
        
        # Teaching state
        self.teaching_image = None
        
        # Track current ROI for saving
        self.current_roi_image = None
        self.current_roi_coords = None
        
        # Register state callbacks
        self._register_state_callbacks()
        
        # Initialize UI
        self.init_ui()
        
        # Try to load existing golden sample
        self._try_load_golden_sample()
        
        # Update UI for initial state
        self.update_ui_for_state()
        
        logger.info("InspectionMainWindow initialized in maximized mode")
    
    def _register_state_callbacks(self):
        """Register callbacks for state transitions."""
        self.behavior_controller.register_state_callback(SystemState.IDLE, self._on_idle_state)
        self.behavior_controller.register_state_callback(SystemState.IMAGE_MODE, self._on_image_mode_state)
        self.behavior_controller.register_state_callback(SystemState.CAMERA_MODE, self._on_camera_mode_state)
        self.behavior_controller.register_state_callback(SystemState.TEACH_SAMPLE, self._on_teach_sample_state)
        self.behavior_controller.register_state_callback(SystemState.READY, self._on_ready_state)
        self.behavior_controller.register_state_callback(SystemState.INSPECTION, self._on_inspection_state)
        self.behavior_controller.register_state_callback(SystemState.ERROR, self._on_error_state)
    
    def _on_idle_state(self, state: SystemState, metadata: dict = None):
        """Callback for IDLE state."""
        logger.info("Entered IDLE state")
        self.update_status_display("IDLE", "gray", "System ready. Select mode or teach sample.")
        
    def _on_image_mode_state(self, state: SystemState, metadata: dict = None):
        """Callback for IMAGE_MODE state."""
        logger.info("Entered IMAGE_MODE state")
        self.update_status_display("IMAGE MODE", "#3498db", "Image mode active. Load image or inspect.")
        
    def _on_camera_mode_state(self, state: SystemState, metadata: dict = None):
        """Callback for CAMERA_MODE state."""
        logger.info("Entered CAMERA_MODE state")
        self.update_status_display("CAMERA MODE", "#9b59b6", "Camera mode active. Live inspection ready.")
        
    def _on_teach_sample_state(self, state: SystemState, metadata: dict = None):
        """Callback for TEACH_SAMPLE state."""
        logger.info("Entered TEACH_SAMPLE state")
        self.update_status_display("TEACHING", "#f39c12", "Teaching mode. Define ROI and save sample.")
        
    def _on_ready_state(self, state: SystemState, metadata: dict = None):
        """Callback for READY state."""
        logger.info("Entered READY state")
        self.update_status_display("READY", "#27ae60", "System ready for inspection.")
        
    def _on_inspection_state(self, state: SystemState, metadata: dict = None):
        """Callback for INSPECTION state."""
        logger.info("Entered INSPECTION state")
        self.update_status_display("INSPECTING", "#e67e22", "Performing inspection...")
        
    def _on_error_state(self, state: SystemState, metadata: dict = None):
        """Callback for ERROR state."""
        error_msg = metadata.get('error', 'Unknown error') if metadata else 'Unknown error'
        logger.error(f"Entered ERROR state: {error_msg}")
        self.update_status_display("ERROR", "#e74c3c", f"Error: {error_msg}")
        QMessageBox.critical(self, "System Error", f"Error occurred:\n{error_msg}\n\nSystem will reset.")
        
    def _try_load_golden_sample(self):
        """Try to load existing golden sample on startup."""
        golden_sample = self.app_controller.load_golden_sample()
        if golden_sample:
            self._update_golden_sample_display(golden_sample)
            logger.info(f"Loaded golden sample: {golden_sample.sample_name}")
            QMessageBox.information(
                self, "Golden Sample Loaded",
                f"Loaded existing golden sample:\n{golden_sample.sample_name}\n"
                f"Wire order: {golden_sample.wire_color_order}"
            )
    
    def _update_golden_sample_display(self, golden_sample: GoldenSample):
        """Update golden sample display in UI."""
        self.golden_sample_label.setText(
            f"Golden Sample: {golden_sample.sample_name}\n"
            f"Wires: {golden_sample.num_wires}\n"
            f"Order: {', '.join(golden_sample.wire_color_order)}"
        )
        self.golden_sample_label.setStyleSheet(
            "padding: 10px; background-color: #d4edda; color: #155724;"
        )
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = QVBoxLayout()
        
        # System State Display
        state_group = QGroupBox("System State")
        state_layout = QVBoxLayout()
        
        self.state_label = QLabel("IDLE")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_font = QFont()
        state_font.setPointSize(24)
        state_font.setBold(True)
        self.state_label.setFont(state_font)
        self.state_label.setStyleSheet("background-color: gray; color: white; padding: 15px;")
        state_layout.addWidget(self.state_label)
        
        self.state_message_label = QLabel("System initialized")
        self.state_message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_layout.addWidget(self.state_message_label)
        
        state_group.setLayout(state_layout)
        left_panel.addWidget(state_group)
        
        # Camera view
        camera_group = QGroupBox("Live View / Image Display")
        camera_layout = QVBoxLayout()
        
        self.roi_editor = InteractiveROIEditor(multi_roi=False)
        self.roi_editor.roi_changed.connect(self.on_roi_changed)
        camera_layout.addWidget(self.roi_editor)
        
        # Mode controls
        mode_controls = QHBoxLayout()
        
        self.mode_group = QButtonGroup()
        self.radio_image_mode = QRadioButton("Image Mode")
        self.radio_camera_mode = QRadioButton("Camera Mode")
        self.mode_group.addButton(self.radio_image_mode, 0)
        self.mode_group.addButton(self.radio_camera_mode, 1)
        self.radio_image_mode.toggled.connect(self.on_mode_changed)
        
        mode_controls.addWidget(QLabel("Mode:"))
        mode_controls.addWidget(self.radio_image_mode)
        mode_controls.addWidget(self.radio_camera_mode)
        mode_controls.addSpacing(20)
        
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image_file)
        mode_controls.addWidget(self.btn_load_image)
        
        self.camera_select = QComboBox()
        self.camera_select.addItems(["Camera 0", "Camera 1", "Camera 2"])
        mode_controls.addWidget(self.camera_select)
        
        self.btn_start_camera = QPushButton("Start Camera")
        self.btn_start_camera.clicked.connect(self.start_camera)
        mode_controls.addWidget(self.btn_start_camera)
        
        self.btn_stop_camera = QPushButton("Stop Camera")
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_stop_camera.setEnabled(False)
        mode_controls.addWidget(self.btn_stop_camera)
        
        mode_controls.addStretch()
        camera_layout.addLayout(mode_controls)
        camera_group.setLayout(camera_layout)
        left_panel.addWidget(camera_group)
        
        # Right panel
        right_panel = QVBoxLayout()
        
        # Result display
        result_group = QGroupBox("Inspection Result")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("NO RESULT")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(36)
        font.setBold(True)
        self.result_label.setFont(font)
        self.result_label.setStyleSheet("background-color: gray; color: white; padding: 20px;")
        result_layout.addWidget(self.result_label)
        
        self.defect_label = QLabel("Waiting for inspection...")
        self.defect_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.defect_label.setWordWrap(True)
        result_layout.addWidget(self.defect_label)
        
        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group)
        
        # Golden Sample Management
        teaching_group = QGroupBox("Golden Sample Management")
        teaching_layout = QVBoxLayout()
        
        self.golden_sample_label = QLabel("No golden sample loaded")
        self.golden_sample_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        self.golden_sample_label.setWordWrap(True)
        teaching_layout.addWidget(self.golden_sample_label)
        
        teach_buttons = QHBoxLayout()
        self.btn_load_for_teaching = QPushButton("Load Sample Image")
        self.btn_load_for_teaching.clicked.connect(self.load_sample_for_teaching)
        teach_buttons.addWidget(self.btn_load_for_teaching)
        
        self.btn_teach_from_camera = QPushButton("Teach from Camera")
        self.btn_teach_from_camera.clicked.connect(self.teach_from_camera)
        teach_buttons.addWidget(self.btn_teach_from_camera)
        teaching_layout.addLayout(teach_buttons)
        
        self.btn_save_golden_sample = QPushButton("Save Golden Sample")
        self.btn_save_golden_sample.clicked.connect(self.save_golden_sample)
        self.btn_save_golden_sample.setEnabled(False)
        teaching_layout.addWidget(self.btn_save_golden_sample)
        
        self.btn_load_golden_sample = QPushButton("Load Golden Sample")
        self.btn_load_golden_sample.clicked.connect(self.load_golden_sample_file)
        teaching_layout.addWidget(self.btn_load_golden_sample)
        
        teaching_group.setLayout(teaching_layout)
        right_panel.addWidget(teaching_group)
        
        # ROI Configuration
        roi_group = QGroupBox("ROI Configuration (Teaching)")
        roi_layout = QGridLayout()
        
        roi_layout.addWidget(QLabel("Connector ROI:"), 0, 0, 1, 4)
        
        roi_layout.addWidget(QLabel("X:"), 1, 0)
        self.spin_connector_x = QSpinBox()
        self.spin_connector_x.setRange(0, 10000)
        self.spin_connector_x.setValue(100)
        roi_layout.addWidget(self.spin_connector_x, 1, 1)
        
        roi_layout.addWidget(QLabel("Y:"), 1, 2)
        self.spin_connector_y = QSpinBox()
        self.spin_connector_y.setRange(0, 10000)
        self.spin_connector_y.setValue(100)
        roi_layout.addWidget(self.spin_connector_y, 1, 3)
        
        roi_layout.addWidget(QLabel("Width:"), 2, 0)
        self.spin_connector_width = QSpinBox()
        self.spin_connector_width.setRange(1, 10000)
        self.spin_connector_width.setValue(800)
        roi_layout.addWidget(self.spin_connector_width, 2, 1)
        
        roi_layout.addWidget(QLabel("Height:"), 2, 2)
        self.spin_connector_height = QSpinBox()
        self.spin_connector_height.setRange(1, 10000)
        self.spin_connector_height.setValue(400)
        roi_layout.addWidget(self.spin_connector_height, 2, 3)
        
        roi_layout.addWidget(QLabel("Wire ROI:"), 3, 0, 1, 4)
        
        roi_layout.addWidget(QLabel("X:"), 4, 0)
        self.spin_wire_x = QSpinBox()
        self.spin_wire_x.setRange(0, 10000)
        self.spin_wire_x.setValue(150)
        roi_layout.addWidget(self.spin_wire_x, 4, 1)
        
        roi_layout.addWidget(QLabel("Y:"), 4, 2)
        self.spin_wire_y = QSpinBox()
        self.spin_wire_y.setRange(0, 10000)
        self.spin_wire_y.setValue(250)
        roi_layout.addWidget(self.spin_wire_y, 4, 3)
        
        roi_layout.addWidget(QLabel("Width:"), 5, 0)
        self.spin_wire_width = QSpinBox()
        self.spin_wire_width.setRange(1, 10000)
        self.spin_wire_width.setValue(700)
        roi_layout.addWidget(self.spin_wire_width, 5, 1)
        
        roi_layout.addWidget(QLabel("Height:"), 5, 2)
        self.spin_wire_height = QSpinBox()
        self.spin_wire_height.setRange(1, 10000)
        self.spin_wire_height.setValue(150)
        roi_layout.addWidget(self.spin_wire_height, 5, 3)
        
        roi_layout.addWidget(QLabel("Num Wires:"), 6, 0)
        self.spin_num_wires = QSpinBox()
        self.spin_num_wires.setRange(1, 20)
        self.spin_num_wires.setValue(4)
        roi_layout.addWidget(self.spin_num_wires, 6, 1, 1, 3)
        
        roi_group.setLayout(roi_layout)
        right_panel.addWidget(roi_group)
        
        # Inspection control
        inspect_group = QGroupBox("Inspection Control")
        inspect_layout = QVBoxLayout()
        
        self.btn_single_inspect = QPushButton("Single Inspection")
        self.btn_single_inspect.clicked.connect(self.run_single_inspection)
        inspect_layout.addWidget(self.btn_single_inspect)
        
        self.btn_continuous = QPushButton("Continuous Inspection")
        self.btn_continuous.setCheckable(True)
        self.btn_continuous.clicked.connect(self.toggle_continuous_inspection)
        inspect_layout.addWidget(self.btn_continuous)
        
        # Add Save ROI button
        self.btn_save_roi = QPushButton("Save ROI Image")
        self.btn_save_roi.clicked.connect(self.save_roi_image)
        self.btn_save_roi.setEnabled(False)
        inspect_layout.addWidget(self.btn_save_roi)
        
        # Add Process ROI button
        self.btn_process_roi = QPushButton("Process ROI Only")
        self.btn_process_roi.clicked.connect(self.process_roi_only)
        self.btn_process_roi.setEnabled(False)
        inspect_layout.addWidget(self.btn_process_roi)
        
        inspect_group.setLayout(inspect_layout)
        right_panel.addWidget(inspect_group)
        
        # System controls
        system_group = QGroupBox("System Control")
        system_layout = QVBoxLayout()
        
        self.btn_reset = QPushButton("Reset System")
        self.btn_reset.clicked.connect(self.reset_system)
        system_layout.addWidget(self.btn_reset)
        
        self.btn_show_state = QPushButton("Show State Info")
        self.btn_show_state.clicked.connect(self.show_state_info)
        system_layout.addWidget(self.btn_show_state)
        
        system_group.setLayout(system_layout)
        right_panel.addWidget(system_group)
        
        right_panel.addStretch()
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
    
    # ===== Mode Selection =====
    
    def on_mode_changed(self, checked):
        """Handle mode selection change."""
        if not checked:
            return
            
        if self.radio_image_mode.isChecked():
            logger.info("User selected IMAGE_MODE")
            self.btn_load_image.setEnabled(True)
            self.btn_start_camera.setEnabled(False)
            self.stop_camera()
        else:
            logger.info("User selected CAMERA_MODE")
            self.btn_load_image.setEnabled(False)
            self.btn_start_camera.setEnabled(True)
    
    def load_image_file(self):
        """Load image file for inspection with automatic ROI visualization."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Image for Inspection", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if not filename:
            return
        
        source = ImageSourceFactory.create_image_source(filename)
        if not source.is_available():
            QMessageBox.warning(self, "Error", f"Failed to load image:\n{filename}")
            return
        
        self.app_controller.set_image_source(source)
        
        # Transition to IMAGE_MODE
        current_state = self.behavior_controller.current_state
        if current_state in [SystemState.IDLE, SystemState.READY, SystemState.IMAGE_MODE]:
            if current_state != SystemState.IMAGE_MODE:
                self.behavior_controller.start_image_mode()
        
        # Display image with automatic ROI visualization
        frame = self.app_controller.get_current_frame()
        if frame is not None:
            self.current_frame = frame
            
            # Automatic ROI visualization if golden sample exists
            if self.app_controller.wire_inspector and self.app_controller.template_localizer:
                vis_frame = self._apply_auto_roi_visualization(frame)
                self.roi_editor.set_image(vis_frame)
            else:
                self.roi_editor.set_image(frame)
        
        self.radio_image_mode.setChecked(True)
        logger.info(f"Loaded image: {filename}")
    
    def _apply_auto_roi_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Apply automatic ROI detection and visualization overlay."""
        golden_sample = self.app_controller.wire_inspector.golden_sample
        if not golden_sample:
            return frame
        
        vis_frame = frame.copy()
        roi_cfg = golden_sample.roi_config
        
        # Try template localization first (position-independent)
        if self.app_controller.template_localizer.template is not None:
            roi_coords, match_result = self.app_controller.template_localizer.locate_roi(
                frame, roi_cfg.wire_width, roi_cfg.wire_height
            )
            
            if match_result.found and roi_coords:
                roi_x, roi_y, roi_w, roi_h = roi_coords
                
                # Save current ROI coordinates and image
                self.current_roi_coords = roi_coords
                self.current_roi_image = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
                
                # Enable ROI buttons
                self.btn_save_roi.setEnabled(True)
                self.btn_process_roi.setEnabled(True)
                
                # Draw template match area (magenta)
                template_h, template_w = self.app_controller.template_localizer.template.shape[:2]
                cv2.rectangle(vis_frame,
                             (match_result.match_x, match_result.match_y),
                             (match_result.match_x + template_w, match_result.match_y + template_h),
                             (255, 0, 255), 2)
                
                cv2.putText(vis_frame, f"Template Match: {match_result.confidence:.2f}",
                           (match_result.match_x, match_result.match_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Draw wire inspection ROI (green)
                cv2.rectangle(vis_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 3)
                
                # Draw wire segments
                segment_width = roi_w // golden_sample.num_wires
                for i in range(golden_sample.num_wires):
                    seg_x = roi_x + (i * segment_width)
                    seg_w = segment_width if i < golden_sample.num_wires - 1 else (roi_x + roi_w - seg_x)
                    
                    cv2.rectangle(vis_frame, (seg_x, roi_y), (seg_x + seg_w, roi_y + roi_h), (255, 255, 0), 2)
                    cv2.putText(vis_frame, f"W{i+1}", (seg_x + 5, roi_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(vis_frame, f"Auto ROI - Offset: ({match_result.offset_x},{match_result.offset_y})",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Expected: {', '.join(golden_sample.wire_color_order)}",
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                logger.info(f"Auto ROI located: confidence={match_result.confidence:.3f}")
            else:
                # Template matching failed
                cv2.putText(vis_frame, f"Template Match Failed (conf: {match_result.confidence:.2f})",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                vis_frame = self._draw_static_roi_overlay(vis_frame, golden_sample)
                
                # Disable ROI buttons
                self.btn_save_roi.setEnabled(False)
                self.btn_process_roi.setEnabled(False)
                
                logger.warning(f"Template match failed: confidence={match_result.confidence:.3f}")
        else:
            vis_frame = self._draw_static_roi_overlay(vis_frame, golden_sample)
            
            # Save static ROI
            self.current_roi_coords = (roi_cfg.wire_x, roi_cfg.wire_y, roi_cfg.wire_width, roi_cfg.wire_height)
            self.current_roi_image = frame[roi_cfg.wire_y:roi_cfg.wire_y+roi_cfg.wire_height,
                                          roi_cfg.wire_x:roi_cfg.wire_x+roi_cfg.wire_width].copy()
            
            # Enable ROI buttons
            self.btn_save_roi.setEnabled(True)
            self.btn_process_roi.setEnabled(True)
            
            logger.info("Using static ROI (no template)")
        
        return vis_frame
    
    def _draw_static_roi_overlay(self, vis_frame: np.ndarray, golden_sample) -> np.ndarray:
        """Draw static ROI overlay (fallback)."""
        roi_cfg = golden_sample.roi_config
        
        cv2.rectangle(vis_frame, (roi_cfg.connector_x, roi_cfg.connector_y),
                     (roi_cfg.connector_x + roi_cfg.connector_width, 
                      roi_cfg.connector_y + roi_cfg.connector_height),
                     (0, 255, 255), 2)
        
        cv2.putText(vis_frame, "Connector Template", (roi_cfg.connector_x, roi_cfg.connector_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.rectangle(vis_frame, (roi_cfg.wire_x, roi_cfg.wire_y),
                     (roi_cfg.wire_x + roi_cfg.wire_width, roi_cfg.wire_y + roi_cfg.wire_height),
                     (0, 255, 0), 3)
        
        segment_width = roi_cfg.wire_width // golden_sample.num_wires
        for i in range(golden_sample.num_wires):
            seg_x = roi_cfg.wire_x + (i * segment_width)
            seg_w = segment_width if i < golden_sample.num_wires - 1 else (roi_cfg.wire_x + roi_cfg.wire_width - seg_x)
            
            cv2.rectangle(vis_frame, (seg_x, roi_cfg.wire_y), (seg_x + seg_w, roi_cfg.wire_y + roi_cfg.wire_height), (255, 255, 0), 2)
            cv2.putText(vis_frame, f"W{i+1}", (seg_x + 5, roi_cfg.wire_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(vis_frame, "Static ROI (No Template Match)", (roi_cfg.wire_x, roi_cfg.wire_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Golden: {golden_sample.sample_name}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Expected: {', '.join(golden_sample.wire_color_order)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    # ===== Camera Management =====
    
    def start_camera(self):
        """Start camera capture."""
        camera_idx = self.camera_select.currentIndex()
        source = ImageSourceFactory.create_camera_source(camera_idx)
        if not source.is_available():
            QMessageBox.warning(self, "Camera Error", f"Failed to open camera {camera_idx}!")
            return
        
        self.app_controller.set_image_source(source)
        
        current_state = self.behavior_controller.current_state
        if current_state in [SystemState.IDLE, SystemState.READY, SystemState.CAMERA_MODE]:
            if current_state != SystemState.CAMERA_MODE:
                self.behavior_controller.start_camera_mode()
        
        self.camera_timer.start(33)
        self.btn_start_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)
        self.radio_camera_mode.setChecked(True)
        logger.info(f"Camera started: device {camera_idx}")
    
    def stop_camera(self):
        """Stop camera capture."""
        self.camera_timer.stop()
        self.btn_start_camera.setEnabled(True)
        self.btn_stop_camera.setEnabled(False)
        logger.info("Camera stopped")
    
    def update_camera_frame(self):
        """Update camera frame with auto ROI visualization."""
        if self.behavior_controller.current_state == SystemState.TEACH_SAMPLE:
            return
        
        frame = self.app_controller.get_current_frame()
        if frame is not None:
            self.current_frame = frame
            
            if self.app_controller.wire_inspector and self.app_controller.template_localizer:
                vis_frame = self._apply_auto_roi_visualization(frame)
                self.roi_editor.set_image(vis_frame)
            else:
                self.roi_editor.set_image(frame)
            
            if self.btn_continuous.isChecked():
                self.run_single_inspection()
    
    # ===== Teaching Workflow =====
    
    def load_sample_for_teaching(self):
        """Load sample image for teaching."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Sample Image for Teaching", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if not filename:
            return
        
        image = cv2.imread(filename)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image!")
            return
        
        if not self.behavior_controller.start_teaching():
            QMessageBox.warning(self, "State Error", 
                              f"Cannot start teaching from {self.behavior_controller.current_state.name}")
            return
        
        self.teaching_image = image
        self.current_frame = image
        self.roi_editor.set_image(image)
        self.btn_save_golden_sample.setEnabled(True)
        self.stop_camera()
        
        logger.info(f"Loaded teaching image: {filename}")
        QMessageBox.information(self, "Teaching Mode", 
                              "Teaching mode active.\nDefine ROI and save golden sample.")
    
    def teach_from_camera(self):
        """Capture camera frame for teaching."""
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "No camera frame available!")
            return
        
        if not self.behavior_controller.start_teaching():
            QMessageBox.warning(self, "State Error", 
                              f"Cannot start teaching from {self.behavior_controller.current_state.name}")
            return
        
        self.teaching_image = self.current_frame.copy()
        self.btn_save_golden_sample.setEnabled(True)
        self.camera_timer.stop()
        
        logger.info("Captured frame for teaching")
        QMessageBox.information(self, "Teaching Mode",
                              "Current frame captured.\nDefine ROI and save golden sample.")
    
    def save_golden_sample(self):
        """Save golden sample with template setup."""
        if self.teaching_image is None:
            QMessageBox.warning(self, "Error", "No teaching image loaded!")
            return
        
        sample_name, ok = QInputDialog.getText(self, "Sample Name", "Enter golden sample name:")
        if not ok or not sample_name:
            return
        
        from ..vision.golden_sample import TeachingMode
        
        roi_config = ROIConfig(
            connector_x=self.spin_connector_x.value(),
            connector_y=self.spin_connector_y.value(),
            connector_width=self.spin_connector_width.value(),
            connector_height=self.spin_connector_height.value(),
            wire_x=self.spin_wire_x.value(),
            wire_y=self.spin_wire_y.value(),
            wire_width=self.spin_wire_width.value(),
            wire_height=self.spin_wire_height.value()
        )
        
        num_wires = self.spin_num_wires.value()
        
        try:
            teaching_mode = TeachingMode()
            golden_sample = teaching_mode.create_golden_sample(
                image=self.teaching_image,
                sample_name=sample_name,
                roi_config=roi_config,
                num_wires=num_wires,
                notes=f"Created via GUI at {Path.cwd()}"
            )
            
            if not self.app_controller.save_golden_sample(golden_sample):
                raise ValueError("Failed to save golden sample")
            
            # Set up template localization
            if self.app_controller.template_localizer:
                self.app_controller.template_localizer.set_golden_template(
                    image=self.teaching_image,
                    template_x=roi_config.connector_x,
                    template_y=roi_config.connector_y,
                    template_width=roi_config.connector_width,
                    template_height=roi_config.connector_height,
                    roi_x=roi_config.wire_x,
                    roi_y=roi_config.wire_y
                )
            
            self.behavior_controller.complete_teaching(success=True)
            self._update_golden_sample_display(golden_sample)
            
            self.btn_save_golden_sample.setEnabled(False)
            self.teaching_image = None
            
            if self.radio_camera_mode.isChecked() and self.app_controller.current_source:
                self.camera_timer.start(33)
            
            logger.info(f"Golden sample saved: {sample_name}")
            QMessageBox.information(
                self, "Success",
                f"Golden sample saved!\n\nName: {sample_name}\n"
                f"Wires: {num_wires}\n"
                f"Order: {', '.join(golden_sample.wire_color_order)}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save golden sample: {e}")
            self.behavior_controller.complete_teaching(success=False)
            QMessageBox.critical(self, "Error", f"Failed to save golden sample:\n{str(e)}")
    
    def load_golden_sample_file(self):
        """Load golden sample from file."""
        golden_sample = self.app_controller.load_golden_sample()
        if golden_sample:
            self._update_golden_sample_display(golden_sample)
            QMessageBox.information(
                self, "Success",
                f"Golden sample loaded:\n{golden_sample.sample_name}\n"
                f"Wire order: {', '.join(golden_sample.wire_color_order)}"
            )
        else:
            QMessageBox.warning(self, "Error", "No golden sample file found!")
    
    # ===== Inspection =====
    
    def run_single_inspection(self):
        """Run single inspection."""
        if not self.app_controller.is_ready_for_inspection():
            QMessageBox.warning(self, "Not Ready", 
                              "System not ready.\nLoad golden sample and image/camera first.")
            return
        
        if not self.behavior_controller.start_inspection():
            QMessageBox.warning(self, "State Error", "Failed to start inspection")
            return
        
        try:
            if self.app_controller.position_independent_inspector and self.current_frame is not None:
                result, match_result = self.app_controller.position_independent_inspector.inspect_with_localization(
                    self.current_frame
                )
            else:
                if self.app_controller.wire_inspector and self.current_frame is not None:
                    result = self.app_controller.wire_inspector.inspect(self.current_frame)
                else:
                    raise ValueError("No inspector available")
            
            self.current_result = result
            self._display_result(result)
            
            logger.info(f"Inspection complete: {result.get_defect_message()} ({result.processing_time_ms:.2f}ms)")
            
        except Exception as e:
            logger.error(f"Inspection failed: {e}", exc_info=True)
            self.behavior_controller.complete_inspection(success=False)
            QMessageBox.critical(self, "Inspection Error", f"Inspection failed:\n{str(e)}")
    
    def _display_result(self, result: WireOrderResult):
        """Display inspection result."""
        if result.is_ok:
            self.result_label.setText("OK")
            self.result_label.setStyleSheet("background-color: green; color: white; padding: 20px;")
            self.defect_label.setText(
                f"✓ Wire order correct\n\n"
                f"Detected: {', '.join(result.detected_order)}\n"
                f"Expected: {', '.join(result.expected_order)}\n\n"
                f"Time: {result.processing_time_ms:.2f} ms"
            )
        else:
            self.result_label.setText("NG")
            self.result_label.setStyleSheet("background-color: red; color: white; padding: 20px;")
            self.defect_label.setText(
                f"✗ {result.get_defect_message()}\n\n"
                f"Detected: {', '.join(result.detected_order) if result.detected_order else 'N/A'}\n"
                f"Expected: {', '.join(result.expected_order) if result.expected_order else 'N/A'}\n\n"
                f"Time: {result.processing_time_ms:.2f} ms"
            )
    
    def toggle_continuous_inspection(self, checked):
        """Toggle continuous inspection."""
        if checked:
            self.btn_continuous.setText("Stop Continuous")
            logger.info("Continuous inspection enabled")
        else:
            self.btn_continuous.setText("Continuous Inspection")
            logger.info("Continuous inspection disabled")
    
    def save_roi_image(self):
        """Save current ROI image to file."""
        if self.current_roi_image is None:
            QMessageBox.warning(self, "No ROI", "No ROI image available to save!")
            return
        
        # Suggest filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"roi_{timestamp}.png"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Image",
            default_filename,
            "Image Files (*.png *.jpg *.bmp)"
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.current_roi_image)
                logger.info(f"ROI image saved: {filename}")
                QMessageBox.information(
                    self, "Success",
                    f"ROI image saved successfully!\n\n"
                    f"File: {filename}\n"
                    f"Size: {self.current_roi_image.shape[1]}x{self.current_roi_image.shape[0]}"
                )
            except Exception as e:
                logger.error(f"Failed to save ROI image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save ROI image:\n{str(e)}")
    
    def process_roi_only(self):
        """Process inspection on ROI only and display results."""
        if self.current_roi_image is None:
            QMessageBox.warning(self, "No ROI", "No ROI image available for processing!")
            return
        
        if not self.app_controller.wire_inspector:
            QMessageBox.warning(self, "Not Ready", "No golden sample loaded!")
            return
        
        try:
            # Create a temporary full-size image with ROI
            roi_x, roi_y, roi_w, roi_h = self.current_roi_coords
            temp_image = np.zeros_like(self.current_frame) if self.current_frame is not None else np.zeros((roi_h*2, roi_w*2, 3), dtype=np.uint8)
            
            # Place ROI at its coordinates
            if roi_y + roi_h <= temp_image.shape[0] and roi_x + roi_w <= temp_image.shape[1]:
                temp_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = self.current_roi_image
            else:
                # If coordinates don't fit, just use ROI directly at origin
                temp_image = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
                temp_image[:] = self.current_roi_image
                roi_x, roi_y = 0, 0
            
            # Run inspection on the ROI
            golden_sample = self.app_controller.wire_inspector.golden_sample
            roi_cfg = golden_sample.roi_config
            
            # Temporarily override ROI config
            original_roi = (roi_cfg.wire_x, roi_cfg.wire_y)
            roi_cfg.wire_x = roi_x
            roi_cfg.wire_y = roi_y
            
            try:
                result = self.app_controller.wire_inspector.inspect(temp_image)
            finally:
                roi_cfg.wire_x, roi_cfg.wire_y = original_roi
            
            # Display ROI processing result
            self._display_roi_processing_result(result)
            
            logger.info(f"ROI processing complete: {result.get_defect_message()}")
            
        except Exception as e:
            logger.error(f"ROI processing failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"ROI processing failed:\n{str(e)}")
    
    def _display_roi_processing_result(self, result: WireOrderResult):
        """Display ROI processing result with visualization."""
        if self.current_roi_image is None:
            return
        
        # Create visualization on ROI
        vis_roi = self.current_roi_image.copy()
        golden_sample = self.app_controller.wire_inspector.golden_sample
        
        # Draw wire segments on ROI
        roi_w = vis_roi.shape[1]
        roi_h = vis_roi.shape[0]
        segment_width = roi_w // golden_sample.num_wires
        
        for i in range(len(result.detected_order)):
            seg_x = i * segment_width
            seg_w = segment_width if i < golden_sample.num_wires - 1 else (roi_w - seg_x)
            
            # Determine if wire is correct
            is_ok = (i < len(result.detected_order) and 
                    i < len(result.expected_order) and
                    result.detected_order[i] == result.expected_order[i])
            
            # Color: Green for OK, Red for NG
            color = (0, 255, 0) if is_ok else (0, 0, 255)
            thickness = 3 if not is_ok else 2
            
            cv2.rectangle(vis_roi, (seg_x, 0), (seg_x + seg_w, roi_h), color, thickness)
            
            # Label with detected color
            label = f"W{i+1}: {result.detected_order[i]}"
            cv2.putText(vis_roi, label, (seg_x + 5, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add result overlay
        result_text = "OK" if result.is_ok else "NG"
        result_color = (0, 255, 0) if result.is_ok else (0, 0, 255)
        
        cv2.putText(vis_roi, result_text, (10, roi_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3)
        
        # Display ROI result
        self.roi_editor.set_image(vis_roi)
        
        # Update result labels
        self._display_result(result)
        
        # Show info message
        QMessageBox.information(
            self, "ROI Processing Complete",
            f"ROI Processing Result: {result_text}\n\n"
            f"Detected: {', '.join(result.detected_order)}\n"
            f"Expected: {', '.join(result.expected_order)}\n\n"
            f"Time: {result.processing_time_ms:.2f} ms\n\n"
            f"ROI Size: {vis_roi.shape[1]}x{vis_roi.shape[0]}"
        )
    
    # ===== System Control =====
    
    def reset_system(self):
        """Reset system."""
        reply = QMessageBox.question(
            self, "Reset System",
            "Reset system?\nThis will clear current mode and stop camera.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.stop_camera()
            self.app_controller.release_image_source()
            
            if self.app_controller.wire_inspector and self.behavior_controller.is_golden_sample_loaded:
                self.behavior_controller.reset_to_ready()
                logger.info("System reset to READY")
            else:
                self.behavior_controller.reset_to_idle()
                logger.info("System reset to IDLE")
            
            self.update_ui_for_state()
            self.result_label.setText("NO RESULT")
            self.result_label.setStyleSheet("background-color: gray; color: white; padding: 20px;")
            self.defect_label.setText("Waiting for inspection...")
    
    def show_state_info(self):
        """Show system state information."""
        summary = self.behavior_controller.get_state_summary()
        history = self.behavior_controller.get_transition_history(5)
        
        history_text = "\n".join([
            f"{t.timestamp}: {t.from_state.name} → {t.to_state.name} ({t.trigger})"
            for t in history
        ])
        
        info_text = f"{summary}\n\nRecent Transitions:\n{history_text}"
        QMessageBox.information(self, "System State Information", info_text)
    
    # ===== UI Updates =====
    
    def update_status_display(self, status_text: str, color: str, message: str):
        """Update status display."""
        self.state_label.setText(status_text)
        self.state_label.setStyleSheet(f"background-color: {color}; color: white; padding: 15px;")
        self.state_message_label.setText(message)
    
    def update_ui_for_state(self):
        """Update UI based on current state."""
        state = self.behavior_controller.current_state
        
        if state == SystemState.IDLE:
            self.btn_single_inspect.setEnabled(False)
            self.btn_continuous.setEnabled(False)
            self.btn_save_golden_sample.setEnabled(False)
        elif state in [SystemState.IMAGE_MODE, SystemState.CAMERA_MODE]:
            self.btn_single_inspect.setEnabled(True)
            self.btn_continuous.setEnabled(True)
        elif state == SystemState.READY:
            self.btn_single_inspect.setEnabled(True)
            self.btn_continuous.setEnabled(True)
        elif state == SystemState.TEACH_SAMPLE:
            self.btn_single_inspect.setEnabled(False)
            self.btn_continuous.setEnabled(False)
            self.btn_save_golden_sample.setEnabled(True)
    
    def on_roi_changed(self, roi: ROI):
        """Handle ROI change from editor."""
        self.spin_connector_x.setValue(roi.x)
        self.spin_connector_y.setValue(roi.y)
        self.spin_connector_width.setValue(roi.width)
        self.spin_connector_height.setValue(roi.height)
    
    # ===== Keyboard Shortcuts =====
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape:
            if self.is_fullscreen:
                self.exit_fullscreen()
        elif event.key() == Qt.Key.Key_F5:
            logger.info("F5 pressed - Quick refresh")
            if self.current_frame is not None:
                if self.app_controller.wire_inspector and self.app_controller.template_localizer:
                    vis_frame = self._apply_auto_roi_visualization(self.current_frame)
                    self.roi_editor.set_image(vis_frame)
                else:
                    self.roi_editor.set_image(self.current_frame)
        elif event.key() == Qt.Key.Key_Space:
            if self.btn_single_inspect.isEnabled():
                self.run_single_inspection()
        elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+S: Save ROI
            if self.btn_save_roi.isEnabled():
                self.save_roi_image()
        elif event.key() == Qt.Key.Key_P and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+P: Process ROI only
            if self.btn_process_roi.isEnabled():
                self.process_roi_only()
        else:
            super().keyPressEvent(event)
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal mode."""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()
    
    def enter_fullscreen(self):
        """Enter fullscreen mode."""
        self.showFullScreen()
        self.is_fullscreen = True
        logger.info("Entered fullscreen mode (Press F11 or ESC to exit)")
    
    def exit_fullscreen(self):
        """Exit fullscreen mode."""
        self.showMaximized()
        self.is_fullscreen = False
        logger.info("Exited fullscreen mode")
    
    # ===== Cleanup =====
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_camera()
        self.app_controller.shutdown()
        event.accept()