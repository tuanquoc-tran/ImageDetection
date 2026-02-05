"""
PyQt6 Industrial Vision Inspection Tool
Features: Live camera, ROI editing, parameter adjustment, config save/load, OK/NG display, Sample Teaching
Refactored with reusable ROI management
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFileDialog, QMessageBox,
    QSpinBox, QComboBox, QGridLayout, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from config import InspectionConfig
from main import ConnectorInspectionSystem
from roi_manager import InteractiveROIEditor, ROI, ROIMode


class InspectionGUI(QMainWindow):
    """Main GUI for industrial vision inspection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Vision Inspection Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Inspection system
        self.config = InspectionConfig()
        self.inspection_system = None
        
        # Camera
        self.camera = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        
        # Current frame
        self.current_frame = None
        self.inspection_result = None
        
        # Product configurations
        self.current_config_file = None
        
        # Teaching mode
        self.teaching_mode = False
        self.teaching_image = None
        self.templates = []  # List of saved templates
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Camera view and ROI editor
        left_panel = QVBoxLayout()
        
        # Camera view with ROI editor
        camera_group = QGroupBox("Live Camera View")
        camera_layout = QVBoxLayout()
        
        # Use refactored ROI editor
        self.roi_editor = InteractiveROIEditor(multi_roi=False)
        self.roi_editor.roi_changed.connect(self.on_roi_changed)
        camera_layout.addWidget(self.roi_editor)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.btn_start_camera = QPushButton("Start Camera")
        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop_camera = QPushButton("Stop Camera")
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_stop_camera.setEnabled(False)
        
        self.camera_select = QComboBox()
        self.camera_select.addItems(["Camera 0", "Camera 1", "Camera 2"])
        
        camera_controls.addWidget(self.camera_select)
        camera_controls.addWidget(self.btn_start_camera)
        camera_controls.addWidget(self.btn_stop_camera)
        camera_controls.addStretch()
        
        camera_layout.addLayout(camera_controls)
        camera_group.setLayout(camera_layout)
        left_panel.addWidget(camera_group)
        
        # Right panel: Controls
        right_panel = QVBoxLayout()
        
        # Inspection result display
        result_group = QGroupBox("Inspection Result")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("READY")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(36)
        font.setBold(True)
        self.result_label.setFont(font)
        self.result_label.setStyleSheet("background-color: gray; color: white; padding: 20px;")
        result_layout.addWidget(self.result_label)
        
        self.defect_label = QLabel("No defects")
        self.defect_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.defect_label)
        
        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group)
        
        # Threshold parameters
        params_group = QGroupBox("Threshold Parameters")
        params_layout = QGridLayout()
        
        # Wire presence threshold
        params_layout.addWidget(QLabel("Wire Presence Threshold:"), 0, 0)
        self.slider_wire_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_wire_threshold.setRange(0, 255)
        self.slider_wire_threshold.setValue(self.config.WIRE_PRESENCE_THRESHOLD)
        self.slider_wire_threshold.valueChanged.connect(self.on_threshold_changed)
        params_layout.addWidget(self.slider_wire_threshold, 0, 1)
        self.label_wire_threshold = QLabel(str(self.config.WIRE_PRESENCE_THRESHOLD))
        params_layout.addWidget(self.label_wire_threshold, 0, 2)
        
        # Min area
        params_layout.addWidget(QLabel("Min Wire Area:"), 1, 0)
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 10000)
        self.spin_min_area.setValue(self.config.WIRE_PRESENCE_MIN_AREA)
        self.spin_min_area.valueChanged.connect(self.on_min_area_changed)
        params_layout.addWidget(self.spin_min_area, 1, 1, 1, 2)
        
        # Color tolerance
        params_layout.addWidget(QLabel("Color Match Tolerance:"), 2, 0)
        self.slider_color_tolerance = QSlider(Qt.Orientation.Horizontal)
        self.slider_color_tolerance.setRange(0, 180)
        self.slider_color_tolerance.setValue(self.config.COLOR_MATCH_TOLERANCE)
        self.slider_color_tolerance.valueChanged.connect(self.on_color_tolerance_changed)
        params_layout.addWidget(self.slider_color_tolerance, 2, 1)
        self.label_color_tolerance = QLabel(str(self.config.COLOR_MATCH_TOLERANCE))
        params_layout.addWidget(self.label_color_tolerance, 2, 2)
        
        # Color confidence
        params_layout.addWidget(QLabel("Color Match Confidence:"), 3, 0)
        self.slider_color_confidence = QSlider(Qt.Orientation.Horizontal)
        self.slider_color_confidence.setRange(0, 100)
        self.slider_color_confidence.setValue(int(self.config.COLOR_MATCH_MIN_CONFIDENCE * 100))
        self.slider_color_confidence.valueChanged.connect(self.on_color_confidence_changed)
        params_layout.addWidget(self.slider_color_confidence, 3, 1)
        self.label_color_confidence = QLabel(f"{self.config.COLOR_MATCH_MIN_CONFIDENCE:.2f}")
        params_layout.addWidget(self.label_color_confidence, 3, 2)
        
        params_group.setLayout(params_layout)
        right_panel.addWidget(params_group)
        
        # ROI Configuration
        roi_group = QGroupBox("ROI Configuration")
        roi_layout = QGridLayout()
        
        roi_layout.addWidget(QLabel("X:"), 0, 0)
        self.spin_roi_x = QSpinBox()
        self.spin_roi_x.setRange(0, 10000)
        self.spin_roi_x.setValue(self.config.CONNECTOR_ROI['x'])
        roi_layout.addWidget(self.spin_roi_x, 0, 1)
        
        roi_layout.addWidget(QLabel("Y:"), 0, 2)
        self.spin_roi_y = QSpinBox()
        self.spin_roi_y.setRange(0, 10000)
        self.spin_roi_y.setValue(self.config.CONNECTOR_ROI['y'])
        roi_layout.addWidget(self.spin_roi_y, 0, 3)
        
        roi_layout.addWidget(QLabel("Width:"), 1, 0)
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, 10000)
        self.spin_roi_width.setValue(self.config.CONNECTOR_ROI['width'])
        roi_layout.addWidget(self.spin_roi_width, 1, 1)
        
        roi_layout.addWidget(QLabel("Height:"), 1, 2)
        self.spin_roi_height = QSpinBox()
        self.spin_roi_height.setRange(1, 10000)
        self.spin_roi_height.setValue(self.config.CONNECTOR_ROI['height'])
        roi_layout.addWidget(self.spin_roi_height, 1, 3)
        
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_apply_roi.clicked.connect(self.apply_roi_from_spinboxes)
        roi_layout.addWidget(self.btn_apply_roi, 2, 0, 1, 4)
        
        roi_group.setLayout(roi_layout)
        right_panel.addWidget(roi_group)
        
        # Sample Teaching
        teaching_group = QGroupBox("Sample Teaching")
        teaching_layout = QVBoxLayout()
        
        self.btn_load_sample = QPushButton("Load Sample Image")
        self.btn_load_sample.clicked.connect(self.load_sample_image)
        teaching_layout.addWidget(self.btn_load_sample)
        
        self.teaching_status_label = QLabel("No sample loaded")
        self.teaching_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        teaching_layout.addWidget(self.teaching_status_label)
        
        self.btn_save_template = QPushButton("Save ROI as Template")
        self.btn_save_template.clicked.connect(self.save_roi_as_template)
        self.btn_save_template.setEnabled(False)
        teaching_layout.addWidget(self.btn_save_template)
        
        self.btn_clear_sample = QPushButton("Clear Sample / Resume Camera")
        self.btn_clear_sample.clicked.connect(self.clear_sample)
        self.btn_clear_sample.setEnabled(False)
        teaching_layout.addWidget(self.btn_clear_sample)
        
        # Template list
        self.template_list_label = QLabel("Saved Templates: 0")
        teaching_layout.addWidget(self.template_list_label)
        
        teaching_group.setLayout(teaching_layout)
        right_panel.addWidget(teaching_group)
        
        # Configuration management
        config_group = QGroupBox("Product Configuration")
        config_layout = QVBoxLayout()
        
        self.current_config_label = QLabel("No configuration loaded")
        config_layout.addWidget(self.current_config_label)
        
        config_buttons = QHBoxLayout()
        self.btn_load_config = QPushButton("Load Config")
        self.btn_load_config.clicked.connect(self.load_configuration)
        self.btn_save_config = QPushButton("Save Config")
        self.btn_save_config.clicked.connect(self.save_configuration)
        
        config_buttons.addWidget(self.btn_load_config)
        config_buttons.addWidget(self.btn_save_config)
        config_layout.addLayout(config_buttons)
        
        config_group.setLayout(config_layout)
        right_panel.addWidget(config_group)
        
        # Inspection control
        inspect_group = QGroupBox("Inspection Control")
        inspect_layout = QVBoxLayout()
        
        self.btn_single_inspect = QPushButton("Single Inspection")
        self.btn_single_inspect.clicked.connect(self.single_inspection)
        inspect_layout.addWidget(self.btn_single_inspect)
        
        self.chk_continuous = QPushButton("Continuous Inspection")
        self.chk_continuous.setCheckable(True)
        self.chk_continuous.clicked.connect(self.toggle_continuous_inspection)
        inspect_layout.addWidget(self.chk_continuous)
        
        inspect_group.setLayout(inspect_layout)
        right_panel.addWidget(inspect_group)
        
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
    def start_camera(self):
        """Start camera capture"""
        camera_idx = self.camera_select.currentIndex()
        self.camera = cv2.VideoCapture(camera_idx)
        
        if not self.camera.isOpened():
            QMessageBox.warning(self, "Camera Error", "Failed to open camera!")
            return
            
        self.camera_timer.start(33)  # ~30 FPS
        self.btn_start_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)
        
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.btn_start_camera.setEnabled(True)
        self.btn_stop_camera.setEnabled(False)
        
    def update_frame(self):
        """Update camera frame"""
        if self.teaching_mode:
            return  # Don't update camera when in teaching mode
            
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame
                self.roi_editor.set_image(frame)
                
                # If continuous inspection is enabled
                if self.chk_continuous.isChecked():
                    self.perform_inspection()
                    
    def on_roi_changed(self, roi: ROI):
        """Handle ROI change from editor"""
        self.spin_roi_x.setValue(roi.x)
        self.spin_roi_y.setValue(roi.y)
        self.spin_roi_width.setValue(roi.width)
        self.spin_roi_height.setValue(roi.height)
        self.update_config_roi()
        
    def apply_roi_from_spinboxes(self):
        """Apply ROI values from spinboxes to editor"""
        roi = ROI(
            x=self.spin_roi_x.value(),
            y=self.spin_roi_y.value(),
            width=self.spin_roi_width.value(),
            height=self.spin_roi_height.value()
        )
        self.roi_editor.add_roi(roi)
        self.update_config_roi()
        
    def update_config_roi(self):
        """Update config with current ROI"""
        self.config.CONNECTOR_ROI = {
            'x': self.spin_roi_x.value(),
            'y': self.spin_roi_y.value(),
            'width': self.spin_roi_width.value(),
            'height': self.spin_roi_height.value()
        }
        # Reinitialize inspection system with updated config
        self.inspection_system = ConnectorInspectionSystem(self.config)
        
    def on_threshold_changed(self, value):
        """Handle wire presence threshold change"""
        self.label_wire_threshold.setText(str(value))
        self.config.WIRE_PRESENCE_THRESHOLD = value
        self.inspection_system = ConnectorInspectionSystem(self.config)
        
    def on_min_area_changed(self, value):
        """Handle min area change"""
        self.config.WIRE_PRESENCE_MIN_AREA = value
        self.inspection_system = ConnectorInspectionSystem(self.config)
        
    def on_color_tolerance_changed(self, value):
        """Handle color tolerance change"""
        self.label_color_tolerance.setText(str(value))
        self.config.COLOR_MATCH_TOLERANCE = value
        self.inspection_system = ConnectorInspectionSystem(self.config)
        
    def on_color_confidence_changed(self, value):
        """Handle color confidence change"""
        self.label_color_confidence.setText(f"{value/100:.2f}")
        self.config.COLOR_MATCH_MIN_CONFIDENCE = value / 100
        self.inspection_system = ConnectorInspectionSystem(self.config)
        
    def single_inspection(self):
        """Perform single inspection on current frame"""
        if self.current_frame is not None:
            self.perform_inspection()
        else:
            QMessageBox.warning(self, "No Image", "No image available for inspection!")
            
    def toggle_continuous_inspection(self, checked):
        """Toggle continuous inspection mode"""
        if checked:
            self.chk_continuous.setText("Stop Continuous")
        else:
            self.chk_continuous.setText("Continuous Inspection")
            
    def perform_inspection(self):
        """Perform inspection on current frame"""
        if self.current_frame is None:
            return
            
        if self.inspection_system is None:
            self.inspection_system = ConnectorInspectionSystem(self.config)
            
        # Perform inspection
        result = self.inspection_system.inspect(self.current_frame)
        self.inspection_result = result
        
        # Update result display
        if result.is_pass:
            self.result_label.setText("OK")
            self.result_label.setStyleSheet("background-color: green; color: white; padding: 20px;")
            self.defect_label.setText("No defects detected")
        else:
            self.result_label.setText("NG")
            self.result_label.setStyleSheet("background-color: red; color: white; padding: 20px;")
            defects = ", ".join([d.name for d in result.defects_found])
            self.defect_label.setText(f"Defects: {defects}")
    
    def load_sample_image(self):
        """Load a sample image for teaching"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Sample Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if filename:
            image = cv2.imread(filename)
            if image is None:
                QMessageBox.warning(self, "Error", "Failed to load image!")
                return
                
            self.teaching_image = image
            self.teaching_mode = True
            self.current_frame = image
            self.roi_editor.set_image(image)
            
            self.teaching_status_label.setText(f"Sample: {Path(filename).name}")
            self.teaching_status_label.setStyleSheet("padding: 5px; background-color: #d4edda; color: #155724;")
            self.btn_save_template.setEnabled(True)
            self.btn_clear_sample.setEnabled(True)
            
            # Disable camera controls during teaching
            self.btn_start_camera.setEnabled(False)
            if self.camera and self.camera.isOpened():
                self.camera_timer.stop()
                
    def clear_sample(self):
        """Clear sample image and resume camera"""
        self.teaching_mode = False
        self.teaching_image = None
        
        self.teaching_status_label.setText("No sample loaded")
        self.teaching_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        self.btn_save_template.setEnabled(False)
        self.btn_clear_sample.setEnabled(False)
        
        # Re-enable camera
        self.btn_start_camera.setEnabled(True)
        if self.camera and self.camera.isOpened():
            self.camera_timer.start(33)
        else:
            self.roi_editor.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
            
    def save_roi_as_template(self):
        """Save current ROI as a reference template"""
        if self.teaching_image is None:
            QMessageBox.warning(self, "Error", "No sample image loaded!")
            return
            
        roi = self.roi_editor.get_active_roi()
        if roi is None:
            QMessageBox.warning(self, "Error", "Please define an ROI first!")
            return
            
        # Validate ROI bounds
        img_h, img_w = self.teaching_image.shape[:2]
        if not roi.is_valid((img_h, img_w)):
            QMessageBox.warning(self, "Error", "ROI is out of image bounds!")
            return
            
        roi_image = roi.extract_from_image(self.teaching_image)
        
        # Ask for template name
        template_name, ok = QInputDialog.getText(
            self, "Template Name", "Enter template name:"
        )
        
        if not ok or not template_name:
            return
            
        # Create templates directory if it doesn't exist
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        # Save template image
        template_filename = templates_dir / f"{template_name}.png"
        cv2.imwrite(str(template_filename), roi_image)
        
        # Update ROI metadata
        roi.name = template_name
        roi.mode = ROIMode.TEACHING
        roi.metadata = {
            'filename': str(template_filename),
            'created_from': 'teaching_mode',
            'parameters': {
                'threshold': self.config.WIRE_PRESENCE_THRESHOLD,
                'min_area': self.config.WIRE_PRESENCE_MIN_AREA,
                'color_tolerance': self.config.COLOR_MATCH_TOLERANCE,
                'color_confidence': self.config.COLOR_MATCH_MIN_CONFIDENCE
            }
        }
        
        # Create template metadata and save
        template_data = roi.to_dict()
        self.templates.append(template_data)
        self.update_template_list()
        
        QMessageBox.information(
            self, "Success", 
            f"Template '{template_name}' saved successfully!\n"
            f"Location: {template_filename}"
        )
        
    def update_template_list(self):
        """Update the template list display"""
        self.template_list_label.setText(f"Saved Templates: {len(self.templates)}")
            
    def save_configuration(self):
        """Save current configuration to JSON file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        
        if filename:
            config_data = {
                'CONNECTOR_ROI': self.config.CONNECTOR_ROI,
                'WIRE_PRESENCE_THRESHOLD': self.config.WIRE_PRESENCE_THRESHOLD,
                'WIRE_PRESENCE_MIN_AREA': self.config.WIRE_PRESENCE_MIN_AREA,
                'COLOR_MATCH_TOLERANCE': self.config.COLOR_MATCH_TOLERANCE,
                'COLOR_MATCH_MIN_CONFIDENCE': self.config.COLOR_MATCH_MIN_CONFIDENCE,
                'WIRE_SLOT_WIDTH': self.config.WIRE_SLOT_WIDTH,
                'WIRE_SLOT_HEIGHT': self.config.WIRE_SLOT_HEIGHT,
                'WIRE_SLOT_SPACING': self.config.WIRE_SLOT_SPACING,
                'WIRE_SLOT_START_X': self.config.WIRE_SLOT_START_X,
                'WIRE_SLOT_START_Y': self.config.WIRE_SLOT_START_Y,
                'templates': self.templates
            }
            
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            self.current_config_file = filename
            self.current_config_label.setText(f"Config: {Path(filename).name}")
            QMessageBox.information(self, "Success", "Configuration saved successfully!")
            
    def load_configuration(self):
        """Load configuration from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = json.load(f)
                    
                # Update config
                self.config.CONNECTOR_ROI = config_data.get('CONNECTOR_ROI', self.config.CONNECTOR_ROI)
                self.config.WIRE_PRESENCE_THRESHOLD = config_data.get('WIRE_PRESENCE_THRESHOLD', 50)
                self.config.WIRE_PRESENCE_MIN_AREA = config_data.get('WIRE_PRESENCE_MIN_AREA', 500)
                self.config.COLOR_MATCH_TOLERANCE = config_data.get('COLOR_MATCH_TOLERANCE', 60)
                self.config.COLOR_MATCH_MIN_CONFIDENCE = config_data.get('COLOR_MATCH_MIN_CONFIDENCE', 0.7)
                self.config.WIRE_SLOT_WIDTH = config_data.get('WIRE_SLOT_WIDTH', 180)
                self.config.WIRE_SLOT_HEIGHT = config_data.get('WIRE_SLOT_HEIGHT', 350)
                self.config.WIRE_SLOT_SPACING = config_data.get('WIRE_SLOT_SPACING', 200)
                self.config.WIRE_SLOT_START_X = config_data.get('WIRE_SLOT_START_X', 520)
                self.config.WIRE_SLOT_START_Y = config_data.get('WIRE_SLOT_START_Y', 320)
                
                # Load templates
                self.templates = config_data.get('templates', [])
                self.update_template_list()
                
                # Update UI
                self.slider_wire_threshold.setValue(self.config.WIRE_PRESENCE_THRESHOLD)
                self.spin_min_area.setValue(self.config.WIRE_PRESENCE_MIN_AREA)
                self.slider_color_tolerance.setValue(self.config.COLOR_MATCH_TOLERANCE)
                self.slider_color_confidence.setValue(int(self.config.COLOR_MATCH_MIN_CONFIDENCE * 100))
                
                self.spin_roi_x.setValue(self.config.CONNECTOR_ROI['x'])
                self.spin_roi_y.setValue(self.config.CONNECTOR_ROI['y'])
                self.spin_roi_width.setValue(self.config.CONNECTOR_ROI['width'])
                self.spin_roi_height.setValue(self.config.CONNECTOR_ROI['height'])
                
                # Set ROI in editor
                roi = ROI(
                    x=self.config.CONNECTOR_ROI['x'],
                    y=self.config.CONNECTOR_ROI['y'],
                    width=self.config.CONNECTOR_ROI['width'],
                    height=self.config.CONNECTOR_ROI['height']
                )
                self.roi_editor.add_roi(roi)
                
                # Reinitialize inspection system
                self.inspection_system = ConnectorInspectionSystem(self.config)
                
                self.current_config_file = filename
                self.current_config_label.setText(f"Config: {Path(filename).name}")
                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")
                
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_camera()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = InspectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
