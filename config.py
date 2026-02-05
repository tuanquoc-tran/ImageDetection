"""
Configuration file for AOI Connector Inspection System
Contains all inspection parameters and thresholds
"""

class InspectionConfig:
    """Configuration class for connector inspection parameters"""
    
    # ==================== Image Acquisition ====================
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    
    # ==================== Connector Geometry ====================
    # Expected number of wires in the connector
    NUM_WIRES = 4
    
    # Expected wire colors in correct order (BGR format)
    WIRE_COLORS = [
        ('red', [0, 0, 255]),
        ('black', [0, 0, 0]),
        ('white', [255, 255, 255]),
        ('blue', [255, 0, 0])
    ]
    
    # ==================== ROI Definitions ====================
    # Connector body ROI for position detection
    CONNECTOR_ROI = {
        'x': 500,
        'y': 300,
        'width': 800,
        'height': 400
    }
    
    # Individual wire slot ROIs (will be calculated dynamically)
    WIRE_SLOT_WIDTH = 180
    WIRE_SLOT_HEIGHT = 350
    WIRE_SLOT_SPACING = 200
    WIRE_SLOT_START_X = 520
    WIRE_SLOT_START_Y = 320
    
    # ==================== Detection Thresholds ====================
    
    # Missing Wire Detection
    WIRE_PRESENCE_MIN_AREA = 500  # Minimum contour area to consider wire present
    WIRE_PRESENCE_THRESHOLD = 50  # Binary threshold for wire detection
    
    # Wire Order Detection (Color matching)
    COLOR_MATCH_TOLERANCE = 60  # HSV color matching tolerance
    COLOR_MATCH_MIN_CONFIDENCE = 0.7  # Minimum confidence for color match
    
    # Connector Position Detection
    POSITION_MAX_OFFSET_X = 15  # Maximum allowed X offset in pixels
    POSITION_MAX_OFFSET_Y = 15  # Maximum allowed Y offset in pixels
    POSITION_TEMPLATE_MATCH_THRESHOLD = 0.75  # Template matching threshold
    
    # Wire Insertion Depth Detection
    INSERTION_MIN_DEPTH = 280  # Minimum expected wire length in ROI (pixels)
    INSERTION_DEPTH_TOLERANCE = 30  # Allowed variation in insertion depth
    INSERTION_EDGE_THRESHOLD = 100  # Edge detection threshold
    
    # ==================== Image Processing Parameters ====================
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MORPHOLOGY_KERNEL_SIZE = (3, 3)
    CANNY_THRESHOLD_LOW = 50
    CANNY_THRESHOLD_HIGH = 150
    
    # ==================== Inspection Workflow ====================
    ENABLE_POSITION_CHECK = True
    ENABLE_WIRE_PRESENCE_CHECK = True
    ENABLE_WIRE_ORDER_CHECK = True
    ENABLE_INSERTION_DEPTH_CHECK = True
    
    # ==================== Output Settings ====================
    SAVE_DEBUG_IMAGES = False  # Set to True for debugging
    DEBUG_IMAGE_PATH = "./debug_output/"
    SHOW_VISUALIZATION = False  # Set to True to display results
    
    # ==================== Performance ====================
    TARGET_PROCESSING_TIME_MS = 100
    
    # ==================== Reference Images ====================
    REFERENCE_CONNECTOR_PATH = None  # Path to reference connector image for template matching
    CALIBRATION_IMAGE_PATH = None  # Path to calibration/golden sample


class DefectTypes:
    """Enumeration of possible defect types"""
    OK = "OK"
    MISSING_WIRE = "MISSING_WIRE"
    WRONG_WIRE_ORDER = "WRONG_WIRE_ORDER"
    POSITION_OFFSET = "POSITION_OFFSET"
    INSUFFICIENT_INSERTION = "INSUFFICIENT_INSERTION"
    MULTIPLE_DEFECTS = "MULTIPLE_DEFECTS"


class InspectionResult:
    """Data class for inspection results"""
    
    def __init__(self):
        self.status = "NG"  # OK or NG
        self.defect_type = None
        self.defects = []  # List of all detected defects
        self.processing_time_ms = 0.0
        self.details = {}  # Additional details for each check
        
    def add_defect(self, defect_type, details=None):
        """Add a defect to the result"""
        self.defects.append(defect_type)
        if details:
            self.details[defect_type] = details
            
    def finalize(self):
        """Finalize the inspection result"""
        if len(self.defects) == 0:
            self.status = "OK"
            self.defect_type = DefectTypes.OK
        elif len(self.defects) == 1:
            self.status = "NG"
            self.defect_type = self.defects[0]
        else:
            self.status = "NG"
            self.defect_type = DefectTypes.MULTIPLE_DEFECTS
            
    def __str__(self):
        """String representation of the result"""
        result_str = f"Status: {self.status}\n"
        result_str += f"Defect Type: {self.defect_type}\n"
        result_str += f"Processing Time: {self.processing_time_ms:.2f} ms\n"
        if self.defects:
            result_str += f"Detected Defects: {', '.join(self.defects)}\n"
        if self.details:
            result_str += "Details:\n"
            for key, value in self.details.items():
                result_str += f"  {key}: {value}\n"
        return result_str
