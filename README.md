# AOI Connector Inspection System

**Automated Optical Inspection (AOI) system for electrical connector quality control**

A production-ready, modular computer vision system for inspecting electrical connectors with multiple wires. Detects defects in real-time using classical image processing techniques with OpenCV.

---

## 🎯 Features

### Defect Detection Capabilities
- **Missing Wire Detection** - Identifies empty connector slots
- **Wire Order Verification** - Validates correct wire color sequence
- **Connector Position Detection** - Detects misalignment/offset
- **Insertion Depth Analysis** - Ensures wires are fully inserted

### System Characteristics
- ✅ **Fast Processing**: Target < 100ms per image
- ✅ **No Deep Learning**: Classical CV techniques only
- ✅ **ROI-Based**: Optimized region-of-interest processing
- ✅ **Modular Design**: Easy to customize and extend
- ✅ **Production Ready**: Error handling, logging, batch processing
- ✅ **Real-time Capable**: Suitable for inline inspection

---

## 📋 Requirements

### Hardware
- Fixed camera position
- Controlled lighting environment
- Known product geometry

### Software
```
Python >= 3.7
OpenCV >= 4.5.0
NumPy >= 1.19.0
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
cd ImageDetection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows: (Git bash)
source venv/Scripts/activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or install manually:
pip install opencv-python numpy
```

### Basic Usage

```python
from main import ConnectorInspectionSystem

# Initialize inspection system
inspector = ConnectorInspectionSystem()

# Inspect single image
result = inspector.inspect_from_file("path/to/image.jpg")

# Check result
print(f"Status: {result.status}")  # OK or NG
print(f"Defect: {result.defect_type}")
print(f"Time: {result.processing_time_ms:.2f} ms")
```

### Run Demo
```bash
python main.py
```

---

## 📁 Project Structure

```
ImageDetection/
│
├── main.py                 # Main inspection system & orchestration
├── config.py               # Configuration & parameters
├── inspectors.py           # Defect detection modules
├── image_utils.py          # Image processing utilities
├── examples.py             # Usage examples & integration patterns
├── README.md               # This file
│
└── Samples/                # Test images directory
    ├── connector_test.jpg
    ├── golden_sample.jpg
    └── ...
```

---

## 🔧 Configuration

Edit `config.py` to customize inspection parameters:

```python
from config import InspectionConfig

config = InspectionConfig()

# Connector geometry
config.NUM_WIRES = 4  # Number of wires in connector
config.WIRE_COLORS = [  # Expected wire colors (BGR)
    ('red', [0, 0, 255]),
    ('black', [0, 0, 0]),
    ('white', [255, 255, 255]),
    ('blue', [255, 0, 0])
]

# ROI positions (adjust for your setup)
config.CONNECTOR_ROI = {'x': 500, 'y': 300, 'width': 800, 'height': 400}
config.WIRE_SLOT_START_X = 520
config.WIRE_SLOT_START_Y = 320
config.WIRE_SLOT_SPACING = 200

# Detection thresholds
config.POSITION_MAX_OFFSET_X = 15  # pixels
config.POSITION_MAX_OFFSET_Y = 15
config.WIRE_PRESENCE_MIN_AREA = 500
config.INSERTION_MIN_DEPTH = 280

# Enable/disable specific checks
config.ENABLE_POSITION_CHECK = True
config.ENABLE_WIRE_PRESENCE_CHECK = True
config.ENABLE_WIRE_ORDER_CHECK = True
config.ENABLE_INSERTION_DEPTH_CHECK = True
```

---

## 💡 Usage Examples

### Example 1: Basic Inspection
```python
from main import ConnectorInspectionSystem

inspector = ConnectorInspectionSystem()
result = inspector.inspect_from_file("Samples/connector_001.jpg")
print(result)
```

### Example 2: Batch Processing
```python
image_list = [
    "Samples/connector_001.jpg",
    "Samples/connector_002.jpg",
    "Samples/connector_003.jpg"
]

results = inspector.inspect_batch(image_list)
```

### Example 3: Camera Integration
```python
import cv2

# Capture from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Inspect frame
result = inspector.inspect(frame, image_id="camera_001")

if result.status == "NG":
    print(f"Defect detected: {result.defect_type}")
    # Trigger reject mechanism
```

### Example 4: Golden Sample Calibration
```python
# Calibrate with known good sample
inspector.calibrate_from_golden_sample("Samples/golden_sample.jpg")

# Now inspect test parts
result = inspector.inspect_from_file("test_part.jpg")
```

### Example 5: Custom Configuration
```python
from config import InspectionConfig

config = InspectionConfig()
config.POSITION_MAX_OFFSET_X = 20  # More tolerant
config.SAVE_DEBUG_IMAGES = True     # Save debug output

inspector = ConnectorInspectionSystem(config)
```

**See `examples.py` for 10+ comprehensive usage examples!**

---

## 🏗️ Architecture

### System Overview

```
Input Image
    ↓
┌─────────────────────────┐
│  ConnectorInspection    │
│  System (main.py)       │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Image Preprocessing    │
│  (image_utils.py)       │
└─────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           Inspector Modules                 │
│  (inspectors.py)                            │
├─────────────────────────────────────────────┤
│  • WirePresenceInspector                    │
│  • WireOrderInspector                       │
│  • ConnectorPositionInspector               │
│  • WireInsertionInspector                   │
└─────────────────────────────────────────────┘
    ↓
InspectionResult
    ↓
OK / NG + Defect Type
```

### Key Components

#### 1. **config.py** - Configuration Management
- `InspectionConfig`: All tunable parameters
- `DefectTypes`: Defect type enumeration
- `InspectionResult`: Result data structure

#### 2. **image_utils.py** - Image Processing Utilities
- `ROIExtractor`: ROI extraction and management
- `ImagePreprocessor`: Image preprocessing (blur, threshold, edge detection)
- `ColorAnalyzer`: Color-based wire identification
- `VisualizationHelper`: Result visualization

#### 3. **inspectors.py** - Defect Detection Modules
- `WirePresenceInspector`: Detects missing wires using contour analysis
- `WireOrderInspector`: Validates wire color sequence
- `ConnectorPositionInspector`: Detects position offset using template matching or centroid
- `WireInsertionInspector`: Measures insertion depth using edge detection
- `InspectorFactory`: Creates and manages inspector instances

#### 4. **main.py** - Main Inspection System
- `ConnectorInspectionSystem`: Orchestrates complete inspection workflow
- Handles image loading, batch processing, result output
- Performance monitoring and debug output

---

## 🎨 Inspection Workflow

```python
1. Load Image
2. [Optional] Enhance Contrast
3. Check Connector Position
    ├─ Template matching OR centroid calculation
    └─ Validate offset within tolerance
4. Check Wire Presence (for each slot)
    ├─ Extract ROI
    ├─ Apply threshold
    ├─ Find contours
    └─ Validate minimum area
5. Check Wire Order (for each wire)
    ├─ Extract dominant color
    ├─ Match to expected colors
    └─ Validate confidence
6. Check Insertion Depth (for each wire)
    ├─ Edge detection
    ├─ Measure wire length
    └─ Validate minimum depth
7. Compile Results
8. Output: OK/NG + Defect Type
```

---

## ⚡ Performance Optimization

### Tips for <100ms Processing

1. **Optimize ROI Sizes**
   ```python
   config.WIRE_SLOT_WIDTH = 150  # Smaller = faster
   config.WIRE_SLOT_HEIGHT = 300
   ```

2. **Reduce Preprocessing**
   ```python
   config.GAUSSIAN_BLUR_KERNEL = (3, 3)  # Smaller kernel
   ```

3. **Selective Inspection**
   ```python
   # Disable non-critical checks
   config.ENABLE_WIRE_ORDER_CHECK = False
   ```

4. **Disable Debug Output**
   ```python
   config.SAVE_DEBUG_IMAGES = False
   config.SHOW_VISUALIZATION = False
   ```

5. **Use NumPy Optimizations**
   - Already implemented in code
   - Vectorized operations where possible

### Benchmark Performance
Run performance test:
```python
python examples.py  # Uncomment example_9_performance_optimization()
```

---

## 🔍 Defect Detection Algorithms

### Missing Wire Detection
- **Method**: Contour area analysis
- **Algorithm**: 
  1. Convert ROI to grayscale
  2. Apply binary threshold
  3. Morphological operations (open/close)
  4. Find contours
  5. Check max contour area vs. threshold

### Wire Order Detection
- **Method**: Color matching
- **Algorithm**:
  1. Extract dominant color from wire ROI
  2. Calculate Euclidean distance to expected colors
  3. Match to closest color within tolerance
  4. Validate confidence score

### Position Detection
- **Method**: Template matching or centroid
- **Algorithm** (Template):
  1. Match reference template to current image
  2. Find best match location
  3. Calculate offset from expected center
  4. Validate offset within tolerance

### Insertion Depth Detection
- **Method**: Edge-based length measurement
- **Algorithm**:
  1. Canny edge detection on wire ROI
  2. Vertical edge projection
  3. Find first/last significant edge rows
  4. Calculate span as insertion depth
  5. Validate against minimum depth

---

## 🔌 Integration Patterns

### PLC Integration
```python
# Wait for trigger from PLC
while plc.wait_for_trigger():
    image = camera.grab_frame()
    result = inspector.inspect(image)
    
    # Send result to PLC
    plc.write("result", 1 if result.status == "OK" else 0)
    plc.write("defect_code", encode_defect(result.defect_type))
```

### Camera Integration
```python
# With industrial camera
from pypylon import pylon

camera = pylon.InstantCamera()
camera.Open()
camera.StartGrabbing()

grab_result = camera.RetrieveResult(5000)
if grab_result.GrabSucceeded():
    image = grab_result.Array
    result = inspector.inspect(image)
```

### Database Logging
```python
# Log results to database
import sqlite3

def log_result(result, image_id):
    conn = sqlite3.connect('inspection_log.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inspections (image_id, status, defect_type, time_ms)
        VALUES (?, ?, ?, ?)
    ''', (image_id, result.status, result.defect_type, result.processing_time_ms))
    conn.commit()
```

---

## 🛠️ Calibration & Setup

### Initial Setup Steps

1. **Position Camera**
   - Fix camera at optimal distance/angle
   - Ensure consistent lighting

2. **Capture Golden Sample**
   ```python
   # Save image of perfect connector
   cv2.imwrite("Samples/golden_sample.jpg", camera_frame)
   ```

3. **Determine ROI Positions**
   - Use visualization to identify ROI coordinates
   ```python
   config.SHOW_VISUALIZATION = True
   inspector.inspect_from_file("golden_sample.jpg")
   # Adjust ROI positions in config.py
   ```

4. **Calibrate System**
   ```python
   inspector.calibrate_from_golden_sample("Samples/golden_sample.jpg")
   ```

5. **Tune Thresholds**
   - Test with known defect samples
   - Adjust thresholds to minimize false positives/negatives

6. **Performance Test**
   - Run batch test to verify processing time
   - Optimize if needed

---

## 📊 Output Format

### InspectionResult Object
```python
result.status            # "OK" or "NG"
result.defect_type       # DefectTypes enum value
result.defects           # List of all detected defects
result.processing_time_ms  # Processing time in milliseconds
result.details           # Dictionary with detailed info per check
```

### Example Output
```
Status: NG
Defect Type: MISSING_WIRE
Processing Time: 45.23 ms
Detected Defects: MISSING_WIRE
Details:
  MISSING_WIRE: Missing wires at positions: [2]
```

---

## 🚧 Troubleshooting

### Common Issues

**Issue**: False positives for missing wires
- **Solution**: Adjust `WIRE_PRESENCE_MIN_AREA` threshold
- Check lighting conditions

**Issue**: Wrong wire colors detected
- **Solution**: Calibrate `WIRE_COLORS` BGR values
- Increase `COLOR_MATCH_TOLERANCE`

**Issue**: Position offset always detected
- **Solution**: Recalibrate with golden sample
- Check `CONNECTOR_ROI` coordinates

**Issue**: Processing time > 100ms
- **Solution**: See Performance Optimization section
- Reduce ROI sizes
- Disable non-essential checks

**Issue**: No image display
- **Solution**: Set `config.SHOW_VISUALIZATION = True`
- Ensure OpenCV GUI support is installed

---

## 🔮 Future Enhancements

Potential improvements for production deployment:

- [ ] Multi-threading for parallel ROI processing
- [ ] GPU acceleration with OpenCV CUDA
- [ ] Adaptive thresholding based on lighting conditions
- [ ] Statistical Process Control (SPC) charting
- [ ] Web-based dashboard for monitoring
- [ ] Integration with MES/ERP systems
- [ ] Automated threshold tuning
- [ ] Support for different connector types
- [ ] 3D depth sensing for insertion verification

---

## 📝 License

This code is provided as-is for industrial automation and quality control applications.

---

## 👤 Author

Developed as an industrial AOI solution for electrical connector inspection.

---

## 📞 Support

For issues or questions:
1. Check the `examples.py` file for usage patterns
2. Review troubleshooting section above
3. Verify configuration parameters in `config.py`

---

## 🙏 Acknowledgments

Built with:
- OpenCV - Computer vision library
- NumPy - Numerical computing
- Python - Programming language

---

**Ready for Production Deployment** ✅

Start inspecting connectors with:
```bash
python main.py
```
