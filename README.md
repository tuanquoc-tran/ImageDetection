# Industrial Vision Inspection System

Production-grade wire order inspection system with automatic ROI localization and deterministic state management.

## Features


## Project Structure

```
ImageDetection\
├── main.py                          # Application entry point
│
├── app/
│   ├── controller/
│   │   ├── behavior_controller.py   # State machine
│   │   └── app_controller.py        # Application flow controller
│   │
│   ├── ui/
│   │   ├── main_window.py           # Main PyQt6 window
│   │   └── roi_editor.py            # ROI drawing widget
│   │
│   ├── vision/
│   │   ├── image_source.py          # Camera/file abstraction
│   │   ├── roi_locator.py           # Template matching
│   │   ├── color_detector.py        # HSV color detection
│   │   ├── wire_segmenter.py        # Wire segment extraction
│   │   ├── inspector.py             # Main inspection logic
│   │   └── golden_sample.py         # Sample management
│   │
│   ├── config/
│   │   └── paths.py                 # Centralized path config
│   │
│   ├── utils/
│   │   ├── logger.py                # Logging setup
│   │   └── timer.py                 # Performance profiling
│   │
│   └── data/
│       └── golden/
│           ├── configs/             # Golden sample configs
│           └── templates/           # Template images
│
├── logs/                            # Application logs
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites and setup for develop

- Python 3.9 or higher
- Camera device (optional, for camera mode)


```bash
# Create virtual environment (recommended)
# Windows (git bash)
python -m venv venv
venv\Scripts\activate

#Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
python main.py
```

### Workflow

1. **Teach Golden Sample** (First Time Only)
   - Click "Load Sample Image" or "Teach from Camera"
   - Define ROI using mouse (connector area for template, wire area for inspection)
   - Enter golden sample name
   - Click "Save Golden Sample"
   - Expected wire colors are automatically detected

2. **Image Mode Inspection**
   - Select "Image Mode"
   - Click "Load Image"
   - Click "Single Inspection"
   - View result (OK/NG) with defect details

3. **Camera Mode Inspection**
   - Select "Camera Mode"
   - Click "Start Camera"
   - Click "Single Inspection" or enable "Continuous Inspection"
   - View real-time results

### State Machine

```
IDLE → IMAGE_MODE/CAMERA_MODE → TEACH_SAMPLE → READY → INSPECTION → (back to mode)
                                                   ↓
                                               ERROR → Recovery
```

## Configuration

Golden sample configurations are stored in:
```
app/data/golden/configs/golden_sample.json
```

## System Requirements


## Logging

Logs are stored in `logs/inspection_YYYYMMDD.log`


## Troubleshooting

### "Golden sample file not found"
- Run teaching mode first to create a golden sample

### "Template match failed"
- Part may be misaligned - check product placement
- Re-teach golden sample with better template area

### "ROI out of bounds"
- Image size differs from golden sample
- Re-teach with correct image resolution

### Camera not opening
- Check camera device index (0, 1, 2)
- Verify camera is not in use by another application
- Check USB connection


## License

Proprietary - Industrial Automation Team

## Support

For issues or questions, contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: 2024-02-06
