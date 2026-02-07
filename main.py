"""
Industrial Vision Inspection System - Entry Point
Main entry point for the wire order inspection application.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from app.utils.logger import setup_logging
from app.ui.main_window import InspectionMainWindow


def main():
    """
    Application entry point.
    Initializes logging, creates Qt application, and shows main window.
    """
    # Setup centralized logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Industrial Vision Inspection System - Starting")
    logger.info("=" * 80)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Wire Order Inspection System")
    app.setOrganizationName("Industrial Automation")
    
    # Create and show main window
    try:
        window = InspectionMainWindow()
        window.show()
        
        logger.info("Application window created successfully")
        logger.info("System ready for operation")
        
        # Run application event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
