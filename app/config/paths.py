"""
Centralized Path Configuration
Manages all file system paths for the application.
"""

from pathlib import Path


class DataPaths:
    """Centralized path configuration for data storage."""
    
    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "app" / "data"
    GOLDEN_DIR = DATA_DIR / "golden"
    TEMPLATES_DIR = GOLDEN_DIR / "templates"
    CONFIGS_DIR = GOLDEN_DIR / "configs"
    SAMPLES_DIR = DATA_DIR / "samples"
    OK_SAMPLES_DIR = SAMPLES_DIR / "ok"
    NG_SAMPLES_DIR = SAMPLES_DIR / "ng"
    
    # Log directory
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Ensure directories exist
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Path) and attr_name.endswith('_DIR'):
                attr.mkdir(parents=True, exist_ok=True)


def get_config_dir():
    """Get configuration directory path."""
    DataPaths.ensure_directories()
    return DataPaths.CONFIGS_DIR


def get_template_dir():
    """Get template directory path."""
    DataPaths.ensure_directories()
    return DataPaths.TEMPLATES_DIR
