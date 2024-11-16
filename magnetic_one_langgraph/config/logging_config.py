# magnetic_one_langgraph/config/logging_config.py

import logging.config
from pathlib import Path

def setup_logging(log_dir: Path) -> None:
    """Configure logging for the application."""
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": str(log_dir / "app.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": str(log_dir / "error.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "ERROR"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file", "error_file"],
                "level": "INFO",
                "propagate": True
            },
            "magnetic_one_langgraph": {
                "handlers": ["console", "file", "error_file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)
