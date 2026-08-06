import logging
from typing import Any, Optional
import pathlib
from datetime import datetime

# ANSI color codes
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
    'RESET': '\033[0m',      # Reset
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to logging output."""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        if fmt is None:
            fmt = '%(asctime)s %(levelname)s %(message)s'
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            # Get the level name and its corresponding color
            levelname = record.levelname
            color = COLORS.get(levelname, COLORS['RESET'])
            
            # Add color to the level name
            record.levelname = f"{color}{levelname}{COLORS['RESET']}"
        
        # Format the message
        return super().format(record)

def setup_colored_logging(level: int = logging.INFO, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
    """Set up colored logging for the application.
    
    Args:
        level: Logging level (default: logging.INFO)
        fmt: Log format string (default: '%(asctime)s %(levelname)s %(message)s')
        datefmt: Date format string (default: '%Y-%m-%d %H:%M:%S')
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate output
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = ColoredFormatter(fmt, datefmt, use_colors=True)
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)

def setup_logging_with_file(
    level: int = logging.INFO, 
    fmt: Optional[str] = None, 
    datefmt: Optional[str] = None,
    log_file: Optional[pathlib.Path] = None,
    file_level: Optional[int] = None
) -> None:
    """Set up logging with both console and file handlers.
    
    Args:
        level: Logging level for console (default: logging.INFO)
        fmt: Log format string (default: '%(asctime)s %(levelname)s %(message)s')
        datefmt: Date format string (default: '%Y-%m-%d %H:%M:%S')
        log_file: Path to log file (optional)
        file_level: Logging level for file (defaults to same as console level)
    """
    if file_level is None:
        file_level = level
        
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(min(level, file_level))
    
    # Remove any existing handlers to avoid duplicate output
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create colored formatter for console
    console_formatter = ColoredFormatter(fmt, datefmt, use_colors=True)
    console_handler.setFormatter(console_formatter)
    
    # Add console handler to the logger
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure the log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler with append mode to reuse existing log files
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(file_level)
        
        # Create colored formatter for file (modern editors handle ANSI colors well)
        file_formatter = ColoredFormatter(fmt, datefmt, use_colors=True)
        file_handler.setFormatter(file_formatter)
        
        # Add file handler to the logger
        logger.addHandler(file_handler)
        
        # Add colored session separator if file already exists to distinguish runs
        if log_file.exists() and log_file.stat().st_size > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            separator = f"\n{COLORS['INFO']}{'='*20} NEW SESSION {timestamp} {'='*20}{COLORS['RESET']}"
            with open(log_file, 'a') as f:
                f.write(separator + '\n')

