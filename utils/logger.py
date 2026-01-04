
import logging
import sys
from datetime import datetime
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     
        'INFO': '\033[32m',       
        'WARNING': '\033[33m',    
        'ERROR': '\033[31m',      
        'CRITICAL': '\033[35m',   
        'RESET': '\033[0m'        
    }
    
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        icon = self.ICONS.get(record.levelname, '')
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        formatted = f"{color}[{timestamp}] {icon} {record.levelname:8}{reset} | {record.name}: {record.getMessage()}"
        
        return formatted


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter())
        
        logger.addHandler(console_handler)
    
    return logger
    
    if not logger.handlers:
        logger.setLevel(level)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter())
        
        logger.addHandler(console_handler)
    
    return logger


def get_pipeline_logger():
    return setup_logger('pipeline')

def get_hdr_logger():
    return setup_logger('hdr')

def get_classifier_logger():
    return setup_logger('classifier')

def get_watcher_logger():
    return setup_logger('watcher')
