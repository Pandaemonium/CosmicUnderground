import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Global debug state
_debug_mode = False
_log_file = None

def set_debug_mode(enabled: bool, log_file_path: str = None):
    """Enable or disable debug mode and optionally set a log file"""
    global _debug_mode, _log_file
    
    _debug_mode = enabled
    
    if log_file_path and enabled:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Open log file for writing
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            
            # Write header
            _log_file.write(f"=== COSMIC UNDERGROUND DEBUG LOG ===\n")
            _log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _log_file.write(f"Debug mode: {enabled}\n")
            _log_file.write("=" * 50 + "\n\n")
            _log_file.flush()
            
            print(f"🎵 Debug mode enabled, logging to: {log_file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not open log file {log_file_path}: {e}")
            _log_file = None
    else:
        if _log_file:
            try:
                _log_file.close()
            except:
                pass
            _log_file = None

def _write_log(level: str, message: str):
    """Write a message to the log file and console"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # HH:MM:SS.mmm
    formatted_message = f"[{timestamp}] {level}: {message}"
    
    # Always print to console
    print(formatted_message)
    
    # Write to log file if available
    if _log_file:
        try:
            _log_file.write(formatted_message + "\n")
            _log_file.flush()  # Ensure it's written immediately
        except Exception as e:
            print(f"⚠️ Warning: Could not write to log file: {e}")

def debug_enter(function_name: str, file_name: str, **kwargs):
    """Log function entry with parameters"""
    if not _debug_mode:
        return
    
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    message = f"ENTER {file_name}:{function_name}()"
    if params:
        message += f" - {params}"
    
    _write_log("DEBUG", message)

def debug_exit(function_name: str, file_name: str, result: str = None):
    """Log function exit with optional result"""
    if not _debug_mode:
        return
    
    message = f"EXIT {file_name}:{function_name}()"
    if result:
        message += f" -> {result}"
    
    _write_log("DEBUG", message)

def debug_file_op(operation: str, file_path: str, success: bool, details: str = None):
    """Log file operation results"""
    if not _debug_mode:
        return
    
    status = "SUCCESS" if success else "FAILED"
    message = f"FILE {operation}: {file_path} - {status}"
    if details:
        message += f" ({details})"
    
    _write_log("FILE", message)

def debug_info(message: str):
    """Log general debug information"""
    if not _debug_mode:
        return
    
    _write_log("INFO", message)

def debug_warning(message: str):
    """Log warning messages"""
    if not _debug_mode:
        return
    
    _write_log("WARN", message)

def debug_error(message: str, exception: Exception = None):
    """Log error messages with optional exception details"""
    if not _debug_mode:
        return
    
    message_text = message
    if exception:
        message_text += f" - Exception: {type(exception).__name__}: {exception}"
    
    _write_log("ERROR", message_text)

def close_log():
    """Close the log file and clean up"""
    global _log_file
    
    if _log_file:
        try:
            _log_file.write(f"\n=== LOG CLOSED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            _log_file.close()
            _log_file = None
            print("🎵 Debug log file closed")
        except Exception as e:
            print(f"⚠️ Warning: Could not close log file: {e}")

