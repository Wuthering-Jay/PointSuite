import sys
import os
import datetime

class DualLogger:
    """
    A logger that writes to both stdout/stderr and a file.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "a", encoding="utf-8")
        self.filepath = filepath

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def setup_logger(output_dir):
    """
    Sets up the logger to redirect stdout and stderr to a timestamped file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    
    logger = DualLogger(log_filepath)
    sys.stdout = logger
    sys.stderr = logger # Redirect stderr as well to capture errors
    
    print(f"{'='*80}")
    print(f"Logging started at {datetime.datetime.now()}")
    print(f"Log file: {log_filepath}")
    print(f"{'='*80}\n")
    
    return log_filepath
