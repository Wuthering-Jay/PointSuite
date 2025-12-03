import sys
import os
import datetime


# ============================================================================
# ANSI Colors (终端颜色输出)
# ============================================================================

class Colors:
    """ANSI 颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


# ============================================================================
# 美化输出函数
# ============================================================================

def print_header(title: str, emoji: str = "🚀"):
    """打印美化的标题"""
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {emoji} {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 50}{Colors.RESET}")


def print_config(configs: dict, title: str = "配置"):
    """打印配置信息（字典）"""
    print_section(title)
    max_key_len = max(len(str(k)) for k in configs.keys()) if configs else 0
    for key, value in configs.items():
        print(f"  {Colors.DIM}├─{Colors.RESET} {key:<{max_key_len}}: {Colors.GREEN}{value}{Colors.RESET}")


def format_size(size_bytes: float) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def format_number(n: int) -> str:
    """格式化数字 (K/M/B)"""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ============================================================================
# Dual Logger (stdout + file)
# ============================================================================

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
