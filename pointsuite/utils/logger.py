"""
PointSuite 统一日志系统

提供统一的终端输出风格和颜色控制:
- 颜色输出: INFO/SUCCESS/WARNING/ERROR/DEBUG
- 美化输出: 标题、章节、配置表格
- 双重日志: 同时写入终端和文件
"""

import sys
import os
import datetime
from typing import Dict, Any, Optional, Union, List


# ============================================================================
# ANSI Colors (终端颜色输出)
# ============================================================================

class Colors:
    """ANSI 颜色代码"""
    # 基础颜色
    HEADER = '\033[95m'      # 紫红色
    BLUE = '\033[94m'        # 蓝色
    CYAN = '\033[96m'        # 青色
    GREEN = '\033[92m'       # 绿色
    YELLOW = '\033[93m'      # 黄色
    RED = '\033[91m'         # 红色
    MAGENTA = '\033[35m'     # 洋红色
    WHITE = '\033[97m'       # 白色
    
    # 样式
    BOLD = '\033[1m'         # 粗体
    DIM = '\033[2m'          # 暗淡
    ITALIC = '\033[3m'       # 斜体
    UNDERLINE = '\033[4m'    # 下划线
    
    # 重置
    RESET = '\033[0m'
    
    # 语义颜色 (便于使用)
    INFO = CYAN              # 信息
    SUCCESS = GREEN          # 成功
    WARNING = YELLOW         # 警告
    ERROR = RED              # 错误
    DEBUG = DIM              # 调试
    HIGHLIGHT = BOLD + CYAN  # 高亮


# ============================================================================
# 日志级别输出函数
# ============================================================================

def _display_width(s: str) -> int:
    """
    计算字符串的显示宽度
    - 中文/全角字符占 2 个宽度
    - 其他字符占 1 个宽度
    """
    import unicodedata
    width = 0
    for c in s:
        # 使用 unicodedata 获取东亚宽度
        ea_width = unicodedata.east_asian_width(c)
        if ea_width in ('F', 'W'):  # Fullwidth, Wide
            width += 2
        elif ea_width == 'A':  # Ambiguous - 在东亚环境中通常是宽字符
            width += 2
        else:
            width += 1
    return width


def _pad_to_width(s: str, target_width: int, fill_char: str = ' ') -> str:
    """
    将字符串填充到指定显示宽度
    """
    current_width = _display_width(s)
    if current_width >= target_width:
        return s
    return s + fill_char * (target_width - current_width)


def log_info(message: str, prefix: str = "[INFO] "):
    """输出信息级别日志 (青色)"""
    print(f"{Colors.INFO}{prefix}{message}{Colors.RESET}")


def log_success(message: str, prefix: str = "[OK] "):
    """输出成功级别日志 (绿色)"""
    print(f"{Colors.SUCCESS}{prefix}{message}{Colors.RESET}")


def log_warning(message: str, prefix: str = "[WARN] "):
    """输出警告级别日志 (黄色)"""
    print(f"{Colors.WARNING}{prefix}{message}{Colors.RESET}")


def log_error(message: str, prefix: str = "[ERROR] "):
    """输出错误级别日志 (红色)"""
    print(f"{Colors.ERROR}{prefix}{message}{Colors.RESET}")


def log_debug(message: str, prefix: str = "[DEBUG] "):
    """输出调试级别日志 (暗淡)"""
    print(f"{Colors.DEBUG}{prefix}{message}{Colors.RESET}")


def log_step(message: str, prefix: str = "  -> "):
    """输出步骤信息 (用于流程中的步骤)"""
    print(f"{Colors.DIM}{prefix}{Colors.RESET}{message}")


# ============================================================================
# 美化输出函数
# ============================================================================

def print_header(title: str, width: int = 70):
    """
    打印美化的主标题
    
    ══════════════════════════════════════════════════════════════════════
      标题内容
    ══════════════════════════════════════════════════════════════════════
    """
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * width}{Colors.RESET}")


def print_section(title: str, width: int = 50):
    """
    打印章节标题
    
    ──────────────────────────────────────────────────
      章节标题
    ──────────────────────────────────────────────────
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * width}{Colors.RESET}")


def print_config(configs: Dict[str, Any], title: str = "配置"):
    """
    打印配置信息（字典形式）
    
    ──────────────────────────────────────────────────
      配置
    ──────────────────────────────────────────────────
      ├─ 键1      : 值1
      ├─ 键2      : 值2
      └─ 键3      : 值3
    """
    print_section(title)
    if not configs:
        print(f"  {Colors.DIM}(空){Colors.RESET}")
        return
    
    max_key_len = max(len(str(k)) for k in configs.keys())
    items = list(configs.items())
    
    for i, (key, value) in enumerate(items):
        # 最后一项用 └─，其他用 ├─
        connector = "└─" if i == len(items) - 1 else "├─"
        # 根据值类型选择颜色
        value_color = _get_value_color(value)
        print(f"  {Colors.DIM}{connector}{Colors.RESET} {key:<{max_key_len}}: {value_color}{value}{Colors.RESET}")


def print_kv(key: str, value: Any, indent: int = 2, connector: str = "├─"):
    """打印单个键值对"""
    spaces = " " * indent
    value_color = _get_value_color(value)
    print(f"{spaces}{Colors.DIM}{connector}{Colors.RESET} {key}: {value_color}{value}{Colors.RESET}")


def print_box(title: str, content: Dict[str, Any], width: int = 70):
    """
    打印带边框的信息框（正确处理中文字符宽度）
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║  标题                                                              ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  键1: 值1                                                          ║
    ║  键2: 值2                                                          ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    # 边框字符 ║ 占 1 个显示宽度，所以内容宽度 = width - 2
    inner_width = width - 2
    
    print()
    print(f"{Colors.CYAN}╔{'═' * (width - 2)}╗{Colors.RESET}")
    
    # 标题行 - 使用显示宽度计算填充
    title_text = f"  {title}"
    title_padded = _pad_to_width(title_text, inner_width)
    print(f"{Colors.CYAN}║{Colors.BOLD}{title_padded}{Colors.RESET}{Colors.CYAN}║{Colors.RESET}")
    
    if content:
        print(f"{Colors.CYAN}╠{'═' * (width - 2)}╣{Colors.RESET}")
        
        for key, value in content.items():
            line = f"  {key}: {value}"
            # 截断过长的行（按显示宽度）
            line_width = _display_width(line)
            if line_width > inner_width:
                # 按显示宽度截断
                truncated = ""
                current_w = 0
                for c in line:
                    c_w = _display_width(c)
                    if current_w + c_w > inner_width - 3:
                        break
                    truncated += c
                    current_w += c_w
                line = truncated + "..."
            
            # 使用显示宽度填充
            line_padded = _pad_to_width(line, inner_width)
            print(f"{Colors.CYAN}║{Colors.RESET}{line_padded}{Colors.CYAN}║{Colors.RESET}")
    
    print(f"{Colors.CYAN}╚{'═' * (width - 2)}╝{Colors.RESET}")


def print_table(headers: List[str], rows: List[List[Any]], title: str = None):
    """
    打印表格
    
    ╔════════════╦══════════╦═════════╗
    ║ Class      ║ IoU      ║ F1      ║
    ╠════════════╬══════════╬═════════╣
    ║ 地面       ║ 0.8500   ║ 0.9200  ║
    ║ 植被       ║ 0.7200   ║ 0.8400  ║
    ╚════════════╩══════════╩═════════╝
    """
    if not headers or not rows:
        return
    
    # 计算每列宽度
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(str(header))
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)
    
    # 生成分隔线
    def make_separator(left, mid, right, fill='═'):
        parts = [fill * w for w in col_widths]
        return left + mid.join(parts) + right
    
    # 打印标题
    if title:
        total_width = sum(col_widths) + len(col_widths) + 1
        print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    
    # 打印表头
    print(f"{Colors.CYAN}{make_separator('╔', '╦', '╗')}{Colors.RESET}")
    header_line = "║".join(f" {h:^{col_widths[i]-2}} " for i, h in enumerate(headers))
    print(f"{Colors.CYAN}║{Colors.BOLD}{header_line}{Colors.RESET}{Colors.CYAN}║{Colors.RESET}")
    print(f"{Colors.CYAN}{make_separator('╠', '╬', '╣')}{Colors.RESET}")
    
    # 打印数据行
    for row in rows:
        row_line = "║".join(f" {str(row[i]) if i < len(row) else '':^{col_widths[i]-2}} " for i in range(len(headers)))
        print(f"{Colors.CYAN}║{Colors.RESET}{row_line}{Colors.CYAN}║{Colors.RESET}")
    
    print(f"{Colors.CYAN}{make_separator('╚', '╩', '╝')}{Colors.RESET}")


def print_progress_line(stage: str, epoch: str, batch_info: str, time_info: str, 
                        metrics: Dict[str, Any] = None, extra: str = ""):
    """
    打印进度行 (训练/验证/测试)
    
    [Train] [1/100] [50/2803] 0:00:14<0:12:59, 0.28s/it, lr=1.00e-03, loss=1.55
    """
    # 阶段颜色
    stage_colors = {
        'Train': Colors.GREEN,
        'Val': Colors.BLUE,
        'Test': Colors.CYAN,
        'Pred': Colors.MAGENTA
    }
    stage_color = stage_colors.get(stage, Colors.WHITE)
    
    # 构建基础信息
    parts = [
        f"{stage_color}[{stage}]{Colors.RESET}",
        f"{epoch}" if epoch else "",
        f"[{batch_info}]",
        f"{time_info}"
    ]
    
    # 添加指标
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
    
    # 添加额外信息
    if extra:
        parts.append(extra)
    
    print(" ".join(filter(None, parts)))


def print_metrics_summary(title: str, metrics: Dict[str, Any], best_metric: str = None, 
                          best_value: float = None, best_epoch: int = None):
    """
    打印指标摘要
    
    ════════════════════════════════════════════════════════════════════════════════════════
    Validation Epoch 1 - Metrics
    ════════════════════════════════════════════════════════════════════════════════════════
    Overall Accuracy: 0.0980 (9.80%)
    Mean IoU (current): 0.0275
    Mean IoU (best)   : 0.0275 (Epoch 1)
    [OK] New best mIoU achieved!
    """
    width = 100
    print(f"\n{Colors.BOLD}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * width}{Colors.RESET}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key.lower() or 'acc' in key.lower():
                print(f"{key}: {Colors.GREEN}{value:.4f}{Colors.RESET} ({value*100:.2f}%)")
            else:
                print(f"{key}: {Colors.GREEN}{value:.4f}{Colors.RESET}")
        else:
            print(f"{key}: {Colors.GREEN}{value}{Colors.RESET}")
    
    # 打印最佳记录
    if best_metric and best_value is not None:
        print(f"{best_metric} (best): {Colors.YELLOW}{best_value:.4f}{Colors.RESET} (Epoch {best_epoch})")
    
    print(f"{Colors.BOLD}{'═' * width}{Colors.RESET}")


def _get_value_color(value: Any) -> str:
    """根据值类型返回合适的颜色"""
    if value is None or value == 'N/A' or value == 'null':
        return Colors.DIM
    elif isinstance(value, bool):
        return Colors.GREEN if value else Colors.RED
    elif isinstance(value, (int, float)):
        return Colors.YELLOW
    elif isinstance(value, str) and ('/' in value or '\\' in value):
        return Colors.CYAN  # 路径
    else:
        return Colors.GREEN


# ============================================================================
# 格式化工具函数
# ============================================================================

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


def format_points(n: int) -> str:
    """格式化点数"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ============================================================================
# Dual Logger (stdout + file)
# ============================================================================

class DualLogger:
    """
    双重日志记录器: 同时写入终端和文件
    
    特性:
    - 自动处理 Unicode 编码问题 (Windows 终端)
    - 写入文件时移除 ANSI 颜色代码
    - 即时刷新确保日志实时性
    """
    
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "a", encoding="utf-8")
        self.filepath = filepath

    def write(self, message: str):
        # 写入文件 (移除 ANSI 颜色代码)
        clean_message = self._strip_ansi(message)
        self.log_file.write(clean_message)
        self.log_file.flush()
        
        # 写入终端 (保留颜色，处理编码错误)
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except UnicodeEncodeError:
            # Windows 终端可能不支持某些 Unicode 字符(如 emoji)
            safe_message = message.encode(
                self.terminal.encoding or 'utf-8', 
                errors='replace'
            ).decode(
                self.terminal.encoding or 'utf-8', 
                errors='replace'
            )
            self.terminal.write(safe_message)
            self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
    
    @staticmethod
    def _strip_ansi(text: str) -> str:
        """移除 ANSI 转义序列"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


def setup_logger(output_dir: str, log_prefix: str = "training_log") -> str:
    """
    设置日志系统
    
    将 stdout 和 stderr 重定向到双重日志记录器，
    同时输出到终端和时间戳命名的日志文件。
    
    Args:
        output_dir: 日志输出目录
        log_prefix: 日志文件名前缀 (默认: "training_log")
        
    Returns:
        日志文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_prefix}_{timestamp}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    
    logger = DualLogger(log_filepath)
    sys.stdout = logger
    sys.stderr = logger
    
    # 打印日志头部
    print()
    print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.INFO}[LOG] 日志记录已启动{Colors.RESET}")
    print(f"{Colors.DIM}   时间: {datetime.datetime.now()}{Colors.RESET}")
    print(f"{Colors.DIM}   文件: {log_filepath}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
    print()
    
    return log_filepath


# ============================================================================
# 训练相关输出函数
# ============================================================================

def print_training_start(max_epochs: int, num_batches: int, **kwargs):
    """打印训练开始信息"""
    print()
    print_box("开始训练", {
        "最大轮数": max_epochs,
        "训练批次数": num_batches,
        **kwargs
    })


def print_epoch_start(current_epoch: int, max_epochs: int):
    """打印 Epoch 开始信息"""
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Epoch {current_epoch}/{max_epochs}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 50}{Colors.RESET}")


def print_validation_start():
    """打印验证开始信息"""
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}  开始验证...{Colors.RESET}")


def print_test_start():
    """打印测试开始信息"""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}  开始测试...{Colors.RESET}")


def print_predict_start():
    """打印预测开始信息"""
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}  开始预测...{Colors.RESET}")


# ============================================================================
# 兼容性导出
# ============================================================================

__all__ = [
    # 颜色
    'Colors',
    # 日志级别
    'log_info', 'log_success', 'log_warning', 'log_error', 'log_debug', 'log_step',
    # 美化输出
    'print_header', 'print_section', 'print_config', 'print_kv', 
    'print_box', 'print_table', 'print_progress_line', 'print_metrics_summary',
    # 格式化
    'format_size', 'format_time', 'format_number', 'format_points',
    # 双重日志
    'DualLogger', 'setup_logger',
    # 训练输出
    'print_training_start', 'print_epoch_start', 
    'print_validation_start', 'print_test_start', 'print_predict_start',
]
