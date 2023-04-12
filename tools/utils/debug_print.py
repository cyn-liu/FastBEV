import inspect
import os

from colorama import Fore, Back, init, Style
init()

# class ShellColor:
#     fg_black: Fore.BLACK
#     fg_blue: Fore.BLUE
#     fg_cyan: Fore.CYAN
#     fg_green: Fore.GREEN
#     fg_magenta: Fore.MAGENTA
#     fg_red: Fore.RED
#     fg_white: Fore.WHITE
#     fg_yellow: Fore.YELLOW
#     bg_black: Back.BLACK
#     bg_blue: Back.BLUE
#     bg_cyan: Back.CYAN
#     bg_green: Back.GREEN
#     bg_magenta: Back.MAGENTA
#     bg_red: Back.RED
#     bg_white: Back.WHITE
#     bg_yellow: Back.YELLOW

def DEBUG_print(*args, level=1, master_only=True, bg_color=True, **kwargs):
    # level
    # 1: 仅打印需要查看中间结果（默认值）
    # 2: 打印所有 DEBUG_print
    if master_only and os.getenv('LOCAL_RANK') and int(os.getenv('LOCAL_RANK')) != 0:
        return

    if os.getenv('DEBUG', None) is not None:
        if os.environ['DEBUG']:
            debug_level = 1
        else:
            debug_level = 2
        frame = inspect.currentframe().f_back
        lineno = frame.f_lineno
        filename = inspect.getframeinfo(frame).filename
        filename = "/".join(filename.split("/")[-3:])
        funcname = inspect.getframeinfo(frame).function
        formatter = f"{Fore.CYAN}[DEBUG] {Fore.MAGENTA}{filename}: {Fore.BLUE}{funcname}:{Fore.BLUE} {Fore.YELLOW}line: <- {Fore.GREEN}{lineno}{Fore.YELLOW} ->---{Fore.BLACK}"
        if bg_color:
            formatter = f"{Back.WHITE}" + formatter + Style.RESET_ALL
        if level <= debug_level:
            print(formatter, *args, **kwargs)
