
from harmonize.python import config

PINK = '\033[95m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
NORMAL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def verbose_print(*args):
    if config.VERBOSE:
        if config.COLOR_LOG:
            print(CYAN,"[ VERBOSE ]",NORMAL,*args,flush=True)
        else:
            print("[ VERBOSE ]",*args,flush=True)

def debug_print(*args):
    if config.INTERNAL_DEBUG:
        if config.COLOR_LOG:
            print(YELLOW,"[  DEBUG  ]",NORMAL,*args,flush=True)
        else:
            print("[  DEBUG  ]",*args,flush=True)

def error_print(*args):
    if config.ERROR_PRINT:
        if config.COLOR_LOG:
            print(RED,"[  ERROR  ]",NORMAL,*args,flush=True)
        else:
            print("[  ERROR  ]",*args,flush=True)


biggest_progress_print = 0

def progress_print(message):
    if config.VERBOSE or config.INTERNAL_DEBUG:
        verbose_print(message)
    else:
        global biggest_progress_print
        length = len(message)
        if length < biggest_progress_print:
            message.ljust(biggest_progress_print)
        elif length > biggest_progress_print:
            biggest_progress_print = length

        print(f"{message:<{biggest_progress_print}}\r",end="",flush=True)

