
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
