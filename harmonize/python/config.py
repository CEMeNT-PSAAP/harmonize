import numba
import subprocess


from os         import path
from enum       import Enum
from os.path    import dirname, abspath
from harmonize.python import errors, logging

class ShouldCompile(Enum):
    ALWAYS = 1
    NEVER  = 2


DEBUG   = False
VERBOSE = True
INTERNAL_DEBUG = False
ERROR_PRINT = True
COLOR_LOG = True

NONE_AVAILABLE = True
ROCM_AVAILABLE = None
CUDA_AVAILABLE = None
SHOULD_COMPILE = None


try:
    import numba.hip as hip
    ROCM_AVAILABLE = True
    NONE_AVAILABLE = False
    logging.verbose_print(f"ROCM_PLATFORM detected. Proceeding under ROCM.")
except ImportError:
    ROCM_AVAILABLE = False


# The HIP version of Numba "has" a cuda module,
# but doesn't support the things normal Numba does
if not ROCM_AVAILABLE:
    try :
        import numba.cuda as cuda
        CUDA_AVAILABLE = True
        NONE_AVAILABLE = False
        logging.verbose_print(f"CUDA_PLATFORM detected. Proceeding under CUDA.")
    except ImportError:
        CUDA_AVAILABLE = False


if NONE_AVAILABLE:
    raise errors.no_platforms()


def set_debug(debug):
    global DEBUG
    DEBUG = debug

def set_verbose(verbose):
    global VERBOSE
    VERBOSE = verbose

def set_internal_debug(internal_debug):
    global INTERNAL_DEBUG
    INTERNAL_DEBUG = internal_debug


class GPUPlatform(Enum):
    CUDA = 1
    ROCM = 2

CUDA_PATH  = None
ROCM_PATH  = None

HARMONIZE_ROOT_DIR    =  dirname(abspath(__file__))+"/.."
HARMONIZE_ROOT_HEADER = HARMONIZE_ROOT_DIR+"/cpp/harmonize.h"


def set_platform(platform):
    global GPU_PLATFORM
    GPU_PLATFORM = platform
    logging.verbose_print(f"GPU_PLATFORM set to {GPU_PLATFORM}")

def set_cuda_path(path):
    global CUDA_PATH
    CUDA_PATH = path
    if path == None:
        logging.verbose_print(f"CUDA_PATH cleared to default")
    else:
        logging.verbose_print(f"CUDA_PATH set to {CUDA_PATH}")

def set_rocm_path(path):
    global ROCM_PATH
    ROCM_PATH = path
    if path == None:
        logging.verbose_print(f"ROCM_PATH cleared to default")
    else:
        logging.verbose_print(f"ROCM_PATH set to {ROCM_PATH}")


def should_compile(mode):
    global SHOULD_COMPILE
    SHOULD_COMPILE = mode

def compilation_gate(auto):
    if   SHOULD_COMPILE == ShouldCompile.ALWAYS:
        return True
    elif SHOULD_COMPILE == ShouldCompile.NEVER:
        return False
    else:
        return auto


def check_rocm_path(quiet=False):

    dep_paths = [
        hipcc_path(),
        hipcc_clang_offload_bundler_path(),
        hipcc_llvm_link_path(),
        hipcc_llvm_as_path(),
    ]

    fail = False
    for dep in dep_paths:
        if not path.isfile(dep):
            if not quiet:
                logging.error_print(f"Path '{dep}' does not correspond to a file.")
            fail = True
    if fail and (not quiet):
        raise RuntimeError("Some required ROCM dependencies missing.")

    return not fail


def try_autoset_rocm_path():
    hipcc_path_cmd = "which hipcc"
    result = subprocess.run(hipcc_path_cmd.split(),shell=False,check=True,stdout=subprocess.PIPE)
    path = result.stdout.decode('utf-8')[:-1]

    tail = "/bin/hipcc"
    if not path.endswith(tail):
        return
    logging.verbose_print(f"Found hipcc located at '{path}'")

    path = path[:-len(tail)]
    set_rocm_path(path)
    if not check_rocm_path(quiet=True):
        set_rocm_path(None)

def nvcc_path():
    if CUDA_PATH == None:
        return "nvcc"
    else:
        return CUDA_PATH + "/bin/nvcc"

def hipcc_path():
    if ROCM_PATH == None:
        return "hipcc"
    else:
        return ROCM_PATH + "/bin/hipcc"

def hipcc_clang_offload_bundler_path():
    if ROCM_PATH == None:
        return "clang-offload-bundler"
    else:
        return ROCM_PATH + "/llvm/bin/clang-offload-bundler"

def hipcc_llvm_link_path():
    if ROCM_PATH == None:
        return "llvm-link"
    else:
        return ROCM_PATH + "/llvm/bin/llvm-link"

def hipcc_llvm_as_path():
    if ROCM_PATH == None:
        return "llvm-as"
    else:
        return ROCM_PATH + "/llvm/bin/llvm-as"


# Uses nvidia-smi to query the compute level of the GPUs on the system. This
# compute level is what is used for compling PTX.
def native_gpu_arch(platform):
    if platform == GPUPlatform.ROCM:
        return ""
    else:
        capability = cuda.get_current_device().compute_capability
        output = f"{capability[0]}{capability[1]}"
        return output

if ROCM_AVAILABLE:
    try_autoset_rocm_path()
