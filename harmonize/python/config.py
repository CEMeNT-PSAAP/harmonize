import numba

from enum       import Enum
from os.path    import dirname, abspath
from harmonize.python import errors, logging

DEBUG   = False
VERBOSE = True
INTERNAL_DEBUG = True
COLOR_LOG = True

NONE_AVAILABLE = True
ROCM_AVAILABLE = None
CUDA_AVAILABLE = None

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
    logging.verbose_print(f"CUDA_PATH set to {CUDA_PATH}")

def set_rocm_path(path):
    global ROCM_PATH
    ROCM_PATH = path
    logging.verbose_print(f"ROCM_PATH set to {ROCM_PATH}")

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

