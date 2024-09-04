import numba

from harmonize.python import errors

DEBUG   = False
VERBOSE = False

NONE_AVAILABLE = True
ROCM_AVAILABLE = None
CUDA_AVAILABLE = None

try:
    import numba.hip as hip
    ROCM_AVAILABLE = True
    NONE_AVAILABLE = False
except ImportError:
    ROCM_AVAILABLE = False


# The HIP version of Numba "has" a cuda module,
# but doesn't support the things normal Numba does
if not ROCM_AVAILABLE:
    try :
        import numba.cuda as cuda
        CUDA_AVAILABLE = True
        NONE_AVAILABLE = False
    except ImportError:
        CUDA_AVAILABLE = False


if NONE_AVAILABLE:
    raise errors.no_platforms()

