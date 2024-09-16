import numba

def no_platforms():
    raise numba.errors.UnsupportedError("No GPU platforms (CUDA/ROCM) ")


