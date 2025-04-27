import numba
from .templates import clock_template

def generate_clock_code():
    return clock_template

sig     = numba.core.typing.signature
ext_fn  = numba.types.ExternalFunction
get_wall_clock = ext_fn("get_wall_clock",sig(numba.types.int64))

