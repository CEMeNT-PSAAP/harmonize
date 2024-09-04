import numba
import llvmlite

from harmonize.python import errors, config


from .prim import *



atomic_op_info = [
    ("add", [
        (int32,  int32),
        (uint32, uint32),
        (int64,  uint64),
        (uint64, uint64),
        (float32,float32),
        (float64,float64),
    ]),
    ("max", [
        (int32,  int32),
        (uint32, uint32),
        (uint64, uint64),
    ])
]


@numba.extending.intrinsic
def cast_ptr(typing_context, input_ptr, ptr_type):

    def impl(context, builder, _signature, args):
        llvm_type = context.get_value_type(target_type)
        val = builder.bitcast(args[0], llvm_type)
        return val

    if isinstance(ptr_type, numba.types.TypeRef):
        target_type = ptr_type.instance_type
    else:
        target_type = ptr_type

    sig = target_type(input_ptr, ptr_type)

    return sig, impl


@numba.extending.intrinsic
def get_ptr(typing_context, array):

    def impl(context, builder, _signature, args):
        llvm_array_type = context.get_value_type(array)
        llvm_vp_type = context.get_value_type(vp)
        four    = llvmlite.ir.Constant(llvmlite.ir.IntType(32),0)
        ptr     = builder.extract_value(args[0],4)
        gen_ptr = builder.bitcast(ptr,llvm_vp_type)
        return gen_ptr

    if not isinstance(array, numba.types.Array):
        return None

    sig = vp(array)

    return sig, impl


def array_atomic_extern(op_name,face_type):
    sig     = numba.core.typing.signature
    ext_fn  = numba.types.ExternalFunction

    signature = sig(face_type,vp,usize,face_type)
    py_name   = prim_info[face_type]["py_name"]
    implementation = ext_fn(f"array_atomic_{op_name}_{py_name}",  signature)

    @numba.njit
    def inner_cast(array,index,value):
        return implementation(array.ctypes,index,value)

    return inner_cast



def array_atomic_add(array,index,value):
    return None

@numba.extending.overload(array_atomic_add)
def array_atomic_add_inner(array,index,value):

    if config.CUDA_AVAILABLE:
        def impl(array,index,value):
            numba.cuda.atomic.add(array,index,value)
        return impl
    elif not config.ROCM_AVAILABLE:
        raise errors.no_platforms()

    if not isinstance(array,numba.types.Array):
        raise numba.errors.TypingError("First argument should be an array.")

    is_tuple = False

    if isinstance(index,numba.types.Tuple):
        if len(index) == 0:
            raise numba.errors.TypingError("Zero-length tuple given as index. At least one index must be provided.")
        for kind in index:
            if not isinstance(kind,numba.types.Integer):
                raise numba.errors.TypingError(f"Non-integer value of type '{kind}' in index tuple.")
        is_tuple = True
    elif isinstance(index,numba.types.UniTuple):
        if len(index) == 0:
            raise numba.errors.TypingError("Zero-length tuple given as index. At least one index must be provided.")
        if not isinstance(index.dtype,numba.types.Integer):
            raise numba.errors.TypingError(f"Non-integer value of type '{kind.dtype}' in index tuple.")
        is_tuple = True
    elif not isinstance(index,numba.types.Integer):
        raise numba.errors.TypingError("Index is not an integer or a positive-length tuple of integers.")

    allowed = {
        int32   : array_atomic_add_i32,
        uint32  : array_atomic_add_u32,
        int64   : array_atomic_add_i64,
        uint64  : array_atomic_add_u64,
        float32 : array_atomic_add_f32,
        float64 : array_atomic_add_f64
    }
    if not (value in allowed):
        return None

    inner = allowed[value]

    if is_tuple:

        def array_atomic_add(array,index,value):

            return inner(array[index[:-1]],index[-1],value)

        return array_atomic_add

    else:

        def array_atomic_add(array,index,value):
            return inner(array,index,value)

        return array_atomic_add



def array_atomic_max(array,index,value):
    return None

@numba.extending.overload(array_atomic_max)
def array_atomic_max_inner(array,index,value):

    if config.CUDA_AVAILABLE:
        def impl(array,index,value):
            numba.cuda.atomic.max(array,index,value)
        return impl
    elif not config.ROCM_AVAILABLE:
        raise errors.no_platforms()

    if not isinstance(array,numba.types.Array):
        raise numba.errors.TypingError("First argument should be an array.")

    if isinstance(index,numba.types.Tuple):
        if len(index) == 0:
            raise numba.errors.TypingError("Zero-length tuple given as index. At least one index must be provided.")
        for kind in index:
            if not isinstance(kind,numba.types.Integer):
                raise numba.errors.TypingError(f"Non-integer value of type '{kind}' in index tuple.")
        is_tuple = True
    elif isinstance(index,numba.types.UniTuple):
        if len(index) == 0:
            raise numba.errors.TypingError("Zero-length tuple given as index. At least one index must be provided.")
        if not isinstance(index.dtype,numba.types.Integer):
            raise numba.errors.TypingError(f"Non-integer value of type '{kind.dtype}' in index tuple.")
        is_tuple = True
    elif not isinstance(index,numba.types.Integer):
        raise numba.errors.TypingError("Index is not an integer or a positive-length tuple of integers.")

    allowed = {
        int32   : array_atomic_add_i32,
        uint32  : array_atomic_add_u32,
        uint64  : array_atomic_add_u64,
    }
    if not (value in allowed):
        return None

    inner = allowed[value]

    def array_atomic_max(array,index,value):
        return inner(array,index,value)

    return array_atomic_max





array_atomic_add_i32 = array_atomic_extern("add",  int32)
array_atomic_add_u32 = array_atomic_extern("add", uint32)
array_atomic_add_i64 = array_atomic_extern("add",  int64)
array_atomic_add_u64 = array_atomic_extern("add", uint64)
array_atomic_add_f32 = array_atomic_extern("add",float32)
array_atomic_add_f64 = array_atomic_extern("add",float64)

array_atomic_max_i32 = array_atomic_extern("max",  int32)
array_atomic_max_u32 = array_atomic_extern("max", uint32)
array_atomic_max_u64 = array_atomic_extern("max", uint64)

