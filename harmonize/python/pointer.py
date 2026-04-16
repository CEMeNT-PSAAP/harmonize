from numba import njit, jit, objmode, literal_unroll, types
from numba.extending import intrinsic
import numba as nb

import cffi
ffi = cffi.FFI()

from .codegen import generate_uuid

# =============================================================================
# uintp/voidptr casters
# =============================================================================


@intrinsic
def cast_any_to_voidptr(typingctx, src):
    # create the expected type signature
    result_type = types.voidptr
    sig = result_type(src)

    # defines the custom code generation
    def codegen(context, builder, signature, args):
        # llvm IRBuilder code here
        [src] = args
        rtype = signature.return_type
        llrtype = context.get_value_type(rtype)
        return builder.bitcast(src, llrtype)

    return sig, codegen


@intrinsic
def cast_uintp_to_voidptr(typingctx, src):
    # check for accepted types
    if isinstance(src, types.Integer):
        # create the expected type signature
        result_type = types.voidptr
        sig = result_type(types.uintp)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)

        return sig, codegen


@intrinsic
def cast_voidptr_to_uintp(typingctx, src):
    # check for accepted types
    if isinstance(src, types.RawPointer):
        # create the expected type signature
        result_type = types.uintp
        sig = result_type(types.voidptr)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.ptrtoint(src, llrtype)

        return sig, codegen

@njit()
def uintp_to_voidptr(value):
    val = nb.uintp(value)
    return cast_uintp_to_voidptr(val)

@njit()
def voidptr_to_uintp(value):
    return cast_voidptr_to_uintp(value)


@njit()
def into_voidptr(value):
    return into_voidptr_python(value)

def into_voidptr_python(value):
    raise RuntimeError("`into_voidptr` is only supported in nopython mode.")

@nb.extending.overload(into_voidptr_python)
def into_voidptr_overload(value):

    if isinstance(value,nb.types.Array) :
        def impl(value):
            ptr = (ffi.from_buffer(value))
            vptr = cast_any_to_voidptr(ptr)
            return vptr
        return impl
    elif isinstance(value,nb.types.CPointer) :
        def impl(value):
            return cast_any_to_voidptr(value)
        return impl
    elif isinstance(value,nb.types.Integer) :
        def impl(value):
            return cast_uintp_to_voidptr(value)
        return impl
    else :
        raise RuntimeError(f"`into_voidptr` is not supported for type '{value}'")




