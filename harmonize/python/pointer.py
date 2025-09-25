from numba import njit, jit, objmode, literal_unroll, types
from numba.extending import intrinsic
import numba


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
    val = numba.uintp(value)
    return cast_uintp_to_voidptr(val)

@njit()
def voidptr_to_uintp(value):
    return cast_voidptr_to_uintp(value)

@njit()
def any_to_voidptr(value):
    return cast_any_to_voidptr(value)

