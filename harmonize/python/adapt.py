import llvmlite
from numba.extending import intrinsic
from .types import *

# =============================================================================
# Atomics
# =============================================================================

ATOMIC_FUNCTION_ROSTER = {
    "add" : [i32,u32,u64,f32,f64],
    "sub" : [i32,u32],
    "exch": [i32,u32,u64,f32],
    "min" : [i32,u32,u64,i64],
    "max" : [i32,u32,u64,i64],
    "inc" : [u32],
    "dec" : [u32],
    "CAS" : [i32,u32,u64,u16],
    "and" : [i32,u32,u64],
    "or"  : [i32,u32,u64],
    "xor" : [i32,u32,u64]
}

def build_rmw_atomic_intrinsic(operation):

    # Defines intrinisic over
    def atomicIntrinsic(typingctx,target,index,value):

        # Atomic operations must be performed upon something in
        # memory (pointed by the first argument)
        print("\n\n\n",type(target),"\n\n\n")

        if not isinstance(target, numba.types.Array):
            raise Exception("First argument is not an array type.")
            return None

        # The base type of the target pointer must match the second
        # argument
        is_lit_int = False
        if isinstance(value,numba.types.Literal) and numba.types.unliteral(value) == numba.types.Integer:
            is_lit_int = True
        if (value != target.dtype):
            raise Exception(f"Type of value argument ({value}) does not match the element type of the array argument ({target.dtype})")
            return None

        # The type must be supported by the function
        if not target.dtype in ATOMIC_FUNCTION_ROSTER[operation]:
            raise Exception(f"Atomic {operation} not supported for values of type {target.dtype}")
            return None

        # create the expected type signature
        result_type = target.dtype
        sig = result_type(target,usize,value)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [target,index,value] = args
            zero = llvmlite.ir.Constant(llvmlite.ir.IntType(usize.bitwidth),0)
            four = llvmlite.ir.Constant(llvmlite.ir.IntType(usize.bitwidth),5)
            print("\n\n\n",target,"\n\n\n")
            ptr = builder.gep(target, [four,four,zero], inbounds=False)
            return  builder.atomic_rmw(operation, ptr, value, 'seq_cst')

        return sig, codegen

    return atomicIntrinsic

@intrinsic
def atomic_add(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("add")(typingctx,target,index,value)

#@intrinsic
def atomic_sub(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("sub")(typingctx,target,index,value)

#@intrinsic
def atomic_exch(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("exch")(typingctx,target,index,value)

#@intrinsic
def atomic_min(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("min")(typingctx,target,index,value)

#@intrinsic
def atomic_max(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("max")(typingctx,target,index,value)

#@intrinsic
def atomic_inc(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("inc")(typingctx,target,index,value)

#@intrinsic
def atomic_dec(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("dec")(typingctx,target,index,value)

#@intrinsic
def atomic_and(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("and")(typingctx,target,index,value)

#@intrinsic
def atomic_or(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic( "or")(typingctx,target,index,value)

#@intrinsic
def atomic_xor(typingctx,target,index,value):
    return build_rmw_atomic_intrinsic("xor")(typingctx,target,index,value)

