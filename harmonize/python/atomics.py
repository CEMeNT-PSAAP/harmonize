import numba
import llvmlite
from .prim import *



@numba.extending.intrinsic
def cast_ptr(_typingctx, input_ptr, ptr_type):

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
def get_ptr(_typingctx, array):

    def impl(context, builder, _signature, args):
        #llvm_type = context.get_value_type(vp.instance_type)
        #val = builder.bitcast(args[0], llvm_type)
        #return val
        llvm_array_type = context.get_value_type(array)
        llvm_vp_type = context.get_value_type(vp)
        loc_arg = builder.alloca(llvm_array_type)
        stor    = builder.store(args[0],loc_arg)
        zero    = llvmlite.ir.Constant(llvmlite.ir.IntType(32),0)
        four    = llvmlite.ir.Constant(llvmlite.ir.IntType(32),4)
        ptr_ptr = builder.gep(loc_arg,[zero,four])
        ptr     = builder.load(ptr_ptr)
        gen_ptr = builder.bitcast(ptr,llvm_vp_type)
        return gen_ptr

    if not isinstance(array, numba.types.Array):
        return None

    sig = vp(array)

    return sig, impl


def array_atomic_extern(name,kind):
    sig     = numba.core.typing.signature
    ext_fn  = numba.types.ExternalFunction

    implementation = ext_fn(f"array_atomic_{name}_{prim_name[kind]}",  sig(kind, vp, usize, kind))

    @numba.njit
    def inner_cast(array,index,value):
        return implementation(get_ptr(array),index,value)

    return inner_cast


array_atomic_add_i32 = array_atomic_extern("add",  int32)
array_atomic_add_u32 = array_atomic_extern("add", uint32)
array_atomic_add_u64 = array_atomic_extern("add", uint64)
array_atomic_add_i64 = array_atomic_extern("add",  int64)
array_atomic_add_f32 = array_atomic_extern("add",float32)
array_atomic_add_f64 = array_atomic_extern("add",float64)




