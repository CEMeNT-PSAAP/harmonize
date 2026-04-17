import numba as nb
import numpy as np

from numba import types

from harmonize.python import config
from functools import reduce
import operator

import cffi
ffi = cffi.FFI()

from .pointer import *
from .templates import *
from .codegen import generate_uuid


def local_array(shape,dtype):
    return np.empty(shape,dtype=dtype)

@nb.extending.type_callable(local_array)
def type_local_array(context):

    from numba.core.typing.npydecl import parse_dtype, parse_shape


    if isinstance(context,nb.core.typing.context.Context):

        # Function repurposed from numba's ol_np_empty.
        def typer(shape, dtype):
            nb.np.arrayobj._check_const_str_dtype("empty", dtype)

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, nb.types.Integer):
                if not isinstance(shape, nb.types.IntegerLiteral):
                    raise nb.core.errors.UnsupportedError(f"Integer shape type {shape} is not literal.")
            elif isinstance(shape, (nb.types.Tuple, nb.types.UniTuple)):
                if any([not isinstance(s, nb.types.IntegerLiteral)
                        for s in shape]):
                    raise nb.core.errors.UnsupportedError(
                        f"At least one element of shape tuple type{shape} is not an integer literal."
                    )
            else:
                raise nb.core.errors.UnsupportedError(f"Shape is of unsupported type {shape}.")

            # No default arguments.
            nb_dtype = parse_dtype(dtype)
            nb_shape = parse_shape(shape)

            if nb_dtype is not None and nb_shape is not None:
                retty = nb.types.Array(dtype=nb_dtype, ndim=nb_shape, layout='C')
                # Inlining the signature construction from numpy_empty_nd
                sig = retty(shape, dtype)
                return sig
            else:
                msg = f"Cannot parse input types to function np.empty({shape}, {dtype})"
                raise nb.errors.TypingError(msg)
        return typer

    elif config.CUDA_AVAILABLE and isinstance(context,nb.cuda.target.CUDATypingContext):

        # Function repurposed from numba's Cuda_array_decl.
        def typer(shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, nb.types.Integer):
                if not isinstance(shape, nb.types.IntegerLiteral):
                    return None
            elif isinstance(shape, (nb.types.Tuple, nb.types.UniTuple)):
                if any([not isinstance(s, nb.types.IntegerLiteral)
                        for s in shape]):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return nb.types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        return typer

    elif config.ROCM_AVAILABLE and isinstance(context,nb.hip.target.HIPTypingContext):

        def typer(shape, dtype):
            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, nb.types.Integer):
                if not isinstance(shape, nb.types.IntegerLiteral):
                    return None
            elif isinstance(shape, (nb.types.Tuple, nb.types.UniTuple)):
                if any([not isinstance(s, nb.types.IntegerLiteral) for s in shape]):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                result = nb.types.Array(dtype=nb_dtype, ndim=ndim, layout="C")
                return result

        return typer

    else:
        raise nb.core.errors.UnsupportedError(f"Unsupported target context {context}.")




@nb.extending.lower_builtin(local_array, nb.types.IntegerLiteral, nb.types.Any)
def builtin_local_array(context, builder, sig, args):

    shape, dtype = sig.args

    from numba.core.typing.npydecl import parse_dtype, parse_shape
    import numba.np.arrayobj as arrayobj

    if isinstance(context,nb.core.cpu.CPUContext):

        # No default arguments.
        nb_dtype = parse_dtype(dtype)
        nb_shape = parse_shape(shape)

        retty = nb.types.Array(dtype=nb_dtype, ndim=nb_shape, layout='C')

        # In ol_np_empty, the reference type of the array is fed into the
        # signatrue as a third argument. This third argument is not used by
        # _parse_empty_args.
        sig = retty(shape, dtype)

        arrtype, shapes = arrayobj._parse_empty_args(context, builder, sig, args)
        ary = arrayobj._empty_nd_impl(context, builder, arrtype, shapes)

        return ary._getvalue()
    elif config.CUDA_AVAILABLE and isinstance(context,nb.cuda.target.CUDATargetContext):
        length = sig.args[0].literal_value
        dtype = parse_dtype(sig.args[1])
        return nb.cuda.cudaimpl._generic_array(
            context,
            builder,
            shape=(length,),
            dtype=dtype,
            symbol_name='_cudapy_harm_lmem',
            addrspace=nb.cuda.cudadrv.nvvm.ADDRSPACE_LOCAL,
            can_dynsized=False
        )
    elif config.ROCM_AVAILABLE and isinstance(context,nb.hip.target.HIPTargetContext):
        length = sig.args[0].literal_value
        dtype = parse_dtype(sig.args[1])
        result = nb.hip.typing_lowering.hip.lowering._generic_array(
            context,
            builder,
            shape=(length,),
            dtype=dtype,
            symbol_name="_HIPpy_lmem",
            addrspace=nb.hip.amdgcn.ADDRSPACE_LOCAL,
            can_dynsized=False,
        )
        return result
    else:
        raise nb.core.errors.UnsupportedError(f"Unsupported target context {context}.")



from numba.core.typing.npydecl import parse_dtype
import llvmlite.binding as ll
from llvmlite import ir
numba_dev_ptr = nb.types.voidptr



def array_from_ptr(ptr,shape,dtype):
    pass


@nb.extending.type_callable(array_from_ptr)
def type_array_from_ptr(context):

    from numba.core.typing.npydecl import parse_dtype, parse_shape

    if isinstance(context, nb.core.typing.context.Context):
        # Function repurposed from Numba's ol_np_empty.
        def typer(ptr,shape, dtype):
            nb.np.arrayobj._check_const_str_dtype("empty", dtype)

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    raise nb.core.errors.UnsupportedError(
                        f"Integer shape type {shape} is not literal."
                    )
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral) for s in shape]):
                    raise nb.core.errors.UnsupportedError(
                        f"At least one element of shape tuple type{shape} is not an integer literal."
                    )
            else:
                raise nb.core.errors.UnsupportedError(
                    f"Shape is of unsupported type {shape}."
                )

            # No default arguments.
            nb_dtype = parse_dtype(dtype)
            nb_shape = parse_shape(shape)

            if nb_dtype is not None and nb_shape is not None:
                retty = types.Array(dtype=nb_dtype, ndim=nb_shape, layout="C")
                # Inlining the signature construction from numpy_empty_nd
                sig = retty(shape, dtype)
                return sig
            else:
                msg = f"Cannot parse input types to function np.empty({shape}, {dtype})"
                raise numba.errors.TypingError(msg)

        return typer

    elif isinstance(context, nb.cuda.target.CUDATypingContext):

        # Function repurposed from Numba's Cuda_array_decl.
        def typer(ptr, shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, nb.types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    msg = "At least one dimension in the input shape is non-literal"
                    raise nb.errors.TypingError(msg)
            elif isinstance(shape, (nb.types.Tuple, nb.types.UniTuple)):
                if any([not isinstance(s, nb.types.IntegerLiteral) for s in shape]):
                    msg = "At least one dimension in the input shape is non-literal"
                    raise nb.errors.TypingError(msg)
            else:
                msg = "Unexpected shape type. Shapes must be literal integers or tuples of literal integers."
                raise nb.errors.TypingError(msg)

            if not isinstance(ptr,nb.types.RawPointer):
                msg = f"Expected raw pointer as third argument, recieved {ptr}"
                raise nb.errors.TypingError(msg)

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return nb.types.Array(dtype=nb_dtype, ndim=ndim, layout="C")

        return typer

    elif isinstance(context, nb.hip.target.HIPTypingContext):
        def typer(ptr, shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, nb.types.Integer):
                if not isinstance(shape, nb.types.IntegerLiteral):
                    msg = "At least one dimension in the input shape is non-literal"
                    raise nb.errors.TypingError(msg)
            elif isinstance(shape, (nb.types.Tuple, nb.types.UniTuple)):
                if any([not isinstance(s, nb.types.IntegerLiteral) for s in shape]):
                    msg = "At least one dimension in the input shape is non-literal"
                    raise nb.errors.TypingError(msg)
            else:
                msg = "Unexpected shape type. Shapes must be literal integers or tuples of literal integers."
                raise nb.errors.TypingError(msg)

            if not isinstance(ptr,nb.types.RawPointer):
                msg = f"Expected raw pointer as third argument, recieved {ptr}"
                raise nb.errors.TypingError(msg)

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return nb.types.Array(dtype=nb_dtype, ndim=ndim, layout="C")

        return typer

    else:
        raise nb.core.errors.UnsupportedError(
            f"Unsupported target context {context}."
        )




@nb.extending.lower_builtin(array_from_ptr, numba_dev_ptr, nb.types.Tuple, nb.types.Any)
@nb.extending.lower_builtin(array_from_ptr, numba_dev_ptr, nb.types.UniTuple, nb.types.Any)
@nb.extending.lower_builtin(array_from_ptr, numba_dev_ptr, nb.types.IntegerLiteral, nb.types.Any)
def def_array_from_ptr(context, builder, sig, args):

    if isinstance(context,nb.core.cpu.CPUContext):
        raw_ptr_type = sig.args[0]
        raw_ptr      = args[0]
        if isinstance(sig.args[1],nb.types.IntegerLiteral):
            shape = [sig.args[1].literal_value]
        else:
            shape = [s.literal_value for s in sig.args[1]]
        dtype = parse_dtype(sig.args[2])

        elemcount = reduce(operator.mul, shape, 1)
        lmod = builder.module

        lldtype = context.get_data_type(dtype)
        itemsize = context.get_abi_sizeof(lldtype)

        laststride = itemsize
        rstrides = []
        for i, lastsize in enumerate(reversed(shape)):
            rstrides.append(laststride)
            laststride *= lastsize
        strides = [s for s in reversed(rstrides)]
        kstrides = [context.get_constant(types.intp, s) for s in strides]

        ndim = len(shape)
        aryty = types.Array(dtype=dtype, ndim=ndim, layout="C")
        ary = context.make_array(aryty)(context, builder)

        dataptr = builder.addrspacecast(
            raw_ptr, ir.PointerType(ir.IntType(8)), "generic"
        )

        kshape = [context.get_constant(types.intp, s) for s in shape]
        context.populate_array(
            ary,
            data=builder.bitcast(dataptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=context.get_constant(types.intp, itemsize),
            meminfo=None,
        )
        return ary._getvalue()
    elif isinstance(context, nb.cuda.target.CUDATargetContext):
        raw_ptr_type = sig.args[0]
        raw_ptr      = args[0]
        if isinstance(sig.args[1],nb.types.IntegerLiteral):
            shape = [sig.args[1].literal_value]
        else:
            shape = [s.literal_value for s in sig.args[1]]
        dtype = parse_dtype(sig.args[2])

        elemcount = reduce(operator.mul, shape, 1)
        lmod = builder.module

        targetdata = ll.create_target_data(nb.cuda.cudadrv.nvvm.NVVM().data_layout)
        lldtype = context.get_data_type(dtype)
        itemsize = lldtype.get_abi_size(targetdata)

        laststride = itemsize
        rstrides = []
        for i, lastsize in enumerate(reversed(shape)):
            rstrides.append(laststride)
            laststride *= lastsize
        strides = [s for s in reversed(rstrides)]
        kstrides = [context.get_constant(types.intp, s) for s in strides]

        ndim = len(shape)
        aryty = types.Array(dtype=dtype, ndim=ndim, layout="C")
        ary = context.make_array(aryty)(context, builder)

        dataptr = builder.addrspacecast(
            raw_ptr, ir.PointerType(ir.IntType(8)), "generic"
        )

        kshape = [context.get_constant(types.intp, s) for s in shape]
        context.populate_array(
            ary,
            data=builder.bitcast(dataptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=context.get_constant(types.intp, itemsize),
            meminfo=None,
        )
        return ary._getvalue()
    elif isinstance(context, nb.hip.target.HIPTargetContext):
        raw_ptr_type = sig.args[0]
        raw_ptr      = args[0]


        if isinstance(sig.args[1],nb.types.IntegerLiteral):
            shape = [sig.args[1].literal_value]
        else:
            shape = [s.literal_value for s in sig.args[1]]

        dtype = parse_dtype(sig.args[2])

        elemcount = reduce(operator.mul, shape, 1)
        lmod = builder.module

        targetdata = ll.create_target_data(
            nb.hip.amdgcn.DATA_LAYOUT
        )
        lldtype = context.get_data_type(dtype)
        itemsize = lldtype.get_abi_size(targetdata)

        laststride = itemsize
        rstrides = []
        for i, lastsize in enumerate(reversed(shape)):
            rstrides.append(laststride)
            laststride *= lastsize
        strides = [s for s in reversed(rstrides)]
        kstrides = [context.get_constant(types.intp, s) for s in strides]

        ndim = len(shape)
        aryty = types.Array(dtype=dtype, ndim=ndim, layout="C")
        ary = context.make_array(aryty)(context, builder)
        

        dataptr = builder.addrspacecast(
            raw_ptr, ir.PointerType(ir.IntType(8)), "generic"
        )

        kshape = [context.get_constant(types.intp, s) for s in shape]


        context.populate_array(
            ary,
            data=builder.bitcast(dataptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=context.get_constant(types.intp, itemsize),
            meminfo=None,
        )
        result = ary._getvalue()
        return result
    else:
        raise nb.core.errors.UnsupportedError(
            f"Unsupported target context {context}."
        )



@nb.njit()
def python_array_from_ptr(ptr):
    return array_from_ptr(ptr)




sig     = nb.core.typing.signature
ext_fn  = nb.types.ExternalFunction


ext_alloc_device_bytes    = ext_fn("harmonize_alloc_device_bytes",    sig(nb.types.voidptr,nb.types.intp))
ext_alloc_managed_bytes   = ext_fn("harmonize_alloc_managed_bytes",   sig(nb.types.voidptr,nb.types.intp))
ext_free_device_bytes     = ext_fn("harmonize_free_device_bytes",     sig(nb.types.void,nb.types.voidptr))
#ext_memcpy_host_to_device = ext_fn("harmonize_memcpy_host_to_device", sig(nb.types.void,nb.types.voidptr,nb.types.void_ptr))
#ext_memcpy_device_to_host = ext_fn("harmonize_memcpy_device_to_host", sig(nb.types.void,nb.types.voidptr,nb.types.void_ptr))

@nb.jit()
def alloc_device_bytes(size):
    return ext_alloc_device_bytes(size)

@nb.jit()
def alloc_managed_bytes(size):
    return ext_alloc_managed_bytes(size)

@nb.jit()
def free_device_bytes(size):
    return ext_free_device_bytes(size)

@nb.jit()
def ptr_from_array(array):
    return nb.carray(array)





host_to_device_array_template = """
def {name}(dev_ptr,host_obj):
    ptr = (ffi.from_buffer(host_obj))
    vptr = into_voidptr(ptr)
    host_to_device(dev_ptr,vptr,size*host_obj.size)
"""

host_to_device_object_template = """
def {name}(dev_ptr,host_obj):
    host_to_device(dev_ptr,host_obj,size)
"""

@nb.njit()
def memcpy_host_to_device(dev_ptr,host_obj):
    memcpy_host_to_device_python(dev_ptr,host_obj)

def memcpy_host_to_device_python(dev_ptr,host_obj):
    raise RuntimeError("Host/device memory copies are only supported in nopython mode, and only on CPU.")

@nb.extending.overload(memcpy_host_to_device_python,target='cpu')
def cpu_memcpy_host_to_device_overload(dev_ptr,host_obj):

    void    = nb.types.void
    vp      = nb.types.voidptr
    uintp   = nb.types.uintp
    sig     = nb.core.typing.signature

    if isinstance(host_obj,nb.types.Array) :
        id = generate_uuid()
        name = f"memcpy_host_to_device_{id}"
        if isinstance(host_obj.dtype,nb.types.Record):
            size = host_obj.dtype.size
        else :
            size = host_obj.dtype.bitwidth / 8
        host_to_device   = ext_fn(f"harmonize_memcpy_host_to_device",   sig(void, vp, vp, uintp))
        exec(host_to_device_array_template.format(name=name),globals()|locals(),locals())
        impl = eval(name)
    else :
        id = generate_uuid()
        name = f"memcpy_host_to_device_{id}"
        size = host_obj.size
        host_to_device   = ext_fn(f"harmonize_memcpy_host_to_device",   sig(void, vp, host_obj, uintp))
        exec(host_to_device_object_template.format(name=name),globals()|locals(),locals())
        impl = eval(name)
    return impl

@nb.extending.overload(memcpy_host_to_device_python,target='gpu')
def gpu_memcpy_host_to_device_overload(dev_ptr,host_obj):
    raise RuntimeError("Host/device memory copies are only supported in nopython mode, and only on CPU.")







device_to_host_array_template = """
def {name}(host_obj,dev_ptr):
    ptr = (ffi.from_buffer(host_obj))
    vptr = into_voidptr(ptr)
    device_to_host(vptr,dev_ptr,size*host_obj.size)
"""

device_to_host_object_template = """
def {name}(host_obj,dev_ptr):
    device_to_host(host_obj,dev_ptr,size)
"""



@nb.njit()
def memcpy_device_to_host(host_obj,dev_ptr):
    memcpy_device_to_host_python(host_obj,dev_ptr)

def memcpy_device_to_host_python(host_obj,dev_ptr):
    raise RuntimeError("Host/device memory copies are only supported in nopython mode, and only on CPU.")

@nb.extending.overload(memcpy_device_to_host_python,target='cpu')
def cpu_memcpy_device_to_host_overload(host_obj,dev_ptr):

    void    = nb.types.void
    vp      = nb.types.voidptr
    uintp   = nb.types.uintp
    sig     = nb.core.typing.signature

    if isinstance(host_obj,nb.types.Array) :
        id = generate_uuid()
        name = f"memcpy_device_to_host_{id}"
        if isinstance(host_obj.dtype,nb.types.Record):
            size = host_obj.dtype.size
        else :
            size = host_obj.dtype.bitwidth / 8
        device_to_host   = ext_fn(f"harmonize_memcpy_device_to_host",   sig(void, vp, vp, uintp))
        exec(device_to_host_array_template.format(name=name),globals()|locals(),locals())
        impl = eval(name)
    else :
        id = generate_uuid()
        name = f"memcpy_device_to_host_{id}"
        device_to_host   = ext_fn(f"harmonize_memcpy_device_to_host",   sig(void, host_obj, vp, uintp))
        size = host_obj.size
        exec(device_to_host_object_template.format(name=name),globals()|locals(),locals())
        impl = eval(name)
    return impl

@nb.extending.overload(memcpy_device_to_host_python,target='gpu')
def gpu_memcpy_device_to_host_overload(host_obj,dev_ptr):
    raise RuntimeError("Host/device memory copies are only supported in nopython mode, and only on CPU.")



def generate_array_code():
    text = ""
    text += alloc_device_bytes_template
    text += alloc_managed_bytes_template
    text += free_device_bytes_template
    text += memcpy_host_to_device_template
    text += memcpy_device_to_host_template
    return text

