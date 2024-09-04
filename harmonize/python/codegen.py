import re
import numba
import numpy as np
import struct

from harmonize.python import config
from harmonize.python.logging import verbose_print, debug_print

# Injects `value` as the value of the global variable named `name` in the module
# that defined the function `index` calls down the stack, from the perspective
# of the function that calls `inject_global`.
def inject_global(name,value,index):
    frm = inspect.stack()[index+1]
    mod = inspect.getmodule(frm[0])
    setattr(sys.modules[mod.__name__], name, value)
    debug_print(f"Defined '{name}' as "+str(value)+" for module "+mod.__name__)


# Returns the type annotations of the arguments for the input function
def fn_arg_ano_list( func ):
    result = []
    for arg,ano in func.__annotations__.items():
        if( arg != 'return' ):
            result.append(ano)
    return result

# Returns the numba type of the function signature of the input function
def fn_sig( func ):
    arg_list = []
    ret    = numba.types.void
    for arg,ano in func.__annotations__.items():
        if( arg != 'return' ):
            arg_list.append(ano)
        else:
            ret = ano
    return ret(*arg_list)


# Returns the type annotations of the arguments for the input function
# as a tuple
def fn_arg_ano( func ):
    return tuple( x for x in fn_arg_ano_list(func) )


# Raises an error if the annotated return type for the input function
# does not patch the input result type
def assert_fn_res_ano( func, res_type ):
    if 'return' in func.__annotations__:
        res_ano = func.__annotations__['return']
        if res_ano != res_type :
            arg_str  = str(fn_arg_ano(func))
            ano_str  = arg_str + " -> " + str(res_ano)
            cmp_str  = arg_str + " -> " + str(res_type)
            err_str  = "Annotated function type '" + ano_str               \
            + "' does not match the type deduced by the compiler '"        \
            + cmp_str + "'\nMake sure the definition of the function '"    \
            + func.__name__ + "' results in a return type  matching its "  \
            + "annotation when supplied arguments matching its annotation."
            raise(TypingError(err_str))


# Returns the ptx of the input function, as a global CUDA function. If the return type deduced
# by compilation is not consistent with the annotated return type, an exception is raised.
def global_ir( func ):
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),debug=config.DEBUG,opt=(not config.DEBUG))
    assert_fn_res_ano(func, res_type)
    return ptx

# Returns the ptx of the input function, as a device CUDA function. If the return type deduced
# by compilation is not consistent with the annotated return type, an exception is raised.
def device_ir( func, platform ):
    if platform == config.GPUPlatform.ROCM:
        ptx, res_type = config.hip.compile_ptx_for_current_device(func,fn_arg_ano(func),device=True,debug=config.DEBUG,opt=(not config.DEBUG),name=f"_{func.__name__}",link_in_hipdevicelib=False)
    else :
        ptx, res_type = config.cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),device=True,debug=config.DEBUG,opt=(not config.DEBUG))
    assert_fn_res_ano(func, res_type)
    return ptx, res_type

# Returns the modify date of the file containing the supplied function. This is useful for
# detecting if a function was possibly changed
def func_defn_time(func):
    return getmtime(func.__globals__['__file__'])



def extern_device_ir( func, type_map, suffix, platform ):
    arg_types   = fn_arg_ano_list(func)
    ir_text, res_type = device_ir(func,platform)
    if platform == config.GPUPlatform.CUDA:
        ir_text = re.sub( \
            r'(?P<before>\.visible\s+\.func\s*\(\s*\.param\s+\.\w+\s+\w+\s*\)\s*)(?P<name>\w+)(?P<after>\((?P<params>(\s*\.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\))', \
            r'\g<before>'+f"_{func.__name__}_{suffix}"+r'\g<after>', \
            ir_text \
        )
    elif platform == config.GPUPlatform.ROCM:
        ir_text = re.sub( r'define', r'define internal', ir_text)
        ir_text = re.sub( r'internal hidden', r'hidden', ir_text)
    return ir_text


# Used to map numba types to numpy d_types
def map_type_to_np(kind):
    return numba.np.numpy_support.as_dtype(kind)
    primitives = {
        numba.none    : np.void,
        bool          : np.bool8,
        numba.boolean : np.bool8,
        numba.uint8   : np.uint8,
        numba.uint16  : np.uint16,
        numba.uint32  : np.uint32,
        numba.uint64  : np.uint64,
        numba.int8    : np.int8,
        numba.int16   : np.int16,
        numba.int32   : np.int32,
        numba.int64   : np.int64,
        numba.float32 : np.float32,
        numba.float64 : np.float64
    }
    if kind in primitives:
        return primitives[kind]
    return kind


# Determines the alignment of an input record type. For proper operation,
# the input type MUST be a record type
def alignment(kind):
    if   isinstance(kind,dict):
        return 8
    elif isinstance(kind,numba.types.Record):
        align = 1
        for name,type in kind.members:
            member_align = kind.alignof(name)
            if member_align != None and member_align > align:
                align = member_align
        return align
    elif isinstance(kind,numba.types.Type):
        return kind.bitwidth // 8
    else:
        raise numba.errors.TypingError(
            f"Cannot find alignment of type {kind}."
        )


# Determines the alignment of an input type. For proper operation,
# the input type MUST be a record type or a dict of dicts/records.
# This limitation applies recursively
def size_of(kind):
    align = 8
    if isinstance(kind,dict):
        result = 0
        for sub_name,sub_kind in kind.items():
            debug_print(f"Size is {result}")
            sub_size = size_of(sub_kind)
            sub_size = ((sub_size+(align-1))//align) * align
            debug_print(f"Adding {sub_size}")
            result += sub_size
        debug_print(f"Final size : {result}")
        return result
    elif isinstance(kind,numba.types.Record):
        return kind.size
    elif isinstance(kind,numba.types.Type):
        return kind.bitwidth // 8
    else:
        raise numba.errors.TypingError(
            "Cannot find size of type '{kind}'."
        )



# Maps an input type to the CUDA/C++ equivalent type name used by Harmonize
def map_type_name(type_map,kind,rec_mode=""):
    primitives = {
        numba.none : "void",
        bool       : "bool",
        np.bool8   : "bool",
        np.uint8   : "uint8_t",
        np.uint16  : "uint16_t",
        np.uint32  : "uint32_t",
        np.uint64  : "uint64_t",
        np.int8    : "int8_t",
        np.int16   : "int16_t",
        np.int32   : "int32_t",
        np.int64   : "int64_t",
        np.float32 : "float",
        np.float64 : "double"
    }

    if isinstance(kind,dict):
        size  = size_of(kind)
        align = 8
        size  = ((size + (align-1)) // align) * align
        result = f"_{size}b{align}"
        if rec_mode == "ptr":
            result += "*"
        elif rec_mode == "void_ptr":
            result = "void*"
        return result
    elif kind in primitives:
        return primitives[kind]
    elif isinstance(kind,numba.types.abstract.Literal):
        return map_type_name(type_map,type(kind._literal_value))
    elif isinstance(kind,numba.types.Integer):
        result = "int"
        if not kind.signed :
            result = "u" + result
        result += str(kind.bitwidth)
        return result + "_t"
    elif isinstance(kind,numba.types.Float):
        return "float" + str(kind.bitwidth)
    elif isinstance(kind,numba.types.Boolean):
        return "bool"
    elif isinstance(kind,numba.types.Record):
        size  = kind.size
        align = alignment(kind)
        align = 8
        size  = ((size + (align-1)) // align) * align
        result = f"_{size}b{align}"
        if rec_mode == "ptr":
            result += "*"
        elif rec_mode == "void_ptr":
            result = "void*"
        return result
    elif isinstance(kind,numba.types.npytypes.NestedArray):
        return "void*"
    else:
        raise RuntimeError("Unrecognized type '"+str(kind)+"' with type '"+str(type(kind))+"'")


# Returns the number of arguments used by the input function
def arg_count(func):
    return len(fn_arg_ano_list(func))


# Returns the CUDA/C++ text used as the parameter list for the input function, with various
# options to map record type names, account for preceding parameters, remove initial
# parameters from the signature, or force the first to be a void*.
def func_param_text(func,type_map,rec_mode="",prior=False,clip=0,first_void=False):
    param_list  = fn_arg_ano_list(func)

    param_list = param_list[clip:]

    if first_void:
        param_list = param_list[1:]
        param_list  = ["void*"] + [ map_type_name(type_map,kind,rec_mode=rec_mode) for kind in param_list ]
    else:
        param_list  = [ map_type_name(type_map,kind,rec_mode=rec_mode) for kind in param_list ]

    param_text  = ", ".join([ kind+" fn_param_"+str(idx+clip+1) for (idx,kind) in enumerate(param_list)])
    if len(param_list) > 0 and prior:
        param_text = ", " + param_text
    return param_text


# Returns the CUDA/C++ text used as the argument list for a call to the input function,
# with various options to cast/deref/getadr values, account for preceding parameters, and
# remove initial parameters from the signature.
def func_arg_text(func,type_map,rec_mode="",prior=False,clip=0):
    param_list  = fn_arg_ano_list(func)
    param_list = param_list[clip:]

    arg_text  = ""
    if len(param_list) > 0 and prior:
        arg_text = ", "
    for (idx,kind) in enumerate(param_list):
        if idx != 0:
            arg_text += ", "
        if isinstance(kind,numba.types.Record):
            if   rec_mode == "deref":
                arg_text += "*"
            elif rec_mode == "adrof":
                arg_text += "&"
            elif rec_mode == "cast_deref":
                arg_text += "*("+map_type_name(type_map,kind,rec_mode="ptr")+")"
        arg_text += "fn_param_"+str(idx+clip+1)
    return arg_text



# Returns the CUDA/C++ text representing the input function as a Harmonize async template
# function. The wrapper type that surrounds such templates is handled in other functions.
def harm_template_func(func,template_name,function_map,type_map,inline,suffix,base=False):


    return_type = "void"
    if 'return' in func.__annotations__:
        return_type = map_type_name(type_map, func.__annotations__['return'])


    param_text  = func_param_text(func,type_map,prior=True,clip=1,first_void=False)
    arg_text    = func_arg_text  (func,type_map,rec_mode="adrof",prior=True,clip=1)

    if base:
        param_text = ""
        arg_text   = ""


    code = "\ttemplate<typename PROGRAM>\n"   \
	     + "\t__device__ static "  \
         + return_type+" "+template_name+"(PROGRAM prog" + param_text + ") {\n"

    if return_type != "void":
        code += "\t\t"+return_type+"  result;\n"
        code += "\t\t"+return_type+" *fn_param_0 = &result;\n"
    else:
        code += "\t\tint  dummy_void_result = 0;\n"
        code += "\t\tint *fn_param_0 = &dummy_void_result;\n"

    if inline:
        code += inlined_device_ir(func,function_map,type_map)
    else:
        code += f"\t\t_{func.__name__}_{suffix}(fn_param_0, &prog"+arg_text+");\n"
        pass

    if return_type != "void":
        code += "\t\treturn result;\n"
    code += "\t}\n"

    return code


# Returns the input string in pascal case
def pascal_case(name):
    return name.replace("_", " ").title().replace(" ", "")


# Returns the CUDA/C++ text for a Harmonize async function that would map onto the
# input function.
def harm_async_func(func, function_map, type_map,inline,suffix):
    return_type = "void"
    if 'return' in func.__annotations__:
        return_type = map_type_name(type_map, func.__annotations__['return'])
    func_name = str(func.__name__)

    struct_name = pascal_case(func_name)
    param_list  = fn_arg_ano_list(func)[1:]
    param_list  = [ map_type_name(type_map,kind) for kind in param_list ]
    param_text  = ", ".join(param_list)

    code = "struct " + struct_name + " {\n"                                              \
	     + "\tusing Type = " + return_type + "(*)(" + param_text + ");\n"                \
         + harm_template_func(func,"eval",function_map,type_map,inline,suffix)                  \
         + "};\n"

    return code



# Returns the address of the input device array
def cuda_adr_of(array):
    return array.__cuda_array_interface__['data'][0]




def find_bundle_triples(bundle_file_name):
    file = open(bundle_file_name,"rb")
    data = file.read()
    magic = struct.unpack_from("24sQ",data,offset=0)
    if magic[0].decode('utf-8') != r"__CLANG_OFFLOAD_BUNDLE__":
        raise RuntimeError(f"Invalid clang bundle file '{bundle_file_name}' - missing magic string.")

    count = magic[1]

    if count != 2:
        raise RuntimeError(f"Invalid clang bundle file '{bundle_file_name}' - does not contain exactly 2 entries.")

    gpu_offset = 32
    gpu_header = struct.unpack_from("QQQ",data,gpu_offset)
    gpu_triple = struct.unpack_from(f"{gpu_header[2]}s",data,gpu_offset+24)

    cpu_offset = gpu_offset + 24 + gpu_header[2]
    cpu_header = struct.unpack_from("QQQ",data,cpu_offset)
    cpu_triple = struct.unpack_from(f"{cpu_header[2]}s",data,cpu_offset+24)

    gpu_triple = gpu_triple[0].decode('utf-8').rstrip('-')
    cpu_triple = cpu_triple[0].decode('utf-8').rstrip('-')

    return gpu_triple, cpu_triple

