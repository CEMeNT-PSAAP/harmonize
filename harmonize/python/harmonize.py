import numpy as np
from os.path import getmtime, exists, dirname, abspath
from os      import makedirs, getcwd, path
from numba import njit, cuda
import numba
import re
import subprocess
from llvmlite import binding
from time import sleep

from .templates import *

import inspect
import sys

NVCC_PATH = "nvcc"
HARMONIZE_ROOT_DIR =  dirname(abspath(__file__))+"/.."
HARMONIZE_ROOT_HEADER = HARMONIZE_ROOT_DIR+"/cpp/harmonize.h"

DEBUG   = False
VERBOSE = False

# Uses nvidia-smi to query the compute level of the GPUs on the system. This
# compute level is what is used for compling PTX.
def native_cuda_compute_level():
    capability = numba.cuda.get_current_device().compute_capability
    output = f"{capability[0]}{capability[1]}"
    return output


# Injects `value` as the value of the global variable named `name` in the module
# that defined the function `index` calls down the stack, from the perspective
# of the function that calls `inject_global`.
def inject_global(name,value,index):
    frm = inspect.stack()[index+1]
    mod = inspect.getmodule(frm[0])
    setattr(sys.modules[mod.__name__], name, value)
    print(f"Defined '{name}' as "+str(value)+" for module "+mod.__name__)


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
            raise(TypeError(err_str))


# Returns the ptx of the input function, as a global CUDA function. If the return type deduced
# by compilation is not consistent with the annotated return type, an exception is raised.
def global_ptx( func ):
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),debug=DEBUG,opt=(not DEBUG))
    assert_fn_res_ano(func, res_type)
    return ptx

# Returns the ptx of the input function, as a device CUDA function. If the return type deduced
# by compilation is not consistent with the annotated return type, an exception is raised.
def device_ptx( func ):
    #print(func.__name__)
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),device=True,debug=DEBUG,opt=(not DEBUG))
    assert_fn_res_ano(func, res_type)
    return ptx, res_type

# Returns the modify date of the file containing the supplied function. This is useful for
# detecting if a function was possibly changed
def func_defn_time(func):
    return getmtime(func.__globals__['__file__'])



def extern_device_ptx( func, type_map, suffix ):
    arg_types   = fn_arg_ano_list(func)
    ptx_text, res_type = device_ptx(func)
    ptx_text = re.sub( \
        r'(?P<before>\.visible\s+\.func\s*\(\s*\.param\s+\.\w+\s+\w+\s*\)\s*)(?P<name>\w+)(?P<after>\((?P<params>(\s*\.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\))', \
        r'\g<before>'+f"_{func.__name__}_{suffix}"+r'\g<after>', \
        ptx_text \
    )
    #print(ptx_text)
    return ptx_text


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
    align = 1
    for name,type in kind.members:
        member_align = kind.alignof(name)
        if member_align != None and member_align > align:
            align = member_align
    return align


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


    if kind in primitives:
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
        #if kind in type_map:
        #    return type_map[kind] + "*"
        #else:
        size  = kind.size
        align = alignment(kind)
        align = 8
        size  = ((size + (align-1)) // align) * align
        result = "_"+str(size)+"b"+str(align)
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


    #code += "\tprintf(\"{"+func.__name__+"}\");\n"
    #code += "\tprintf(\"(prog%p)\",&prog);\n"
    #code += "\tprintf(\"{ctx%p}\",&prog._dev_ctx);\n"
    #code += "\tprintf(\"(sta%p)\",prog.device);\n"
    #code += "\t//printf(\"{pre%p}\",preamble(prog.device));\n"

    if inline:
        code += inlined_device_ptx(func,function_map,type_map)
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






# A base class representing all possible runtime types
class Runtime():
    def __init__(self,spec,context,state,init_fn,exec_fn):
        pass

    def init(self,*args):
        pass

    def exec(self,*args):
        pass

    def load_state(self):
        pass

    def store_state(self):
        pass



# The class representing the Async runtime type, which
# asynchronously schedules calls on-GPU and within a single kernel
class AsyncRuntime(Runtime):

    def __init__(self,spec,context,state,fn,gpu_objects):
        self.spec          = spec
        self.context       = context
        self.context_ptr   = cuda_adr_of(context)
        self.state         = state
        self.state_ptr     = cuda_adr_of(state)
        self.fn            = fn
        self.gpu_objects   = gpu_objects

    # Initializes the runtime using `wg_count` work groups
    def init(self,wg_count=1):
        self.fn["init_program"](self.context_ptr,self.state_ptr,wg_count,32)

    # Executes the runtime through `cycle_count` iterations using `wg_count` work_groups
    def exec(self,cycle_count,wg_count=1):
        self.fn["exec_program"](self.context_ptr,self.state_ptr,cycle_count,wg_count,32)

    # Loads the device state of the runtime, returning value normally and also through the
    # supplied argument, if it is not None
    def load_state(self,result = None):
        res = self.state.copy_to_host(result)
        return res


    # Stores the input value into the device state
    def store_state(self,state):
        cpu_state = np.zeros((1,),self.spec.dev_state)
        cpu_state[0] = state
        return self.state.copy_to_device(cpu_state)


# The Harmonize implementation of standard event-based execution
class EventRuntime(Runtime):


    def __init__(self,spec,context,state,fn,checkout,io_buffers):
        self.spec          = spec
        self.context       = context
        self.context_ptr   = cuda_adr_of(context)
        self.state         = state
        self.state_ptr     = cuda_adr_of(state)
        self.fn            = fn
        self.checkout      = checkout
        self.io_buffers    = io_buffers

    # Used for debugging. Prints out the contents of the buffers used to store
    # intermediate data
    def io_summary(self):
        for field in ["handle","data_a","data_b"]:
            bufs = [ buf[field].copy_to_host() for buf in self.io_buffers ]
            for buf in bufs:
                print(buf)
        return True

    # Returns true if and only if the instance has halted, indicating that no
    # work is left to be performed
    def halted(self):
        bufs = [ buf["handle"].copy_to_host()[0] for buf in self.io_buffers ]
        for buf in bufs:
            if buf["input_iter"]["limit"] != 0:
                return False
        return True

    # Does nothing, since no on-GPU initialization is required for the runtime
    def init(self,wg_count=1):
        pass

    # Executes the program, claiming events from the intermediate data buffers in
    # `chunk_size`-sized units for each thread. Execution continues until the program
    # is halted
    def exec(self,chunk_size,wg_count=1):
        has_halted = False
        count = 0
        while not has_halted:
            self.exec_fn[wg_count,32](self.context_ptr,self.state_ptr,chunk_size)
            has_halted = self.halted()
            count += 1

    # Loads the device state of the runtime, returning value normally and also through the
    # supplied argument, if it is not None
    def load_state(self,result = None):
        res = self.state.copy_to_host(result)
        return res

    # Stores the input value into the device state
    def store_state(self,state):
        cpu_state = np.zeros((1,),self.spec.dev_state)
        cpu_state[0] = state
        return self.state.copy_to_device(cpu_state)


# Represents the specification for a specific program and its
# runtime meta-parameters
class RuntimeSpec():

    obj_set    = set()
    registry   = {}
    kinds = [("Event","event"),("Async","async")]
    compute_level = native_cuda_compute_level()
    cache_path = "__ptxcache__/"
    debug_flag = " -g "
    dirty      = False

    def __init__(
            self,
            # The name of the program specification
            spec_name,
            # A triplet containing types which should be
            # held per-device, per-group, and per-thread
            state_spec,
            # A triplet containing the functions that
            # should be run to initialize/finalize states
            # and to generate work for running programs
            base_fns,
            # A list of python functions that are to be
            # included as async functions in the program
            async_fns,
            **kwargs
        ):

        self.spec_name = spec_name


        # The states tracked per-device, per-group, and per-thread
        self.dev_state, self.grp_state, self.thd_state = state_spec

        # The base functions used to set up states and generate work
        self.init_fn, self.final_fn, self.source_fn = base_fns

        self.async_fns = async_fns

        self.meta = kwargs

        if 'function_map' in kwargs:
            self.function_map = kwargs['function_map']
        else:
            self.function_map = {}

        if 'type_map' in kwargs:
            self.type_map = kwargs['type_map']
        else:
            self.type_map = {}

        self.generate_meta()
        self.generate_code()

        RuntimeSpec.registry[self.spec_name] = self




    # Generates the meta-parameters used by the runtimme, applying defaults for whichever
    # fields that aren't defined
    def generate_meta(self):

        # The number of work links that work groups hold in their shared memory
        if 'STASH_SIZE' not in self.meta:
            self.meta['STASH_SIZE'] = 8

        # The size of the work groups used by the program. Currently, the only
        # supported size is 32
        if 'GROUP_SIZE' not in self.meta:
            self.meta['GROUP_SIZE'] = 32

        # Finding all input types, each represented as dtypes
        input_list = []
        for async_fn in self.async_fns:
            arg_list = []
            for idx, arg in enumerate(fn_arg_ano_list(async_fn)):
                np_type = map_type_to_np(arg)
                arg_list.append(("param_"+str(idx),np_type))
            input_list.append(np.dtype(arg_list))

        # The maximum size of all input types. This must be known in order to
        # allocate memory spans with a size that can support all async functions
        max_input_size = 0
        for input in input_list:
            max_input_size = max(max_input_size,input.itemsize)
        self.meta['MAX_INP_SIZE'] = max_input_size #type_.particle.itemsize

        # The integer type used by the system to address promises
        self.meta['ADR_TYPE']  = np.uint32

        # The type used to store per-link meta-data, currently used only
        # for debugging purposes
        self.meta['META_TYPE'] = np.uint32

        # The type used to store the per-link discriminant, identifying the
        # async call that corresponds to the link's data
        if 'OP_DISC_TYPE' not in self.meta:
            self.meta['OP_DISC_TYPE'] = np.uint32

        # The type used to represent the union of all potential inputs
        self.meta['UNION_TYPE']       = np.dtype((np.void,self.meta['MAX_INP_SIZE']))

        # The type used to represent the array of promises held by a link
        self.meta['PROMISE_ARR_TYPE'] = np.dtype((self.meta['UNION_TYPE'],self.meta['GROUP_SIZE']))

        # The work link type, the fundamental type used to organize work
        # in the system
        self.meta['WORK_LINK_TYPE']   = np.dtype([
            ('promises',self.meta['PROMISE_ARR_TYPE']),
            ('next',self.meta['ADR_TYPE']),
            ('meta_data',self.meta['META_TYPE']),
            ('id',self.meta['OP_DISC_TYPE']),
        ])

        # The integer type used to iterate across IO buffers
        self.meta['IO_ITER_TYPE'] = self.meta['ADR_TYPE']

        # The atomic iterator type used to iterate across IO buffers
        self.meta['IO_ATOMIC_ITER_TYPE'] = np.dtype([
            ('value',self.meta['IO_ITER_TYPE']),
            ('limit',self.meta['IO_ITER_TYPE']),
        ])

        # The atomic iterator type used to iterate across IO buffers
        self.meta['IO_BUFFER_TYPE'] = np.dtype([
            ('toggle',     np.bool_),
            ('padding1',   np.uint8),
            ('padding2',   np.uint16),
            ('padding3',   np.uint32),
            ('data_a',     np.uintp),
            ('data_b',     np.uintp),
            ('capacity',   self.meta['IO_ITER_TYPE']),
            ('input_iter', self.meta['IO_ATOMIC_ITER_TYPE']),
            ('output_iter',self.meta['IO_ATOMIC_ITER_TYPE']),
        ])

        # The array of IO buffers used by the event-based method
        self.meta['IO_BUFFER_ARRAY_TYPE'] = np.dtype(
            (np.uintp,(len(self.async_fns),))
        )


        # The size of the array used to house all work links (measured in links)
        if 'WORK_ARENA_SIZE' not in self.meta:
            self.meta['WORK_ARENA_SIZE'] = 65536

        # The type used as a handle to the arena of work links
        self.meta['WORK_ARENA_TYPE']     = np.dtype([('size',np.intp),('links',np.intp)])

        # The number of independent queues maintained per stack frame and per
        # link pool
        if 'QUEUE_COUNT' not in self.meta:
            self.meta['QUEUE_COUNT']     = 8192 #1024
        # The type of value used to represent links
        self.meta['QUEUE_TYPE']          = np.intp
        # The type used to hold a series of work queues
        self.meta['WORK_POOL_ARR_TYPE']  = np.dtype((
            self.meta['QUEUE_TYPE'],
            (self.meta['QUEUE_COUNT'],)
        ))

        # The struct type that wraps a series of work queues
        self.meta['WORK_POOL_TYPE']      = np.dtype([
            ('queues',self.meta['WORK_POOL_ARR_TYPE'])
        ])
        # The type used to represent
        self.meta['WORK_FRAME_TYPE']     = np.dtype([
            ('children_residents',np.uint32),
            ('padding',np.uintp),
            ('pool',self.meta['WORK_POOL_TYPE'])
        ])

        # The size of the stack (only size-one stacks are currently supported)
        self.meta['STACK_SIZE']         = 1
        # The type used to hold a series of frames
        self.meta['FRAME_ARR_TYPE']     = np.dtype((
            self.meta['WORK_FRAME_TYPE'],
            (self.meta['STACK_SIZE'],)
        ))
        # The type used to hold a work stack and it's associated meta-data
        self.meta['WORK_STACK_TYPE']    = np.dtype([
            ('checkout',    np.uint32),
            ('status_flags',np.uint32),
            ('depth_live',  np.uint32),
            ('frames',      self.meta['FRAME_ARR_TYPE'])
        ])

        # The device context type, tracking the arena, how much of the arena has
        # been claimed the 'easy' way, the pool for allocating/deallocating work links
        # in the arena, and the stack for tracking work links that contain outstanding
        # promises waiting for processing
        self.meta['DEV_CTX_TYPE'] = {}

        self.meta['DEV_CTX_TYPE']["Async"] = np.dtype([
            ('claim_count',np.intp),
            ('arena',      self.meta['WORK_ARENA_TYPE']),
            ('pool' ,      np.intp),
            ('stack',      np.intp)
        ])

        self.meta['DEV_CTX_TYPE']["Event"] = np.dtype([
            ('checkout',   np.uintp),
            ('load_margin',np.uint32),
            ('padding',    np.uint32),
            ('event_io',   self.meta['IO_BUFFER_ARRAY_TYPE']),
        ])

        return self.meta



    # Generates the CUDA/C++ code specifying the program and its rutimme's
    # meta-parameters
    def generate_specification_code(
            self,
            suffix,
            # Whether or not the async functions should be inlined through ptx
            # (currently, this feature is NOT supported)
            inline=False
        ):


        # Accumulator for type definitions
        type_defs = ""

        # A map to store the set of parameter size/alignment specifications
        param_specs = {}

        # Add in parameter specs from the basic async functions
        for func in self.async_fns:
            for kind in fn_arg_ano_list(func):
                if isinstance(kind,numba.types.Record):
                    size      = kind.size
                    align     = alignment(kind)
                    param_specs[(size,align)] = ()

        # Add in parameter specs from the required async functions
        state_kinds = [self.dev_state, self.grp_state, self.thd_state]
        for kind in state_kinds:
                if isinstance(kind,numba.types.Record):
                    size      = kind.size
                    align     = alignment(kind)
                    param_specs[(size,align)] = ()


        # An accumulator for parameter type declarations
        param_decls = ""

        # A map from alignment sizes to their corresponding element type
        element_map = {
            1 : "unsigned char",
            2 : "unsigned short int",
            4 : "unsigned int",
            8 : "unsigned long long int"
        }

        # Create types matching the required size and alignment. Until reliable alignment
        # deduction is implemented, an alignment of 8 will always be used.
        for size, align in param_specs.keys():
            align = 8
            count =  (size + (align-1)) // align
            size  = ((size + (align-1)) // align) * align
            param_decls += "struct _"+str(size)+"b"+str(align) \
                        +" { "+element_map[align]+" data["+str(count)+"]; };\n"



        # Accumulator for prototypes of extern definitions of async functions, initialized
        # with prototypes corresponding to the required async functions
        proto_decls = ""                                                             \
            + f"extern \"C\" __device__ int _initialize_{suffix}(void*, void* prog);\n"        \
            + f"extern \"C\" __device__ int _finalize_{suffix}  (void*, void* prog);\n"        \
            + f"extern \"C\" __device__ int _make_work_{suffix} (bool* result, void* prog);\n"

        # Generate and accumulate prototypes for other async function definitons
        for func in self.async_fns:
            param_text = func_param_text(func,self.type_map,rec_mode="void_ptr",prior=True,clip=0,first_void=True)
            return_type = "void"
            if 'return' in func.__annotations__:
                return_type = map_type_name(self.type_map, func.__annotations__['return'])
            proto_decls += f"extern \"C\" __device__ int _{func.__name__}_{suffix}" \
                        +  f"({return_type}*{param_text});\n"

        # Accumulator for async function definitions
        async_defs = ""

        # Forward-declare the structs used to house the definitons
        for func in self.async_fns:
            async_defs += "struct " + pascal_case(func.__name__) + ";\n"

        # Generate an async function struct (with appropriate members)
        # for each async function provided, either attempting to inline the
        # function definition (this currently does not work), or inserting
        # a call to a matching extern function, allowing the scheduling
        # program to jump to the appropriate logic
        for func in self.async_fns:
            async_defs += harm_async_func(func,self.function_map,self.type_map,inline,suffix)

        # The metaparameters of the program specification, indicating the sizes
        # of the data structures used
        metaparams = {
            "STASH_SIZE" : self.meta['STASH_SIZE'],
            "FRAME_SIZE" : self.meta['QUEUE_COUNT'],
            "POOL_SIZE"  : self.meta['QUEUE_COUNT'],
        }

        # The declarations of each of the metaparameters
        meta_defs = "".join(["\tstatic const size_t "+name+" = "+str(value) +";\n" for name,value in metaparams.items()])

        # The declaration of the union of all input types
        union_def = "\ttypedef OpUnion<"+",".join([pascal_case(str(func.__name__)) for func in self.async_fns])+"> OpSet;\n"

        # The definition of the device, group, and thread states
        state_defs = "\ttypedef "+map_type_name(self.type_map,self.dev_state,rec_mode="ptr")+" DeviceState;\n" \
                   + "\ttypedef "+map_type_name(self.type_map,self.grp_state,rec_mode="ptr")+" GroupState;\n"  \
                   + "\ttypedef "+map_type_name(self.type_map,self.thd_state,rec_mode="ptr")+" ThreadState;\n"

        # The base function definitions
        spec_def = "struct " + self.spec_name + "{\n"                                                     \
                 + meta_defs + union_def + state_defs                                                     \
                 + harm_template_func(self.init_fn  ,"initialize",self.function_map,self.type_map,inline,suffix,True) \
                 + harm_template_func(self.final_fn ,"finalize"  ,self.function_map,self.type_map,inline,suffix,True) \
                 + harm_template_func(self.source_fn,"make_work" ,self.function_map,self.type_map,inline,suffix,True) \
                 + "};\n"

        return type_defs + param_decls + proto_decls + async_defs + spec_def



    # Returns the CUDA/C++ code specializing the specification for a program type
    def generate_specialization_code(self,kind,shorthand,suffix):
        # String template to alias the appropriate specialization to a convenient name
        spec_decl_template = "typedef {kind}Program<{name}> {short_name};\n"


        state_struct = map_type_name(self.type_map,self.dev_state,rec_mode="")

        # The set of fields that should have accessors, each annotated with
        # the code (if any) that should prefix references to those fields.
        # This is mainly useful for working with references.
        program_fields = [
            (  "device", ""), (   "group", ""), (  "thread", ""),
            #("_dev_ctx","&"), ("_grp_ctx","&"), ("_thd_ctx","&")
        ]


        # Accumulator for includes and initial declarations/typedefs
        preamble      = ""

        # Accumulator for init/exec/async/sync wrappers
        dispatch_defs = ""
        # Accumulator for accessors
        accessor_defs = ""

        # The name used to refer to the program template specialization
        short_name = self.spec_name.lower() +"_"+shorthand
        # Typedef the specialization to a more convenient shothand
        preamble += spec_decl_template.format(kind=kind,name=self.spec_name,short_name=short_name,suffix=suffix)

        # Generate the wrappers for kernel entry points
        dispatch_defs += init_template.format(short_name=short_name,name=self.spec_name,kind=kind,suffix=suffix)
        dispatch_defs += exec_template.format(short_name=short_name,name=self.spec_name,kind=kind,suffix=suffix)

        if kind == "Event":
            dispatch_defs += alloc_event_prog_template.format(short_name=short_name,suffix=suffix)
        else :
            dispatch_defs += alloc_harm_prog_template.format(short_name=short_name,suffix=suffix)

        dispatch_defs += free_prog_template  .format(short_name=short_name,suffix=suffix)
        dispatch_defs += alloc_state_template.format(state_struct=state_struct,suffix=suffix)
        dispatch_defs += free_state_template .format(suffix=suffix)
        dispatch_defs += load_state_template .format(state_struct=state_struct,suffix=suffix)
        dispatch_defs += store_state_template.format(state_struct=state_struct,suffix=suffix)
        dispatch_defs += complete_template   .format(short_name=short_name,suffix=suffix)
        dispatch_defs += clear_flags_template.format(short_name=short_name,suffix=suffix)


        # Generate the dispatch functions for each async function
        for fn in self.async_fns:
            # Accepts record parameters as void pointers
            param_text = func_param_text(fn,self.type_map,rec_mode="void_ptr",prior=True,clip=0,first_void=True)
            # Casts the void pointers of parameters to types with matching size and alignment
            arg_text   = func_arg_text(fn,self.type_map,rec_mode="cast_deref",prior=False,clip=1)
            # Generates a wrapper for both async and sync dispatches
            for kind in ["async","sync"]:
                dispatch_defs += dispatch_template.format(
                    short_name=short_name,
                    fn=fn.__name__,
                    fn_type=pascal_case(fn.__name__),
                    params=param_text,
                    args=arg_text,
                    kind=kind,
                    suffix=suffix,
                )

        # Creates a field accesing function for each field
        for (field,prefix) in program_fields:
            accessor_defs += accessor_template.format(short_name=short_name,field=field,prefix=prefix,suffix=suffix)

        # Query definitions currently disabled
        fn_query_defs = ""
        query_defs = ""
        if False:
            fn_query_list = [ ("load_fraction","float", "") ]
            for (field,kind,prefix) in fn_query_list:
                for fn in self.async_fns:
                    fn_query_defs += fn_query_template.format(
                            short_name=short_name,
                            fn_type=pascal_case(fn.__name__),
                            field=field,
                            kind=kind,
                            prefix=prefix,
                        )

            query_list = [ ]
            for (field,kind,prefix) in query_list:
                query_defs += query_template.format(
                    short_name=short_name,
                    field=field,
                    kind=kind,
                    prefix=prefix
                )

        return preamble + dispatch_defs + accessor_defs + fn_query_defs + query_defs


    def generate_async_ptx(self,cache_path,suffix):
        # The list of required async functions
        base_fns = [self.init_fn, self.final_fn, self.source_fn]
        # The full list of async functions
        comp_list = [fn for fn in base_fns] + self.async_fns



        rep_list  = [ f"_{fn.__name__}" for fn in comp_list]
        rep_list += [ f"dispatch_{fn.__name__}_async" for fn in comp_list ]
        rep_list += [ f"dispatch_{fn.__name__}_sync" for fn in comp_list ]
        rep_list += [ f"access_{state}" for state in ["device","group","thread"] ]

        # Compile each user-provided function defintion to ptx
        # and save it to an appropriately named file
        for fn in comp_list:
            base_name = fn.__name__
            base_name = base_name + "_" + suffix
            base_name = cache_path + base_name

            ptx_path = f"{base_name}.ptx"
            obj_path = f"{base_name}.o"
            def_path = inspect.getfile(fn)

            touched = False
            if not path.isfile(def_path):
                touched = True
            elif not path.isfile(ptx_path):
                touched = True
            elif getmtime(def_path) > getmtime(ptx_path):
                touched = True

            if touched:

                ptx_text  = extern_device_ptx(fn,self.type_map,suffix)
                for term in rep_list:
                    ptx_text = re.sub( \
                        f'(?P<before>[^a-zA-Z0-9_])(?P<name>{term})(?P<after>[^a-zA-Z0-9_])', \
                        f"\g<before>\g<name>_{suffix}\g<after>", \
                        ptx_text \
                    )
                ptx_file  = open(ptx_path,mode='a+')
                ptx_file.seek(0)
                old_text  = ptx_file.read()

                dirty = False
                if old_text != ptx_text:
                    dirty = True
                    RuntimeSpec.dirty = True

                if touched or dirty:
                    ptx_file.seek(0)
                    ptx_file.truncate()
                    ptx_file.write(ptx_text)
                ptx_file.close()

                if dirty:
                    dev_comp_cmd = f"{NVCC_PATH} -rdc=true -dc -arch=compute_{RuntimeSpec.compute_level} --cudart shared --compiler-options -fPIC {ptx_path} -o {obj_path} {RuntimeSpec.debug_flag}"
                    if VERBOSE:
                        print(dev_comp_cmd)
                    subprocess.run(dev_comp_cmd.split(),shell=False,check=True)

            # Record the path of the generated (or pre-existing) object
            RuntimeSpec.obj_set.add(obj_path)


    # Generates the CUDA/C++ code specifying the structure of the program, for later
    # specialization to a specific program type, and compiles the ptx for each async
    # function supplied to the specfication. Both this cuda code and the ptx are
    # saved to the `__ptxcache__` directory for future re-use.
    def generate_code(self):

        # Folder used to cache cuda and ptx code

        makedirs(RuntimeSpec.cache_path,exist_ok=True)

        self.fn = {}

        # A list to record the files containing the definitions
        # of each async function definition
        self.fn_def_list = []

        self.link_list = []

        # Generate and compile specializations of the specification for
        # each kind of runtime
        for kind, shortname in RuntimeSpec.kinds:

            self.fn[kind] = {}

            # Generate the cuda code implementing the specialization
            suffix = self.spec_name+"_"+shortname

            # Generate and save generic program specification
            base_code = self.generate_specification_code(suffix)

            # Compile the async function definitions to ptx
            self.generate_async_ptx(RuntimeSpec.cache_path,suffix)

            spec_code = self.generate_specialization_code(kind,shortname,suffix)
            # Save the code to an appropriately named file
            spec_filename = RuntimeSpec.cache_path+suffix
            spec_file = open(spec_filename+".cu",mode='a+')
            spec_file.seek(0)
            old_text = spec_file.read()
            new_text = base_code + spec_code
            if old_text != new_text:
                RuntimeSpec.dirty = True
                spec_file.seek(0)
                spec_file.truncate()
                spec_file.write(new_text)
            spec_file.close()


            source_list = [ (f"{spec_filename}.cu", f"{spec_filename}.o") ]
            for (source,obj) in source_list:

                touched = False
                if not path.isfile(obj):
                    touched = True
                elif getmtime(source) > getmtime(obj):
                    touched = True

                if touched:
                    RuntimeSpec.dirty = True
                    dev_comp_cmd = f"{NVCC_PATH} -x cu -rdc=true -dc -arch=compute_{RuntimeSpec.compute_level} --cudart shared --compiler-options -fPIC {source} -include {HARMONIZE_ROOT_HEADER} -o {obj} {RuntimeSpec.debug_flag}"
                    if VERBOSE:
                        print(dev_comp_cmd)
                    subprocess.run(dev_comp_cmd.split(),shell=False,check=True)
                RuntimeSpec.obj_set.add(obj)





    def async_functions(self):
        return self.fn["Async"]

    def event_functions(self):
        return self.fn["Event"]




    # Injects the async/sync dispatch functions and state accessors into the
    # global scope of the caller. This should only be done if you are sure that
    # the injection of these fields into the global namespace of the calling
    # module won't overwrite anything. While convenient in some respects, this
    # function also gives no indication to linters that the corresponding fields
    # will be injected, leading to linters incorrectly (though understandably)
    # marking the fields as undefined
    def inject_fns(state_spec,async_fns):

        dev_state, grp_state, thd_state = state_spec

        for func in async_fns:
            sig = fn_sig(func)
            name = func.__name__
            for kind in ["async","sync"]:
                dispatch_fn = cuda.declare_device("dispatch_"+name+"_"+kind, sig)
                inject_global(kind+"_"+name,dispatch_fn,1)

        field_list = [
            ("device",dev_state),
            ("group",grp_state),
            ("thread",thd_state),
        ]
        for name, kind in field_list:
            sig = kind(numba.uintp)
            access_fn = cuda.declare_device("access_"+name,sig)
            inject_global(name,access_fn,1)

    # Returns the `device`, `group`, and `thread` accessor function
    # handles of the specification as a triplet
    def access_fns(state_spec):

        dev_state, grp_state, thd_state = state_spec

        field_list = [
            ("device",dev_state),
            ("group",grp_state),
            ("thread",thd_state),
        ]

        result = []
        for name, kind in field_list:
            sig = kind(numba.uintp)
            result.append(cuda.declare_device("access_"+name,sig))
        return tuple(result)

    # Returns the async/sync function handles for the supplied functions, using
    # `kind` to switch between async and sync
    def dispatch_fns(kind,*async_fns):
        async_fns, = async_fns
        result = []
        for func in async_fns:
            sig = fn_sig(func)
            name = func.__name__
            result.append(cuda.declare_device("dispatch_"+name+"_"+kind, sig))
        return tuple(result)

    #def query(kind,*fields):
    #    fields, = fields
    #    result = []
    #    for field in fields:
    #        result.append(cuda.declare_device("dispatch_"+name+"_"+kind, sig))
    #    return tuple(result)


    # Returns the handles for the on-gpu functions used to asynchronously schedule
    # the supplied function
    def async_dispatch(*async_fns):
        return RuntimeSpec.dispatch_fns("async",async_fns)

    # Returns the handles for the on-gpu functions used to immediately call
    # the supplied function
    def sync_dispatch(*async_fns):
        return RuntimeSpec.dispatch_fns("sync",async_fns)


    @staticmethod
    def bind_and_load():

        dev_path = f"{RuntimeSpec.cache_path}harmonize_device.o"
        so_path  = f"{RuntimeSpec.cache_path}harmonize.so"
        touched = False

        if not path.isfile(dev_path):
            touched = True
        elif not path.isfile(so_path):
            touched = True

        if touched or RuntimeSpec.dirty:
            link_list = [ obj for obj in  RuntimeSpec.obj_set ]

            dev_link_cmd = f"{NVCC_PATH} -dlink {' '.join(link_list)} -arch=compute_{RuntimeSpec.compute_level} --cudart shared -o {dev_path} --compiler-options -fPIC {RuntimeSpec.debug_flag}"

            comp_cmd = f"{NVCC_PATH} -shared {' '.join(link_list)} {dev_path} -arch=compute_{RuntimeSpec.compute_level} --cudart shared -o {so_path} {RuntimeSpec.debug_flag}"

            if VERBOSE:
                print(dev_link_cmd)
            subprocess.run(dev_link_cmd.split(),shell=False,check=True)

            if VERBOSE:
                print(comp_cmd)
            subprocess.run(comp_cmd.split(),shell=False,check=True)


        abs_so_path = abspath(so_path)
        #print(path)
        binding.load_library_permanently(abs_so_path)

        for name, spec in RuntimeSpec.registry.items():
            for kind, shortname in RuntimeSpec.kinds:

                suffix = spec.spec_name+"_"+shortname

                # Create handles to reference the cuda entry wrapper functions
                void    = numba.types.void
                vp      = numba.types.voidptr
                i32     = numba.types.int32
                usize   = numba.types.uintp
                sig     = numba.core.typing.signature
                ext_fn  = numba.types.ExternalFunction
                context = numba.from_dtype(spec.meta['DEV_CTX_TYPE'][kind])
                boolean = numba.types.boolean
                #print("\n\n\n\n",self.dev_state,"\n\n\n")
                state   = spec.dev_state #numba.from_dtype(self.dev_state)

                init_program  = ext_fn(f"init_program_{suffix}",  sig(void, vp, usize))
                exec_program  = ext_fn(f"exec_program_{suffix}",  sig(void, vp, usize, usize))
                store_state   = ext_fn(f"store_state_{suffix}",   sig(void, vp, state))
                load_state    = ext_fn(f"load_state_{suffix}",    sig(void, state, vp))

                if kind == "Event":
                    # IO_SIZE, LOAD_MARGIN
                    alloc_program = ext_fn(f"alloc_program_{suffix}", sig(vp,vp,usize))
                    free_program  = ext_fn(f"free_program_{suffix}",  sig(void,vp))
                else:
                    # ARENA SIZE, POOL SIZE, STACK SIZE
                    alloc_program = ext_fn(f"alloc_program_{suffix}", sig(vp,vp,usize))
                    free_program  = ext_fn(f"free_program_{suffix}",  sig(void,vp))

                alloc_state   = ext_fn(f"alloc_state_{suffix}",   sig(vp))
                free_state    = ext_fn(f"free_state_{suffix}",    sig(void,vp))

                complete      = ext_fn(f"complete_{suffix}",    sig(i32,vp))
                clear_flags   = ext_fn(f"clear_flags_{suffix}", sig(void,vp))

                # Finally, compile the entry functions, saving it for later use
                spec.fn[kind]['init_program']  = init_program
                spec.fn[kind]['exec_program']  = exec_program

                spec.fn[kind]['store_state']   = store_state
                spec.fn[kind]['load_state']    = load_state

                spec.fn[kind]['alloc_program'] = alloc_program
                spec.fn[kind]['free_program']  = free_program

                spec.fn[kind]['alloc_state']   = alloc_state
                spec.fn[kind]['free_state']    = free_state

                @njit(sig(boolean,vp))
                def complete_wrapper(instance):
                    result = complete(instance)
                    return (result != 0)

                spec.fn[kind]['complete']      = complete_wrapper
                spec.fn[kind]['clear_flags']   = clear_flags

