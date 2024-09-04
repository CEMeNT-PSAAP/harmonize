import numba
import llvmlite
from .templates import print_template
from .prim import *


print_specialization_registry = {}
print_extern_registry = {}


@numba.njit
def print_formatted(value):
    print_formatted_inner(value)



def print_formatted_inner(value):
    pass
    #print("(",value,")")



@numba.extending.overload(print_formatted_inner,target='cpu')
def cpu_print_formatted_overload(value):

    print(value,flush=True)

    def impl(value):
        pass
        #print("(",value,")")

    return impl


# WIP Printing function
@numba.extending.overload(print_formatted_inner,target='gpu')
def gpu_print_formatted_overload(value):

    kind = None
    if isinstance(value,numba.types.Literal):
        kind = value.literal_type
    else:
        kind = value

    if not (kind in prim_info):
        print(f"{kind} not in prim_info",flush=True)
        return None


    if not (kind in print_specialization_registry):
        print(f"{kind} not in spec registry",flush=True)
        py_name  = prim_info[kind]["py_name"]
        cpp_name = prim_info[kind]["cpp_name"]
        fmt_name = prim_info[kind]["fmt_name"]
        print_specialization_registry[kind] = {
            "type_sig"   : py_name,
            "args"       : f"{cpp_name} v0",
            "arg_vals"   : "v0",
            "format_str" : fmt_name
        }
        print(print_specialization_registry[kind])

        sig     = numba.core.typing.signature
        ext_fn  = numba.types.ExternalFunction

        signature = sig(void,kind)
        print(signature)
        py_name   = prim_info[kind]["py_name"]
        implementation = numba.hip.declare_device(f"harmonize_print_{py_name}",  signature)
        #implementation = ext_fn(f"harmonize_print_{py_name}",  signature)
        print_extern_registry[kind] = implementation
    else:
        print(f"{kind} already in spec registry",flush=True)

    inner_impl = print_extern_registry[kind]
    print(inner_impl,flush=True)

    def impl(value):
        pass
        #inner_impl(value)

    print(impl,flush=True)

    return impl


def generate_print_code():
    text = ""
    for pair in print_specialization_registry.items():
        kind, info = pair
        text += print_template.format(
            type_sig   = info["type_sig"],
            args       = info["args"],
            arg_vals   = info["arg_vals"],
            format_str = info["format_str"],
        )
    return text


