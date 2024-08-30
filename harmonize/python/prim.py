from numba.types import int8, int16,int32,int64,uint8,uint16,uint32,uint64,float32,float64
from numba.types import uintp as usize
from numba.types import voidptr as vp
from numba.types import void

prim_info = {
    int8:    {
        "py_name": "int8",
        "cpp_name": "char",
        "fmt_name": "%hhd"
    },
    int16:   {
        "py_name": "int16",
        "cpp_name": "short",
        "fmt_name": "%hd"
    },
    int32:   {
        "py_name": "int32",
        "cpp_name": "int",
        "fmt_name": "%d"
    },
    int64:   {
        "py_name": "int64",
        "cpp_name": "long long int",
        "fmt_name": "%lld"
    },
    uint8:   {
        "py_name": "uint8",
        "cpp_name": "unsigned char",
        "fmt_name": "%hhu"
    },
    uint16:  {
        "py_name": "uint16",
        "cpp_name": "unsigned short",
        "fmt_name": "%hu"
    },
    uint32:  {
        "py_name": "uint32",
        "cpp_name": "unsigned int",
        "fmt_name": "%u"
    },
    uint64:  {
        "py_name": "uint64",
        "cpp_name": "unsigned long long int",
        "fmt_name": "%llu"
    },
    float32: {
        "py_name": "float32",
        "cpp_name": "float",
        "fmt_name": "%f"
    },
    float64: {
        "py_name": "float64",
        "cpp_name": "double",
        "fmt_name": "%f"
    },
}


