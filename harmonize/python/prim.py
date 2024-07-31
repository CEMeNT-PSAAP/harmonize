from numba.types import int8, int16,int32,int64,uint8,uint16,uint32,uint64,float32,float64
from numba.types import uintp as usize
from numba.types import voidptr as vp

prim_info = {
    int8:    { "py_name": "int8",    "cpp_name": "char" },
    int16:   { "py_name": "int16",   "cpp_name": "short" },
    int32:   { "py_name": "int32",   "cpp_name": "int" },
    int64:   { "py_name": "int64",   "cpp_name": "long long int" },
    uint8:   { "py_name": "uint8",   "cpp_name": "unsigned char" },
    uint16:  { "py_name": "uint16",  "cpp_name": "unsigned short" },
    uint32:  { "py_name": "uint32",  "cpp_name": "unsigned int" },
    uint64:  { "py_name": "uint64",  "cpp_name": "unsigned long long int" },
    float32: { "py_name": "float32", "cpp_name": "float" },
    float64: { "py_name": "float64", "cpp_name": "double" },
}


