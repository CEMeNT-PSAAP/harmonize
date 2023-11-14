#pragma once


#if defined(__HIP_PLATFORM_AMD__)
#undef __noinline__
#endif

#include "math.h"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <iostream>

#if defined(__HIP_PLATFORM_AMD__)
#define __noinline__ __attribute__((noinline))
#endif
