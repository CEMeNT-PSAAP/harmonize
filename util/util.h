#pragma once
#include "includes.h"


namespace util {




#if defined(__NVCC__) || HIPIFY

	#include "basic.h"

	#include "host.h"

	#include "mem.h"

	#include "iter.h"

	#include "cli.h"

#elif defined(__HIP__)
	
	#include "basic.h.hip"

	#include "host.h.hip"

	#include "mem.h.hip"

	#include "iter.h.hip"

	#include "cli.h.hip"

#endif


}




