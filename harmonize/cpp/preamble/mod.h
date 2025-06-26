#ifndef HARMONIZE_PREAMBLE
#define HARMONIZE_PREAMBLE




#define HARMONIZE

//#define DEBUG_PRINT
//#define RACE_COND_PRINT
//#define QUEUE_PRINT

#define INF_LOOP_SAFE




#define NOOP(x) ;

#ifdef QUEUE_PRINT
	#define q_printf  printf
#else
	#define q_printf(fmt, ...) NOOP(...);
#endif


#ifdef RACE_COND_PRINT
	#define rc_printf  printf
#else
	#define rc_printf(fmt, ...) ;
#endif


#ifdef DEBUG_PRINT
	#define db_printf  printf
#else
	#define db_printf(fmt, ...) ;
#endif



#include "adapt.h"


//#define HRM_TIME 16

#ifdef HRM_TIME
	#define beg_time(idx) if(util::current_leader()) { _grp_ctx.time_totals[idx] -= clock64(); }
	#define end_time(idx) if(util::current_leader()) { _grp_ctx.time_totals[idx] += clock64(); }
#else
	#define beg_time(idx) ;
	#define end_time(idx) ;
#endif





//#define ASYNC_LOADS

#ifdef ASYNC_LOADS
#include <cuda/barrier>
#define BARRIER_SPILL
#endif





#include "../util/mod.h"

#endif

