

#ifndef HARMONIZE_PREAMBLE
#define HARMONIZE_PREAMBLE


#include "math.h"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <memory>
#include <stdexcept>


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


//#define HRM_TIME 16

#ifdef HRM_TIME
	#define beg_time(idx) if(util::current_leader()) { _grp_ctx.time_totals[idx] -= clock64(); }
	#define end_time(idx) if(util::current_leader()) { _grp_ctx.time_totals[idx] += clock64(); }
#else
	#define beg_time(idx) ;
	#define end_time(idx) ;
#endif

#include "../util/mod.h"

//#define ASYNC_LOADS

#ifdef ASYNC_LOADS
#include <cuda/barrier>
#define BARRIER_SPILL
#endif



//! A set of templates used to detect the presence of internal types
//! and constants.
namespace detector {

	template <class... TYPES>
	using void_t = void;

	template <size_t VALUE>
	using void_size = void;

	template <template <class...> class LOOKUP, class GUARD, class... ARGS>
	struct is_detected
	{ const static bool value = false; };

	template <template <class...> class LOOKUP, class... ARGS>
	struct is_detected<LOOKUP, void_t<LOOKUP<ARGS...>>, ARGS...>
	{ const static bool value = true; };

	template<bool COND, class TRUE_TYPE = void, class FALSE_TYPE = void>
	struct type_switch;

	template<class TRUE_TYPE, class FALSE_TYPE>
	struct type_switch<true, TRUE_TYPE,FALSE_TYPE>
	{
		using type = TRUE_TYPE;
	};

	template<class TRUE_TYPE, class FALSE_TYPE>
	struct type_switch<false, TRUE_TYPE, FALSE_TYPE>
	{
		using type = FALSE_TYPE;
	};


}

//! Returns a type that communicates whether or not a certain templat can be
//! instantiated with the given set of arguments
template<template<class...> class LOOKUP, class... ARGS>
using is_detected = typename detector::is_detected<LOOKUP,void,ARGS...>;

//! Switches an internal type between the first and last type argument based upon whether or not
//! the second, template parameter can be instantiated with the last type argument.
template<class DEFAULT, template<class> class LOOKUP, class TYPE>
using type_switch = typename detector::type_switch<is_detected<LOOKUP,TYPE>::value ,TYPE,DEFAULT>::type;



#include "../util/mod.h"

#endif


