#!/bin/python3

import os
import sys
from enum import Enum
from subprocess import Popen, check_call, check_output, STDOUT, PIPE
from tempfile import NamedTemporaryFile


argc = len(sys.argv)

if ( argc < 8 or argc > 9 ):
	print("Invalid argument count. 7 or 8 arguments required")
	sys.exit(1)

captX   = float(sys.argv[1])
scatX   = float(sys.argv[2])
fissX   = float(sys.argv[3])
samp    = int  (sys.argv[4])
sec     = int  (sys.argv[5])
num     = int  (sys.argv[6])
dev_idx = int  (sys.argv[7])

size    = sec + 1

max_pool_size = 134217728
#max_pool_size = 33554432

wg_count = 1024

hrzn_range=[ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 ]

opt=""
suf=f"_s{sec}_n{num}"
mark=""

if   ( argc == 9 ):
	wlim_str = sys.argv[8]
	wlim_val = float(wlim_str)
	opt = f"-imp_cap -wlim {wlim_str}"
	mark = f"_imp_{wlim_str}"
	tune = f"tune_imp_{wlim_str}"
	


superdirname = f"bench{suf}{mark}"

filename  = f"c{captX}s{scatX}f{fissX}"

base_args = f"-cx {captX} -sx {scatX} -fx {fissX} -res 0.1 -size {size} -time {sec} -num {num} -wg_count {wg_count} -mult 2 {opt} -dev_idx {dev_idx}"

print( f"filename is {filename} and superdirname is {superdirname}" )
print( f"base_args is '{base_args}'" )


class RunMode(Enum):
	RUNTIME   = 0
	MEMORY    = 2


def perform_run (exe,horizon,pool_size,mode,dup=1):
	command  = f"{exe}"
	if ( mode == RunMode.MEMORY ):
		command += "_level"
	command += f" {base_args} -hrzn {horizon} -pool {pool_size} -dup {dup}"
	if ( mode == RunMode.RUNTIME ) :
		command += " -value"

	print(command,flush=True)	
	cmd_list = command.split()
	
	result_str = check_output(cmd_list, shell=False ).decode()

	if   ( mode == RunMode.MEMORY ) :
		sample_strings = result_str.split(';')
		total = 0
		for string in sample_strings:
			total += int(string)
		result = int( total / dup )
		return int  (total)
	else:
		sample_strings = result_str.split(';')
		result = []
		for string in sample_strings:
			pairlist = string.split(',')
			result .append( ( float(pairlist[0]),  float(pairlist[1]) ) )
		return result

do_debug_log=True

def debug_log( msg ):
	if ( do_debug_log ) :
		print( msg, flush=True )



def bound_pool_size ( exe, horizon, base_pool_size, check_lim, approach, overstep, target ):

	best_time = sys.maxsize


	if ( base_pool_size > max_pool_size ):
		debug_log( "Maxed out" )
		pool_size = max_pool_size
	else:
		debug_log( f"Base pool size is {base_pool_size}" )
		pool_size = base_pool_size
	
	best_size = pool_size
	
	debug_log( f"Raw pool_size is {pool_size}" )

	canon_result = perform_run( exe, horizon, int(max_pool_size), RunMode.RUNTIME)
	canon_value = canon_result[0][0]


	low_count=0

	while ( True ) :


		debug_log( f"Trying pool size {pool_size}" )


		try:
			run_results = perform_run( exe, horizon, int(pool_size), RunMode.RUNTIME, dup=check_lim)
		except:
			debug_log( f"Got OOM at size {pool_size}" )
			pool_size *= approach
			return best_size

		time_total = 0.0
		time_min = float('inf')
		
		for check_value, check_time in run_results:

			if ( abs( check_value - canon_value ) > 0.00001 ):
				debug_log( f"Got bad results at size {pool_size}. "
					   f"Check value {check_value} was not close to canon value {canon_value}" )
				pool_size *= approach
				return best_size
			time_min    = min(time_min,check_time)
			time_total += check_time


		time_avg = time_total / check_lim	

		if ( best_time*1.05 >= time_min ):
			best_time = time_min
			debug_log( f"New best time {best_time}" )
			best_size = pool_size
		else :
			low_count += 1
			if ( low_count > overstep ) :
				return best_size
			debug_log( f"Time {time_avg} not as good as best {best_time}" )
		
		debug_log( f"Got okay results at size {pool_size}" )
		
		pool_size /= approach

		if ( pool_size <= 2048 ):
			best_size = 2048
			return best_size


	return best_size




def do_bench ( exe, pool_mult, sweep_range ):
	
	dirname=exe

	targ=f"{superdirname}/{dirname}/{filename}"

	avg_file      = open(f"{targ}_avg","w")
	avg_tune_file = open(f"{targ}_avg_tune","w")

	bst_file      = open(f"{targ}_bst","w")
	bst_tune_file = open(f"{targ}_bst_tune","w")

	mem_file      = open(f"{targ}_mem","w")


	


	min_avg_time=float('inf')
	avg_tune=sys.maxsize
	min_bst_time=float('inf')
	bstTune=sys.maxsize
	size_used=float('inf')





	#ratios = [ (2,32,6), (1.5,32,6), (1.25,32,6), (1.125,32,6) ]

	for hrzn in sweep_range :


		
		try:
			pool_size = perform_run(exe,hrzn,int(max_pool_size),RunMode.MEMORY) * pool_mult
		except ValueError as e:
			raise Exception(
			       f"Could not get config for {targ} to work even at max memory capacity."
			       f" Got output {base_pool_size_str}" ) from e

		ratios = [ (1.5,samp,6), (1.25,samp,6), (1.125,samp,6) ]
		for idx, (ratio,check_count,overstep) in enumerate(ratios):
			pool_size = bound_pool_size(exe,hrzn,pool_size, check_count, ratio, overstep, targ)
			if ( idx < len(ratios) - 1 ):
				pool_size *= ratio

		best_time = sys.maxsize

		while ( True ):
			try:
				debug_log( f"Collecting samples for horizon {hrzn} of {targ}" )
				results = perform_run( exe, hrzn, int(pool_size), RunMode.RUNTIME, dup=samp )
				break
			except:
				pool_size *= 1.1
				if ( pool_size > max_pool_size ):
					sys.exit(1)
				continue
		
		total_time = 0
		for check_value, check_time in results :
			total_time += check_time
			best_time = min( best_time, check_time )

		average_time = total_time / samp

		avg_file.write(f"{hrzn}: {average_time:.6f}\n")
		bst_file.write(f"{hrzn}: {best_time:.6f}\n")

		if ( average_time*1.00 < min_avg_time ) :
			min_avg_time = average_time
			avg_tune=hrzn

		if ( best_time*1.00 < min_bst_time ) :
			min_bst_time = best_time
			bst_tune=hrzn
			size_used=pool_size

	avg_file.write(f"{min_avg_time:.6f}\n")
	avg_tune_file.write(f"{avg_tune}\n")

	bst_file.write(f"{min_bst_time:.6f}\n")
	bst_tune_file.write(f"{bst_tune}\n")
	mem_file.write(f"{int(size_used)}\n")


do_bench( "neut_hrm", 10, [1])

do_bench( "neut_evt", 10, [1, 2, 3, 4, 6, 8, 12, 16, 24, 32])

print( f"{filename} completed" )




