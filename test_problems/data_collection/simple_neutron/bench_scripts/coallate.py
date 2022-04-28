#!/bin/python3

import os
import sys
import glob
from enum import Enum
from subprocess import Popen, check_call, check_output, STDOUT, PIPE
from tempfile import NamedTemporaryFile


argc = len(sys.argv)

if ( argc < 3 or argc > 4 ):
	print("Invalid argument count. 2 or 3 arguments required: <program> <info> [wlim]")
	sys.exit(1)

program = sys.argv[1]
info    = sys.argv[2]
suffix  = ""

if ( argc >= 4 ):
	suffix = f"_imp_{sys.argv[3]}"

n_range = [ 1000, 10000, 100000, 1000000 ]
t_range = [ 2, 4, 6, 8  ]

for t in t_range :
	for n in n_range:
		print(f"{t}s{n}n",end=' ')
		for int_f in range (0, 11):
			f = int_f / 10.0
			print(f"{f:.1f}",end="")
			if( int_f != 10 or n != n_range[-1]):
				print(' ',end="")
	print("")
	for int_c in range(10, -1, -1):
		c = int_c / 10.0
		f_lim = 10 - int_c
		for n in n_range :
			print(f"{c:.1f}",end=' ')
			for int_f in range(0, f_lim+1):
				f = int_f/10.0
				s = abs(1.0-f-c)
				file_name = f"bench_s{t}_n{n}{suffix}/{program}/c{c:.1f}s{s:.1f}f{f:.1f}_{info}"
				#print(file_name)
				try:
					file_handle = open(file_name)
					value = file_handle.read().split('\n')[-2]
					file_handle.close()
				except:
					value = "nan"
				print(value,end="")
				if( int_f != 10 or n != n_range[-1]):
					print(' ',end="")


			for int_f in range(f_lim+1,11):
				print(' ',end="")
		print("")


