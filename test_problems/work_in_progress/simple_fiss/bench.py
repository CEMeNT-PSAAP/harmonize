#!/usr/bin/python3

import subprocess


#fiss_std size=20 res=0.1 cx=0.01 sx=0.99 num=10000 span=32 hrzn=256 flux=false show=true

#make WG_COUNT=240 STD_WG_COUNT=1024

make_cmd = [ 	'make',     'hard',           'WG_COUNT=240',      'STD_WG_COUNT=1024', 
		'NEU_3D=X', 'QUEUE_WIDTH=64', 'ARENA_SIZE=0xFFFFF' ]

#make_cmd = [ 	'make',     'hard',           'WG_COUNT=240',      'STD_WG_COUNT=1024', 
#		'NEU_3D=X', 'QUEUE_WIDTH=64', 'ARENA_SIZE=0xFFFFF', 'INDIRECT=X' ]

params={}

params["show"]="false"
#params["flux"]="false"
params["flux"]="false"
params["span"]=32


particle_start = 10

if params["flux"] == "true" :
	particle_lim = 19
else:
	particle_lim = 24

ratio_start = 0
ratio_lim   = 9#12

time_start = 9
time_lim   = 9

scaled_res = 0.001
sample_count = 4


csvmode=True


def build_command(name,par):
	result = [name]
	for arg, val in par.items():
		result.append(str(arg)+"="+str(val))
	return result


def bench (name,par):
	cmd = build_command(name,par)
	result = subprocess.run(cmd, stdout=subprocess.PIPE)	
	return float(result.stdout.decode('utf-8'))


def sweet_spot(name,par):

	if name == "fiss_std":
		return 1

	mom=0
	old = bench(name,par)
	while True:
		#print("Time: "+str(old)+"\thrzn: "+str(par["hrzn"]))
		if mom == 0:
			par["hrzn"] += 1
			high = bench(name,par)
			if high < old:
				mom = 1
				old = high
				continue
			par["hrzn"] -= 2
			low  = bench(name,par)
			if low < old:
				old = low
				mom = -1
				continue
			par["hrzn"] += 1
			return
				
		elif mom > 0:
			par["hrzn"] += 1
			high = bench(name,par)
			if high > old:
				par["hrzn"] -= 1
				return
			old = high
		else:	
			par["hrzn"] -= 1
			low = bench(name,par)
			if low > old:
				par["hrzn"] += 1
				return
			old = low


def perf(name,par):
	sweet_spot(name,par)
	#print("Sweet spot: "+str(par["hrzn"]))
	result = 0
	for _ in range(sample_count):
		result += bench(name,par)
	result /= sample_count
	return result



subprocess.run(make_cmd, stdout=subprocess.PIPE)


def run_bench(name):

	spots = []

	for ratio_mag in range(ratio_start,ratio_lim+1):
		ratio = 2**ratio_mag
		if csvmode:
			print(","+str(ratio),end='')
		else:
			print("\t"+"{0:12.0f}".format(ratio),end='')
	print("")

	last = 32
	for t_mag in range(time_start,time_lim+1):
		time = 2**t_mag
		for p_mag in range(particle_start,particle_lim+1):
			params["hrzn"] = last
			num = int(2**p_mag)
			print(num,end='')
			spots.append([])
			for ratio_mag in range(ratio_start,ratio_lim+1):
				spread_bound    = 4*(1.41**ratio_mag)
				params["time"]  = time
				params["size"]  = min(spread_bound,time)
				params["res"]   = params["size"]*scaled_res
				ratio           = 2**ratio_mag
				params["cx"]    = 1/ratio
				params["sx"]    = 1-params["cx"]
				params["num"]   = num
				std_val = perf(name,params)
				if ratio_mag == ratio_start:
					last = params["hrzn"]
				if csvmode:
					print(","+str(std_val),end='')
				else:
					print("\t"+"{0:12.6f}".format(std_val),end='')
				spots[-1].append(params["hrzn"])
			print("")
		

	print("\n\nTUNED HORIZONS")	
	for row in spots:
		for elem in row:
			print(str(elem)+",",end="")	

		print("")

#print("STANDARD:")
#run_bench("fiss_std")
print("HARMONIZED:")
run_bench("fiss_hrm")




