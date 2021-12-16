#!/usr/bin/python3


import subprocess


on_DGX=True

make_cmd = [ 'make' ]

params={}


if(on_DGX):
	# FOR DGX
	params["wg_count"] = 480 
else:
	# FOR RABBIT
	params["wg_count"] = 240 



params["span"]=32


particle_start = 23
particle_lim = 23

ratio_start =  0
ratio_lim   = 12

#ratio_start =  5
#ratio_lim   =  8

time_start = 0
time_lim   = 10

scaled_res = 0.001
sample_count = 3
spot_sample_count = 3

csvmode=True


def build_command(name,par):
	result = [name]
	for arg, val in par.items():
		result.append("-"+str(arg))
		result.append(str(val))
	return result


def bench_sample (name,par):
	cmd = build_command(name,par)
	result = subprocess.run(cmd, stdout=subprocess.PIPE)	
	return float(result.stdout.decode('utf-8'))


def bench (name,par,count):
	cmd = build_command(name,par)
	result = 0
	for i in range(count+1):
		samp = subprocess.run(cmd, stdout=subprocess.PIPE)
		if i == 0:
			continue # throw out first result, which is usually bad due to lazy compilation
		
		result += float(samp.stdout.decode('utf-8'))
	return (result/count)



def sweet_spot(name,par):

	if name == "neut_std":
		return 1

	mom=0
	old = bench(name,par,spot_sample_count)
	best_hrzn = par["hrzn"]
	best_time = old


	lookabout=3
	changed = True

	tries={}
	
	while changed:

		new_best=best_hrzn
		new_time=best_time

		min_look=0

		for i in range(lookabout):
			try_hrzn = int(best_hrzn - 2**i)
			if try_hrzn <= 0:
				continue
			if try_hrzn in tries:
				continue
			tries[try_hrzn] = True
			par["hrzn"] = try_hrzn
			try_time = bench(name,par,spot_sample_count)
			#print("Found time "+str(try_time)+" for horizon "+str(try_hrzn))
			if try_time < new_time:
				new_best = try_hrzn
				new_time = try_time
				if i > min_look:
					min_look = i


		for i in range(lookabout):
			try_hrzn = int(best_hrzn + 2**i)
			if try_hrzn <= 0:
				continue
			if try_hrzn in tries:
				continue
			tries[try_hrzn] = True
			par["hrzn"] = try_hrzn
			try_time = bench(name,par,spot_sample_count)
			#print("Found time "+str(try_time)+" for horizon "+str(try_hrzn))
			if try_time < new_time:
				new_best = try_hrzn
				new_time = try_time
				if i > min_look:
					min_look = i

		if lookabout == (min_look+1):
			lookabout += 1
		else:
			lookabout = min_look+1

		changed = (best_hrzn != new_best)

		best_hrzn = new_best
		best_time = new_time

	par["hrzn"] = best_hrzn
	#print("Found horizon "+str(best_hrzn))



def perf(name,par):
	sweet_spot(name,par)
	#print("Sweet spot: "+str(par["hrzn"]))
	return bench(name,par,sample_count)



subprocess.run(make_cmd, stdout=subprocess.PIPE)


def run_bench(name):

	spots = []

	for ratio_mag in range(ratio_start,ratio_lim+1):
		ratio = 2**ratio_mag
		if csvmode:
			print(","+str(ratio),end='',flush=True)
		else:
			print("\t"+"{0:12.0f}".format(ratio),end='')
	print("")

	last = 1
	for t_mag in range(time_start,time_lim+1):
		time = 2**t_mag
		for p_mag in range(particle_start,particle_lim+1):
			params["hrzn"] = last
			num = int(2**p_mag)
			print(num,end='')
			spots.append([])
			for ratio_mag in range(ratio_start,ratio_lim+1):
				spread_bound    = time #4*(1.41**ratio_mag)
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
					print(","+str(std_val),end='',flush=True)
				else:
					print("\t"+"{0:12.6f}".format(std_val),end='')
				spots[-1].append(params["hrzn"])
			print("")
		

	print("\n\nTUNED HORIZONS")	
	for row in spots:
		for elem in row:
			print(str(elem)+",",end="")	

		print("")

print("STANDARD:")
run_bench("neut_std")
#print("HARMONIZED:")
#run_bench("neut_hrm")




