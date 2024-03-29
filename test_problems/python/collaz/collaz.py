import numpy as np
from numba import cuda, njit
import numba
from numba import config

import sys

sys.path.append("../../..")

import harmonize as harm



mode = "async"

config.DISABLE_JIT = False

val_count = 65536
dev_state_type = numba.from_dtype(np.dtype([ ('val',np.dtype((np.uintp,val_count+1))) ]))
grp_state_type = numba.from_dtype(np.dtype([ ]))
thd_state_type = numba.from_dtype(np.dtype([ ]))

collaz_iter_dtype = np.dtype([('value',np.int64), ('start',np.int64), ('steps',np.int64)])
collaz_iter = numba.from_dtype(collaz_iter_dtype)

@njit
def should_halt(iter: collaz_iter) -> numba.boolean :
    return (iter['value'] <= 1)


def even(prog: numba.uintp, iter: collaz_iter):
    if should_halt(iter):
        device(prog)['val'][1+iter['start']] = iter['steps']
    else:
        iter['steps'] += 1
        iter['value'] /= 2
        if iter['value'] % 2 == 0:
            even_async(prog,iter)
        else :
            odd_async(prog,iter)

def odd(prog: numba.uintp, iter: collaz_iter):
    if iter['value'] <= 1:
        device(prog)['val'][1+iter['start']] = iter['steps']
    else:
        iter['value'] = iter['value'] * 3 + 1
        iter['steps'] += 1
        if iter['value'] % 2 == 0:
            even_async(prog,iter)
        else :
            odd_async(prog,iter)

def initialize(prog: numba.uintp):
    pass

def finalize(prog: numba.uintp):
    pass

def make_work(prog: numba.uintp) -> numba.boolean:
    step_max = 4
    old = numba.cuda.atomic.add(device(prog)['val'],0,step_max)
    if old >= val_count:
        return False

    iter = numba.cuda.local.array(1,collaz_iter)
    step = 0
    while (old+step < val_count) and (step < step_max):
        val = old + step
        iter[0]['value'] = val
        iter[0]['start'] = val
        iter[0]['steps'] = 0
        if val % 2 == 0:
            even_async(prog,iter[0])
        else:
            odd_async(prog,iter[0])
        step += 1

    return True




base_fns   = (initialize,finalize,make_work)
state_spec = (dev_state_type,grp_state_type,thd_state_type)
async_fns  = [odd,even]

device, group, thread = harm.RuntimeSpec.access_fns(state_spec)
odd_async, even_async = harm.RuntimeSpec.async_dispatch(odd,even)

collaz_spec = harm.RuntimeSpec("collaz",state_spec,base_fns,async_fns)


if mode == "async":
    fns = collaz_spec.harmonize_fns()

    alloc_state   = fns["alloc_state"]
    free_state    = fns["free_state"]
    alloc_program = fns["alloc_program"]
    free_program  = fns["free_program"]
    load_state    = fns["load_state"]
    store_state   = fns["store_state"]
    init_program  = fns["init_program"]
    exec_program  = fns["exec_program"]


    @njit
    def exec_fn(state):
        arena_size = 0x100000

        gpu_state  = alloc_state()
        program    = alloc_program(gpu_state,arena_size)

        grid_size  = 4096
        store_state(gpu_state,state)
        init_program(program,grid_size)

        iter_count = 655360
        exec_program(program,grid_size,iter_count)
        load_state(state,gpu_state)

        free_program(program)
        free_state(gpu_state)
else:
    runtime = collaz_spec.event_instance(io_capacity=65536*4,load_margin=1024)
    runtime.init(4096)
    runtime.exec(4,4096)


@njit
def collaz_check(value):
    step = 0
    while value > 1:
        step += 1
        if value % 2 == 0:
            value /= 2
        else :
            value = value * 3 + 1
    return step



@njit
def run():

    state = np.zeros((1,),dev_state_type)[0]
    exec_fn(state)

    print("Finished GPU Pass")

    total = 0
    diff  = 0
    for val in range(val_count):
        steps = collaz_check(val)
        gpu_steps = state['val'][1+val]
        if steps != gpu_steps:
            diff += 1
            print(f"@{val} CPU {steps} different from {gpu_steps}")
        if diff > 10:
            break
        total += steps

    print(f"Number of inconsistent results : at least {diff}")


run()
