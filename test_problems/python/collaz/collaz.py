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


state = np.zeros((0,),dev_state_type)


if mode == "async":
    fns = collaz_spec.harmonize_fns()
    arena_size = 0x1000000
    pool_size  = 8191
    stack_size = 8191

    context    = fns["alloc_context"](arena_size,pool_size,stack_size)
    gpu_state  = fns["alloc_state"]()

    grid_size  = 4095
    block_size = 32
    fns["load_state"](gpu_state,state)
    fns["init_program"](context,state,grid_size,block_size)

    iter_count = 65536
    fns["exec_program"](context,state,iter_count,grid_size,block_size)
    fns["load_state"](state,gpu_state)
    fns["rt_free"](context)
    fns["rt_free"](state)
else:
    runtime = collaz_spec.event_instance(io_capacity=65536*4,load_margin=1024)
    runtime.init(4096)
    runtime.exec(4,4096)


state = runtime.load_state()[0]
print("Finished GPU Pass")

def collaz_check(value):
    step = 0
    while value > 1:
        step += 1
        if value % 2 == 0:
            value /= 2
        else :
            value = value * 3 + 1
    return step

total = 0
diff  = 0
for val in range(val_count):
    steps = collaz_check(val)
    gpu_steps = state['val'][1+val]
    if steps != gpu_steps:
        diff += 1
        print(f"@{val} CPU {steps} different from {gpu_steps}")
    total += steps

print(f"Number of inconsistent results : {diff}")

