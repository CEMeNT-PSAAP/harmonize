


#ifndef HARMONIZE_PROGRAM
#define HARMONIZE_PROGRAM

// The set of available program types
#include "async.h"
#include "event.h"



/*
// These functions are here just to trampoline into the actual main functions for a given program.
// This additonal layer of calls is present to allow for linking to these calls as device functions.
*/
template<typename ProgType>
__device__ void _inner_dev_init(typename ProgType::DeviceContext& _dev_ctx, typename ProgType::DeviceState& device) {


	__shared__ typename ProgType::GroupContext _grp_ctx;
	__shared__ typename ProgType::GroupState   group;


	typename ProgType::ThreadContext _thd_ctx;
	typename ProgType::ThreadState   thread;

	ProgType prog(_dev_ctx,_grp_ctx,_thd_ctx,device,group,thread);

	prog.init_program();

}


template<typename ProgType>
__device__ void _inner_dev_exec(typename ProgType::DeviceContext& _dev_ctx, typename ProgType::DeviceState& device, size_t cycle_count) {

	__shared__ typename ProgType::GroupContext _grp_ctx;
	__shared__ typename ProgType::GroupState   group;

	typename ProgType::ThreadContext _thd_ctx;
	typename ProgType::ThreadState   thread;

	ProgType prog(_dev_ctx,_grp_ctx,_thd_ctx,device,group,thread);

	//printf("(ctx%p)",&prog._dev_ctx);
	//printf("(sta%p)",&prog.device);
	//printf("(pre%p)",((void**)&prog.device)[-1]);
	//printf("(gtx%p)",(&prog._grp_ctx));
	prog.exec(cycle_count);
}



/*
// These functions are here just to trampoline into the device trampoline functions for a given program.
// This is done because structs/classes may not have global member functions.
*/
template<typename ProgType>
__global__ void _dev_init(typename ProgType::DeviceContext _dev_ctx, typename ProgType::DeviceState device) {
	_inner_dev_init<ProgType>(_dev_ctx, device);
}
template<typename ProgType>
__global__ void _dev_exec(typename ProgType::DeviceContext _dev_ctx, typename ProgType::DeviceState device, size_t cycle_count) {
	_inner_dev_exec<ProgType>(_dev_ctx, device,cycle_count);
}



//! These functions unwrap an instance into its device context and passes it to the responsible
//! kernel.
template<typename ProgType>
__host__ void init(typename ProgType::Instance& instance,size_t group_count) {
	_dev_init<ProgType><<<group_count,ProgType::WORK_GROUP_SIZE>>>(instance.to_context(),instance.device_state);
}
template<typename ProgType>
__host__ void exec(typename ProgType::Instance& instance,size_t group_count, size_t cycle_count) {
	_dev_exec<ProgType><<<group_count,ProgType::WORK_GROUP_SIZE>>>(instance.to_context(),instance.device_state,cycle_count);
}


#endif

