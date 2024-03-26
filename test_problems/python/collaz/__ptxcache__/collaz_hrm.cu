#include "collaz.cu"
typedef HarmonizeProgram<collaz> collaz_hrm;
extern "C"
void init_program(
	void *_dev_ctx_arg,
	void *device_arg,
	int   grid_size,
	int   block_size
) {
	auto _dev_ctx = (typename collaz_hrm::DeviceContext*) _dev_ctx_arg;
	auto device   = (typename collaz_hrm::DeviceState) device_arg;
	_dev_init<collaz_hrm><<<grid_size,block_size>>>(*_dev_ctx,device);
}
extern "C"
void exec_program(
	void   *_dev_ctx_arg,
	void   *device_arg,
	size_t  cycle_count,
	int     grid_size,
	int     block_size
) {
	auto _dev_ctx = (typename collaz_hrm::DeviceContext*) _dev_ctx_arg;
	auto device   = (typename collaz_hrm::DeviceState) device_arg;
	_dev_exec<collaz_hrm><<<grid_size,block_size>>>(*_dev_ctx,device,cycle_count);
}
extern "C" __device__ 
int dispatch_odd_async(void*, void* fn_param_1, void* fn_param_2){
	((collaz_hrm*)fn_param_1)->template async<Odd>(*(_24b8*)fn_param_2);
	return 0;
}
extern "C" __device__ 
int dispatch_odd_sync(void*, void* fn_param_1, void* fn_param_2){
	((collaz_hrm*)fn_param_1)->template sync<Odd>(*(_24b8*)fn_param_2);
	return 0;
}
extern "C" __device__ 
int dispatch_even_async(void*, void* fn_param_1, void* fn_param_2){
	((collaz_hrm*)fn_param_1)->template async<Even>(*(_24b8*)fn_param_2);
	return 0;
}
extern "C" __device__ 
int dispatch_even_sync(void*, void* fn_param_1, void* fn_param_2){
	((collaz_hrm*)fn_param_1)->template sync<Even>(*(_24b8*)fn_param_2);
	return 0;
}
extern "C" __device__ 
int access_device(void* result, void* prog){
	(*(void**)result) = ((collaz_hrm*)prog)->device;
	return 0;
}
extern "C" __device__ 
int access_group(void* result, void* prog){
	(*(void**)result) = ((collaz_hrm*)prog)->group;
	return 0;
}
extern "C" __device__ 
int access_thread(void* result, void* prog){
	(*(void**)result) = ((collaz_hrm*)prog)->thread;
	return 0;
}
