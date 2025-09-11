
# String template for the program initialization wrapper
init_template = """
extern "C"
void init_program_{suffix}(
    void   *instance_ptr,
    size_t  grid_size
) {{
    auto instance = (typename {short_name}::Instance*) instance_ptr;
    //printf("\\n\\nINIT (instance: %p)\\n\\n",instance_ptr);
    init<{short_name}>(*instance,grid_size);
    util::host::auto_throw(adapt::GPUrtDeviceSynchronize());
}}
"""

# String template for the program execution wrapper
exec_template = """
extern "C"
void exec_program_{suffix}(
    void   *instance_ptr,
    size_t  grid_size,
	size_t  cycle_count
) {{
    auto instance = (typename {short_name}::Instance*) instance_ptr;
    //printf("\\n\\nEXEC (instance: %p)\\n\\n",instance_ptr);
    exec<{short_name}>(*instance,grid_size,cycle_count);
    util::host::auto_throw(adapt::GPUrtDeviceSynchronize());
}}
"""


alloc_event_prog_template = """
extern "C"
void *alloc_program_{suffix}(void* device_arg, size_t io_size) {{
    //printf("allocating event program instance\\n");
    auto  state  = (typename {short_name}::DeviceState) device_arg;
    void *result = new {short_name}::Instance(io_size,state);
    //printf("allocated event program instance:%p\\n",result);
    return result;
}}
"""

alloc_harm_prog_template = """
extern "C"
void *alloc_program_{suffix}(void* device_arg, size_t arena_size) {{
	//printf("allocting async program instance\\n");
	auto  state  = (typename {short_name}::DeviceState) device_arg;
	void *result = new {short_name}::Instance(arena_size,state);
	//printf("allocted async program instance:%p\\n",result);
	return result;
}}
"""

free_prog_template = """
extern "C"
void free_program_{suffix}(void* instance_ptr) {{
	auto instance  = (typename {short_name}::Instance*) instance_ptr;
	delete instance;
}}
"""

alloc_state_template = """
extern "C"
void *alloc_state_{suffix}() {{
	void *result = nullptr;
	//printf("allocting async program instance with size:%ld\\n",sizeof({state_struct}));
	util::host::auto_throw(adapt::GPUrtMalloc(&result,sizeof({state_struct})));
	//printf("allocated gpu_state:%p\\n",result);
	return result;
}}
"""

load_state_template = """
extern "C"
void load_state_{label}_{suffix}(void *host_ptr, void *dev_ptr) {{
    //printf("cpu_state:%p\\n",host_ptr);
    //printf("gpu_state:%p\\n", dev_ptr);
    void *offset_dev_ptr = (void*)(((char*)dev_ptr)+{offset});
    util::host::auto_throw(adapt::GPUrtMemcpy(
        host_ptr,
        offset_dev_ptr,
        {size},
        adapt::GPUrtMemcpyDeviceToHost
    ));
    //size_t *data = (size_t*) host_ptr;
    //for(int i=0; i<10; i++) {{
    //    //printf("%zu,",data[i]);
    //}}
    //printf("\\n");
}}
"""

store_state_template = """
extern "C"
void store_state_{label}_{suffix}(void *dev_ptr, void *host_ptr) {{
    //printf("cpu_state:%p\\n",host_ptr);
    //printf("gpu_state:%p\\n", dev_ptr);
    void *offset_dev_ptr = (void*)(((char*)dev_ptr)+{offset});
    //size_t *data = (size_t*) host_ptr;
    //for(int i=0; i<10; i++) {{
    //    //printf("%zu,",data[i]);
    //}}
    //printf("\\n");
    util::host::auto_throw(adapt::GPUrtMemcpy(
        offset_dev_ptr,
        host_ptr,
        {size},
        adapt::GPUrtMemcpyHostToDevice
    ));
}}
"""

free_state_template = """
extern "C"
void free_state_{suffix}(void *state_ptr) {{
	util::host::auto_throw(adapt::GPUrtFree(state_ptr));
}}
"""

complete_template = """
extern "C"
int complete_{suffix}(void *instance_ptr) {{
	auto instance  = (typename {short_name}::Instance*) instance_ptr;
	bool result = instance->complete();
	return result;
}}
"""

clear_flags_template = """
extern "C"
void clear_flags_{suffix}(void *instance_ptr) {{
	auto instance  = (typename {short_name}::Instance*) instance_ptr;
	instance->clear_flags();
}}
"""

set_device_template = """
extern "C"
void set_device_{suffix}(int device) {{
    util::host::auto_throw(adapt::GPUrtSetDevice(device));
}}
"""

# String template for async function dispatches
dispatch_template = """
extern "C" __device__
int dispatch_{fn}_{kind}_{suffix}(void*fn_param_0{params}){{
	(({short_name}*)fn_param_1)->template {kind}<{fn_type}>({args});
	//printf("{{ {fn} wrapper }}");
	return 0;
}}
"""

# String template for the program execution wrapper
fn_query_template = """
extern "C" __device__
int query_{field}_{suffix}(void *result, void *prog){{
	(*({kind}*)result) = {prefix}(({short_name}*)prog)->template {field}<{fn_type}>();
	return 0;
}}
"""

# String template for the program execution wrapper
query_template = """
extern "C" __device__
int query_{field}_{suffix}(void *result, void *prog){{
	(*({kind}*)result) = {prefix}(({short_name}*)prog)->{field}();
	return 0;
}}
"""

# String template for field accessors
accessor_template = """
extern "C" __device__
int access_{label}_{suffix}(void* result, void* prog){{
    void*& adr = *(void**)result;
	adr = {prefix}(({short_name}*)prog)->{field};
    adr = (void*)(((char*)adr)+{offset});
    if(threadIdx.x == 0) {{
        //printf("{{ {label} accessor }}");
        //printf("{{prog %p}}",prog);
        //printf("{{field%p}}",adr);
    }}
	return 0;
}}
"""

atomic_template="""
extern "C" __device__
{face_type_cpp} array_atomic_{op_py}_{face_type_py}(void* ptr, size_t index, {face_type_cpp} value) {{
    //printf("{{atomic add at %p}}",ptr);
    volatile bool always_true = true;
    if (always_true) {{
        {real_type_cpp} *array = ({real_type_cpp} *) ptr;
        return {op_cpp}(array+index,({real_type_cpp}) value);
    }} else {{
        return 0;
    }}
}}
"""


clock_template="""
extern "C" __device__
long long int get_wall_clock() {{
    #ifdef HARMONIZE_PLATFORM_CUDA
    long long int result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
    #else
    return wall_clock64();
    #endif
}}

extern "C"
long long int wall_clock_rate() {{
    #ifdef HARMONIZE_PLATFORM_CUDA
    return 1000000000;
    #else
    int device_id;
    util::host::auto_throw(hipGetDevice(&device_id));
    int wall_clock_rate = 0; //in kilohertz
    util::host::auto_throw(hipDeviceGetAttribute(
        &wall_clock_rate,
        hipDeviceAttributeWallClockRate,
        device_id
    ));
    return wall_clock_rate * 1000;
    #endif
}}
"""


early_halt_template="""
extern "C" __device__
int halt_early_{suffix}(void* result, void *prog) {{
	(({short_name}*)prog)->halt_early();
    return 0;
}}
"""

print_template="""
extern "C" __device__
int harmonize_print_{type_sig}(void* result, {args}) {{
    printf("( {format_str} )\\n",{arg_vals});
    return 0;
}}
"""

