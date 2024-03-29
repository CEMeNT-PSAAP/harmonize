
# String template for the program initialization wrapper
init_template = """
extern "C"
void init_program(
    void   *instance_ptr,
    size_t  grid_size
) {{
    auto instance = (typename {short_name}::Instance*) instance_ptr;
    //printf("\\n\\nINIT\\n\\n");
    init<{short_name}>(*instance,grid_size);
    util::host::auto_throw(cudaDeviceSynchronize());
}}
"""

# String template for the program execution wrapper
exec_template = """
extern "C"
void exec_program(
    void   *instance_ptr,
    size_t  grid_size,
	size_t  cycle_count
) {{
    auto instance = (typename {short_name}::Instance*) instance_ptr;
    // printf("<ctx%p>",_dev_ctx);
    // printf("<sta%p>",device);
    //printf("\\n\\nEXEC\\n\\n");
    exec<{short_name}>(*instance,grid_size,cycle_count);
    util::host::auto_throw(cudaDeviceSynchronize());
}}
"""


alloc_event_prog_template = """
extern "C"
void *alloc_program(void* device_arg, size_t io_size) {{
    auto  state  = (typename {short_name}::DeviceState) device_arg;
    void *result = new {short_name}::Instance(io_size,state);
    return result;
}}
"""

alloc_harm_prog_template = """
extern "C"
void *alloc_program(void* device_arg, size_t arena_size) {{
	auto  state  = (typename {short_name}::DeviceState) device_arg;
	void *result = new {short_name}::Instance(arena_size,state);
	return result;
}}
"""

free_prog_template = """
extern "C"
void free_program(void* instance_ptr) {{
	auto instance  = (typename {short_name}::Instance*) instance_ptr;
	delete instance;
}}
"""

alloc_state_template = """
extern "C"
void *alloc_state() {{
	void *result = nullptr;
	util::host::auto_throw(cudaMalloc(&result,sizeof({state_struct})));
	return result;
}}
"""

load_state_template = """
extern "C"
void load_state(void *host_ptr, void *dev_ptr) {{
    util::host::auto_throw(cudaMemcpy (host_ptr,dev_ptr,sizeof({state_struct}),cudaMemcpyDeviceToHost));
    //size_t *data = (size_t*) host_ptr;
    //for(int i=0; i<10; i++) {{
    //    printf("%d,",data[i]);
    //}}
    //printf("\\n");
}}
"""

store_state_template = """
extern "C"
void store_state(void *dev_ptr, void *host_ptr) {{
    //size_t *data = (size_t*) host_ptr;
    //for(int i=0; i<10; i++) {{
    //    printf("%d,",data[i]);
    //}}
    //printf("\\n");
    util::host::auto_throw(cudaMemcpy (dev_ptr,host_ptr,sizeof({state_struct}),cudaMemcpyHostToDevice));
}}
"""

free_state_template = """
extern "C"
void free_state(void *state_ptr) {{
	util::host::auto_throw(cudaFree(state_ptr));
}}
"""

# String template for async function dispatches
dispatch_template = """
extern "C" __device__
int dispatch_{fn}_{kind}(void*{params}){{
	(({short_name}*)fn_param_1)->template {kind}<{fn_type}>({args});
    //printf("{{ {fn} wrapper }}");
	return 0;
}}
"""

# String template for the program execution wrapper
fn_query_template = """
extern "C" __device__
int query_{field}(void *result, void *prog){{
	(*({kind}*)result) = {prefix}(({short_name}*)prog)->template {field}<{fn_type}>();
	return 0;
}}
"""

# String template for the program execution wrapper
query_template = """
extern "C" __device__
int query_{field}(void *result, void *prog){{
	(*({kind}*)result) = {prefix}(({short_name}*)prog)->{field}();
	return 0;
}}
"""


# String template for field accessors
accessor_template = """
extern "C" __device__
int access_{field}(void* result, void* prog){{
	(*(void**)result) = {prefix}(({short_name}*)prog)->{field};
    // printf("{{ {field} accessor }}");
    // printf("{{prog %p}}",prog);
    // printf("{{field%p}}",*(void**)result);
	return 0;
}}
"""



