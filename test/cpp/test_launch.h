
#ifndef HARMONIZE_TEST_LAUNCH_GLUE
#define HARMONIZE_TEST_LAUNCH_GLUE


#include <thread>
#include <vector>
#include <iostream>

class TestLaunchConfig {

    size_t cpu_thread_count;
    size_t gpu_block_count;
    size_t gpu_block_size;

    public:

    TestLaunchConfig (size_t cpu_thread_count,size_t gpu_block_count,size_t gpu_block_size);

    size_t get_cpu_thread_count();
    size_t get_gpu_block_count();
    size_t get_gpu_block_size();

};



class TestLaunchResult {

    bool success;
    std::string desc;

    public:

    TestLaunchResult ();
    TestLaunchResult (bool success);
    TestLaunchResult (const char *desc);
    operator bool();
    operator std::string();

};



#define DEFINE_LAUNCH_GLUE(func)                                               \
                                                                               \
template<typename... TYPE_PARAMS>                                              \
struct func ## _launch_glue {                                                  \
                                                                               \
template <typename... ARGS>                                                    \
static __global__                                                              \
void gpu_helper (ARGS... args) {                                               \
    func<TYPE_PARAMS...>(args...);                                             \
}                                                                              \
                                                                               \
static std::string func_name() { return #func ;}                               \
                                                                               \
template <typename... ARGS>                                                    \
static void launch (                                                           \
    TestLaunchConfig config,                                                   \
    ARGS... args                                                               \
) {                                                                            \
    std::vector<std::thread> cpu_thread_team;                                  \
    for (size_t i=0; i<config.get_cpu_thread_count(); i++) {                   \
        cpu_thread_team.emplace_back(func<TYPE_PARAMS...>,args...);            \
    }                                                                          \
    if ((config.get_gpu_block_count()>0) && (config.get_gpu_block_size()>0)) { \
        gpu_helper <<<                                                         \
            config.get_gpu_block_count(),                                      \
            config.get_gpu_block_size()                                        \
        >>>(args...);                                                          \
        util::host::auto_throw(adapt::GPUrtDeviceSynchronize());               \
    }                                                                          \
    for (std::thread& t: cpu_thread_team) {                                    \
        t.join();                                                              \
    }                                                                          \
}                                                                              \
                                                                               \
template <typename... ARGS>                                                    \
static void launch_verbose (                                                   \
    TestLaunchConfig config,                                                   \
    ARGS... args                                                               \
) {                                                                            \
    std::cout << func_name() << std::endl;                                     \
    launch(config,args...);                                                    \
}                                                                              \
                                                                               \
};                                                                             \


class TestFunctionEntry {

    std::string name;
    TestLaunchResult(*func)(TestLaunchConfig);

    public:

    TestFunctionEntry(std::string name,TestLaunchResult(*func)(TestLaunchConfig));
    TestLaunchResult run(TestLaunchConfig config);
    std::string get_name();

};

class TestModule;

class TestLaunchSet {

    TestModule &parent;
    std::string name;
    std::vector<TestFunctionEntry> tests;
    std::vector<TestLaunchConfig> configs;

    public:

    TestLaunchSet (
        TestModule &parent,
        std::string name,
        std::vector<TestFunctionEntry> tests,
        std::vector<TestLaunchConfig> configs
    );

    TestLaunchResult run();

};


class TestModule {

    friend class TestLaunchSet;

    TestModule *parent;
    std::string name;
    std::vector<TestModule*> children;
    std::vector<TestLaunchSet*> test_sets;


    public:


    TestModule(std::string name);
    TestModule(TestModule &parent, std::string name);
    TestLaunchResult run();

    static TestModule& get_root();
    static void run_root();

    std::string full_path();

};

#endif

