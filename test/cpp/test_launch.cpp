#include "test_launch.h"


TestLaunchConfig::TestLaunchConfig (
    size_t cpu_thread_count,
    size_t gpu_block_count,
    size_t gpu_block_size
)
    : cpu_thread_count(cpu_thread_count)
    , gpu_block_count(gpu_block_count)
    , gpu_block_size(gpu_block_size)
{}

size_t TestLaunchConfig::get_cpu_thread_count() {
    return cpu_thread_count;
}

size_t TestLaunchConfig::get_gpu_block_count() {
    return gpu_block_count;
}

size_t TestLaunchConfig::get_gpu_block_size() {
    return gpu_block_size;
}





TestLaunchResult::TestLaunchResult ()
    : success(false)
    , desc("Unspecified error.")
{}

TestLaunchResult::TestLaunchResult (bool success)
    : success(success)
    , desc(success ? "Success" : "Unspecified error.")
{}

TestLaunchResult::TestLaunchResult (const char *desc)
    : success(false)
    , desc(desc)
{}

TestLaunchResult::operator bool() { return success; }
TestLaunchResult::operator std::string() { return desc; }




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
};                                                                             \



TestFunctionEntry::TestFunctionEntry(
    std::string name,
    TestLaunchResult(*func)(TestLaunchConfig)
)
    : name(name)
    , func(func)
{}

TestLaunchResult TestFunctionEntry::run(TestLaunchConfig config) {
    return func(config);
}

std::string TestFunctionEntry::get_name() {
    return name;
}



TestLaunchSet::TestLaunchSet (
    TestModule &parent,
    std::string name,
    std::vector<TestFunctionEntry> tests,
    std::vector<TestLaunchConfig> configs
)
    : parent(parent)
    , name(name)
    , tests(tests)
    , configs(configs)
{
    parent.test_sets.push_back(this);
}

TestLaunchResult TestLaunchSet::run() {
    for (TestFunctionEntry& test : tests) {
        for (TestLaunchConfig& config : configs) {
            std::cout << "[ ] Running '" << name << "' "
                    << "with config {"
                    << "cpu=" << config.get_cpu_thread_count() << ", "
                    << "gpu=(" << config.get_gpu_block_count() << "," << config.get_gpu_block_size() << ")"
                    << "}";
            std::cout.flush();
            TestLaunchResult result = test.run(config);
            if (!result) {
                std::cout << std::endl << "Launch encountered error: '" << result << "'" << std::endl;
                return result;
            } else {
                std::cout << "\r[X" << std::endl;
            }
        }
    }
    return TestLaunchResult(true);
}


std::string TestModule::full_path() {
    std::string result = name;
    TestModule *iter = parent;
    while(iter) {
        result = iter->name + "." + result;
        iter = iter->parent;
    }
    return result;
}

TestModule::TestModule(std::string name)
    : name(name)
    , parent(nullptr)
{}

TestModule::TestModule(TestModule &parent, std::string name)
    : name(name)
    , parent(&parent)
{
    if (this != &parent) {
        parent.children.push_back(this);
    }
}

TestLaunchResult TestModule::run() {

    std::cout << "Running TestModule '" << full_path() << "'" << std::endl;

    for (TestLaunchSet *test_set : test_sets) {
        TestLaunchResult result = test_set->run();
        if (!result) {
            return result;
        }
    }

    for (TestModule *child : children) {
        TestLaunchResult result = child->run();
        if (!result) {
            return result;
        }
    }

    return TestLaunchResult(true);
}


