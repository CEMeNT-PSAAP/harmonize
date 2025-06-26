
#include "test_launch.h"
TestModule root("root");

#include "../../harmonize/cpp/harmonize.h"

namespace hrm = harmonize;

namespace test {

#include "type/mod.h"
#include "mem/mod.h"

} // namespace test

int main() {

    root.run();

    return 0;

}

