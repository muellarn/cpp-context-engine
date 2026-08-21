#include "dispatch.hpp"

namespace dispatch_fixture {

int virtual_call(const Root& value) {
    return value.run();
}

int final_call() {
    FinalLeaf value;
    return value.run();
}

int callable_forms() {
    auto lambda = [](int value) { return value + 1; };
    auto generic = [](auto value) { return value + 2; };
    Functor functor;
    return lambda(1) + generic(2L) + functor(3);
}

int repeated_direct_calls() {
    direct_target(1);
    direct_target(1);
    return 0;
}

int macro_generated_call(int value) {
    return FORWARD_GENERATED(value);
}

long template_calls(long value) {
    return transform<int>(1) + transform<long>(value) + transform<double>(2.0);
}

template double transform<double>(double);

}  // namespace dispatch_fixture
