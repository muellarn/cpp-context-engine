#include "analysis.hpp"

namespace analyzer_fixture {

int Derived::evaluate(int value) const {
    auto lambda = [factor = 2](int input) { return input * factor; };
    return identity<int>(lambda(FORWARD_TWICE(value)));
}

template int identity<int>(int);

}  // namespace analyzer_fixture
