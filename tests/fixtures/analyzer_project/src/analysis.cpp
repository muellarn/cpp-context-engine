#include "analysis.hpp"

namespace analyzer_fixture {

int Derived::evaluate(int value) const {
    auto lambda = [factor = 2](int input) { return input * factor; };
    auto doubled = APPLY_TWICE(value);
    return identity<int>(lambda(FORWARD_TWICE(value))) + doubled;
}

int construct() {
    Derived derived;
    return derived.evaluate(1);
}

template int identity<int>(int);

}  // namespace analyzer_fixture
