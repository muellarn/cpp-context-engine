#include "model.hpp"

namespace demo {

int helper(int value) { return value + 1; }

int Derived::compute(int value) const { return SCALE_VALUE(helper(value)); }

}  // namespace demo
