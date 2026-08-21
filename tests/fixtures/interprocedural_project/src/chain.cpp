#include "interprocedural.hpp"

namespace interprocedural_fixture {

int top(int input) {
  int reference_output = 0;
  int pointer_output = 0;
  Record record{0};
  return middle(input, reference_output, &pointer_output, record) + record.value;
}

int recursive_even(int value) {
  if (value == 0)
    return 1;
  return recursive_odd(value - 1);
}

int recursive_odd(int value) {
  if (value == 0)
    return 0;
  return recursive_even(value - 1);
}

int call_external(int value) {
  external_sink(&value);
  return value;
}

void virtual_caller(Base &base, int &value) { base.update(value); }

} // namespace interprocedural_fixture
