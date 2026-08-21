#include "interprocedural.hpp"

namespace interprocedural_fixture {

int global_value = 0;

void leaf(int input, int &reference_output, int *pointer_output, Record &record) {
  reference_output = input;
  *pointer_output = input;
  record.value = input;
#if ALT_BUILD
  global_value = input + 1;
#else
  global_value = input;
#endif
}

int middle(int input, int &reference_output, int *pointer_output, Record &record) {
  leaf(input, reference_output, pointer_output, record);
  return reference_output;
}

void Base::update(int &value) { value += 1; }
void Derived::update(int &value) { value += 2; }

} // namespace interprocedural_fixture
