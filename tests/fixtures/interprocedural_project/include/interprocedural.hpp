#pragma once

namespace interprocedural_fixture {

struct Record {
  int value;
};

void leaf(int input, int &reference_output, int *pointer_output, Record &record);
int middle(int input, int &reference_output, int *pointer_output, Record &record);
int top(int input);
int recursive_even(int value);
int recursive_odd(int value);
int call_external(int value);
int unrelated(int value);

void external_sink(int *value);

struct Base {
  virtual ~Base() = default;
  virtual void update(int &value);
};

struct Derived final : Base {
  void update(int &value) override;
};

void virtual_caller(Base &base, int &value);

} // namespace interprocedural_fixture
