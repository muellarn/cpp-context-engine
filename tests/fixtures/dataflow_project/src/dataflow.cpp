namespace dataflow_fixture {

int target_a(int value) { return value + 1; }
int target_b(int value) { return value + 2; }

struct Pair {
  int first;
  int second;
};

struct Handler {
  int first(int value) { return value + 3; }
  int second(int value) { return value + 4; }
};

struct Deep {
  int value;
  Deep *next;
};

union Choice {
  int integer;
  float real;
};

int definitions_and_join(int input, bool choose) {
  int overwritten = 1;
  overwritten = 2;
  int joined = 0;
  if (choose)
    joined = input;
  else
    joined = overwritten;
  for (int index = 0; index < 2; ++index)
    joined += index;
  return joined;
}

int aliases_and_fields(int input, bool choose) {
  int left = input;
  int right = input + 1;
  int &reference = left;
  int *pointer = &left;
  if (choose)
    pointer = &right;
  *pointer = reference;
  Pair pair{left, right};
  pair.first = *pointer;
  return pair.first;
}

using Function = int (*)(int);

int singleton_pointer(int value) {
  Function selected = &target_a;
  Function copy = selected;
  return copy(value);
}

int conditional_pointer(int value, bool choose) {
  Function selected = choose ? &target_a : &target_b;
  return selected(value);
}

int null_pointer(int value) {
  Function selected = nullptr;
  return selected(value);
}

int unknown_pointer(Function selected, int value) { return selected(value); }

using MemberFunction = int (Handler::*)(int);

int singleton_member(Handler &handler, int value) {
  MemberFunction selected = &Handler::first;
  return (handler.*selected)(value);
}

int conditional_member(Handler &handler, int value, bool choose) {
  MemberFunction selected = choose ? &Handler::first : &Handler::second;
  return (handler.*selected)(value);
}

int conservative_cases(int *pointer, int value) {
  volatile int observed = value;
  auto bytes = reinterpret_cast<unsigned char *>(pointer);
  int result = *(pointer + 1) + *bytes + observed;
  asm volatile("" : "+r"(result));
  return result;
}

int deep_access_path(Deep &root) {
  return root.next->next->next->next->next->value;
}

int union_access(Choice &choice) { return choice.integer; }

#if ALT_BUILD
int build_only_target(int value) { return value + 5; }
#endif

} // namespace dataflow_fixture
