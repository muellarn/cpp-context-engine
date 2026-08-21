#define IS_POSITIVE(value) ((value) > 0)

namespace cfg_fixture {

struct Tracked {
  explicit Tracked(int value) : value(value) {}
  ~Tracked() {}
  int value;
};

int branching(int value) {
  int result = 0;
  if (IS_POSITIVE(value)) {
    result = 1;
  } else {
    result = -1;
  }
  return result;
}

int choose(int value) {
  switch (value) {
  case 1:
    return 10;
  case 2:
    return 20;
  default:
    return 0;
  }
}

int loop(int limit) {
  int sum = 0;
  for (int index = 0; index < limit; ++index) {
    if (index == 2)
      continue;
    if (index == 8)
      break;
    sum += index;
  }
  return sum;
}

int early_return(int value) {
  if (value < 0)
    return -1;
  return value + 1;
}

int jump(int value) {
  if (value == 0)
    goto done;
  value += 2;
done:
  return value;
}

int exception_flow(int value) {
#if defined(__cpp_exceptions)
  try {
    if (value < 0)
      throw value;
    return value;
  } catch (int error) {
    return -error;
  }
#else
  return value;
#endif
}

int unreachable_after_return() {
  return 1;
  int never = 2;
  return never;
}

int lifetime(int value) {
  Tracked tracked(value);
  if (tracked.value)
    return tracked.value;
  return 0;
}

int local_object_macro_ranges(int value) {
#define LOCAL_CHUNK_SIZE 8
  int remaining = value;
  remaining = value - LOCAL_CHUNK_SIZE;
  value += LOCAL_CHUNK_SIZE;
#undef LOCAL_CHUNK_SIZE
  return remaining + value;
}

} // namespace cfg_fixture
