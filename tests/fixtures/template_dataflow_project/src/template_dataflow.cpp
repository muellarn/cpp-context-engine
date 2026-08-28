namespace template_dataflow_fixture {

template <typename Value> struct basic_format_arg {
  Value value;

  explicit basic_format_arg(Value input) : value(input) {}

  Value get() const { return value; }
};

int instantiate_basic(int value) {
  basic_format_arg<int> argument(value);
  return argument.get();
}

template <typename Value> Value substitute(Value input) {
  Value local = input;
  return local;
}

int instantiate_int(int value) { return substitute(value); }

long instantiate_long(long value) { return substitute(value); }

int increment(int value) { return value + 1; }

int decrement(int value) { return value - 1; }

template <typename Value> Value choose_target(bool up, Value input) {
  auto target = up ? &increment : &decrement;
  return target(static_cast<int>(input));
}

int instantiate_indirect(bool up, int value) { return choose_target(up, value); }

template <typename Context> class visiting_format_arg {
public:
  explicit visiting_format_arg(int value) : value_(value) {}

  template <typename Visitor> auto visit(Visitor&& visitor) const -> decltype(visitor(0)) {
    switch (value_ & 7) {
    case 0: return visitor(value_);
    case 1: return visitor(value_ + 1);
    case 2: return visitor(value_ + 2);
    case 3: return visitor(value_ + 3);
    case 4: return visitor(value_ + 4);
    case 5: return visitor(value_ + 5);
    case 6: return visitor(value_ + 6);
    default: return visitor(value_ + 7);
    }
  }

private:
  int value_;
};

struct context {};
struct add_one {
  int operator()(int value) const { return value + 1; }
};
struct add_two {
  int operator()(int value) const { return value + 2; }
};

int instantiate_visitors(int value) {
  visiting_format_arg<context> argument(value);
  return argument.visit(add_one{}) + argument.visit(add_two{});
}

} // namespace template_dataflow_fixture
