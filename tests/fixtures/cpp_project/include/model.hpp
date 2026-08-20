#pragma once

#define SCALE_VALUE(value) ((value) * 2)

namespace demo {

enum class Kind { primary, secondary };

struct Base {
  virtual ~Base() = default;
  virtual int compute(int value) const = 0;
};

class Derived final : public Base {
 public:
  int compute(int value) const override;
};

int helper(int value);

}  // namespace demo
