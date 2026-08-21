#pragma once

#define APPLY_TWICE(value) ((value) + (value))
#define FORWARD_TWICE(value) APPLY_TWICE(value)

namespace analyzer_fixture {

struct Base {
    virtual ~Base() = default;
    virtual int evaluate(int value) const = 0;
};

struct Derived final : Base {
    int evaluate(int value) const override;
};

template <typename T>
T identity(T value) {
    return value;
}

}  // namespace analyzer_fixture
