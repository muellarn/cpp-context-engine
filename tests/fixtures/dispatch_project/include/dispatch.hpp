#pragma once

#define GENERATED_CALL(value) direct_target(value)
#define FORWARD_GENERATED(value) GENERATED_CALL(value)

namespace dispatch_fixture {

inline int direct_target(int value) { return value + 1; }

struct Root {
    virtual ~Root() = default;
    virtual int run() const { return 0; }
};

struct OverrideA : virtual Root {
    int run() const override { return 1; }
};

struct OverrideB : virtual Root {
    int run() const override { return 2; }
};

struct Auxiliary {
    virtual ~Auxiliary() = default;
    virtual int auxiliary() const { return 3; }
};

struct Multi final : OverrideA, Auxiliary {
    int run() const override { return 4; }
};

struct FinalLeaf final : Root {
    int run() const override final { return 5; }
};

#ifdef EXTRA_OVERRIDE
struct BuildOnlyOverride final : Root {
    int run() const override { return 6; }
};
#endif

struct Functor {
    int operator()(int value) const { return value * 2; }
};

template <typename T>
T transform(T value) {
    return value;
}

template <>
inline int transform<int>(int value) {
    return value + 10;
}

template <typename T>
int dependent_uninstantiated(T& value) {
    return value.missing();
}

int virtual_call(const Root& value);
int final_call();
int callable_forms();
int repeated_direct_calls();
int macro_generated_call(int value);
long template_calls(long value);

}  // namespace dispatch_fixture
