namespace implicit_fixture {

template <typename Callback>
void for_each(Callback callback) {
    auto dispatch = [callback](int value) { callback(value); };
    dispatch(1);
}

int captured_reference() {
    int result = 0;
    for_each([&](int value) { result += value; });
    return result;
}

}  // namespace implicit_fixture
