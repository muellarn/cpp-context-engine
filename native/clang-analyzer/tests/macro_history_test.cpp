#include "macro_history.h"

#include <stdexcept>

namespace {
void require(bool condition) {
  if (!condition)
    throw std::runtime_error("macro history assertion failed");
}
}

int main() {
  cpp_context::MacroHistoryIndex index;
  const auto a = std::filesystem::path("/project/a.hpp");
  const auto b = std::filesystem::path("/generated/b.hpp");
  index.add(10, 5, a, "outer", 0);
  index.add(11, 5, a, "inner", 1);
  // A newer fallback must never replace an older exact match.
  require(index.exact(10) == 0);
  require(index.fallback(5, a) == 1);
  require(!index.fallback(5, b));
  // Exact duplicates select the last record even if its span is later rejected.
  // Span validity is deliberately not an input to the index.
  index.add(10, 7, b, "outer", 2);
  require(index.exact(10) == 2);
  require(index.fallback(5, a) == 1);
  require(index.fallback(7, b) == 2);
  require(!index.exact(99));
  index.add(12, 0, std::nullopt, "no-file", 3);
  require(index.fallback(0, std::nullopt) == 3);
  require(!index.fallback(0, a));
  index.add(13, -1, std::nullopt, "invalid-offset", 4);
  require(index.fallback(-1, std::nullopt) == 4);
  require(!index.fallback(4294967295LL, std::nullopt));
  // Stack order/duplicates do not alter forward-history first-key edge order.
  require(index.matchingRecords({"inner", "outer", "inner", "missing"}) ==
          std::vector<std::size_t>({0, 1}));
  require(index.matchingRecords({}).empty());
  require(index.matchingRecords({"no-file", "outer"}) ==
          std::vector<std::size_t>({0, 3}));
  cpp_context::MacroHistoryIndex next;
  require(!next.exact(10));
  require(next.matchingRecords({"outer"}).empty());
}
