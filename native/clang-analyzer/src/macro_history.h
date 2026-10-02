#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpp_context {

// TU-local positions in the existing history, built only after preprocessing.
// Reverse matching needs the last position; edge emission needs the first one.
class MacroHistoryIndex {
public:
  using Path = std::optional<std::filesystem::path>;

  void add(unsigned location, std::int64_t offset, Path path, const std::string &key,
           std::size_t position) {
    exact_.insert_or_assign(location, position);
    fallback_.insert_or_assign(std::make_pair(offset, std::move(path)), position);
    first_.emplace(key, position);
  }

  std::optional<std::size_t> exact(unsigned location) const {
    const auto found = exact_.find(location);
    if (found == exact_.end())
      return std::nullopt;
    return found->second;
  }

  std::optional<std::size_t> fallback(std::int64_t offset, const Path &path) const {
    const auto found = fallback_.find(std::make_pair(offset, path));
    if (found == fallback_.end())
      return std::nullopt;
    return found->second;
  }

  std::vector<std::size_t> matchingRecords(const std::vector<std::string> &keys) const {
    std::vector<std::size_t> result;
    result.reserve(keys.size());
    for (const auto &key : keys)
      if (const auto found = first_.find(key); found != first_.end())
        result.push_back(found->second);
    // Preserve forward-history, first-write-wins order, not stack/hash order.
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

private:
  std::unordered_map<unsigned, std::size_t> exact_;
  std::map<std::pair<std::int64_t, Path>, std::size_t> fallback_;
  std::unordered_map<std::string, std::size_t> first_;
};

} // namespace cpp_context
