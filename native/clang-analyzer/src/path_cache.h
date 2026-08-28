#pragma once

#include <filesystem>
#include <functional>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace cpp_context {

class CanonicalPathCache {
public:
  using Path = std::filesystem::path;
  using Resolver = std::function<Path(const Path &, std::error_code &)>;

  CanonicalPathCache()
      : resolver_([](const Path &path, std::error_code &error) {
          return std::filesystem::weakly_canonical(path, error);
        }) {}

  explicit CanonicalPathCache(Resolver resolver) : resolver_(std::move(resolver)) {}

  Path canonical(const Path &path) const {
    const auto key = path.native();
    if (const auto found = paths_.find(key); found != paths_.end())
      return found->second;

    std::error_code error;
    auto resolved = resolver_(path, error);
    if (error)
      resolved = path.lexically_normal();
    paths_.emplace(key, resolved);
    return resolved;
  }

  std::size_t size() const { return paths_.size(); }

private:
  Resolver resolver_;
  mutable std::unordered_map<Path::string_type, Path> paths_;
};

} // namespace cpp_context
