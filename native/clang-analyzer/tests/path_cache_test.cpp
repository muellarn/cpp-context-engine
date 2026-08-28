#include "path_cache.h"

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

using Path = std::filesystem::path;

void require(bool condition) {
  if (!condition)
    throw std::runtime_error("path cache assertion failed");
}

void testCachesSuccessAndFallbackByExactRawPath() {
  std::map<std::string, unsigned> calls;
  cpp_context::CanonicalPathCache cache(
      [&calls](const Path &path, std::error_code &error) {
        ++calls[path.native()];
        if (path == "missing/../broken") {
          error = std::make_error_code(std::errc::permission_denied);
          return Path{};
        }
        error.clear();
        return Path("/canonical") / path;
      });

  require(cache.canonical("source.cpp") == Path("/canonical/source.cpp"));
  require(cache.canonical("source.cpp") == Path("/canonical/source.cpp"));
  require(calls["source.cpp"] == 1);

  require(cache.canonical("missing/../broken") == Path("broken"));
  require(cache.canonical("missing/../broken") == Path("broken"));
  require(calls["missing/../broken"] == 1);

  require(cache.canonical("./source.cpp") == Path("/canonical/./source.cpp"));
  require(calls["./source.cpp"] == 1);
  require(calls["source.cpp"] == 1);
}

void testCanonicalizesRelativeAndSymlinkPaths() {
  const auto base = std::filesystem::temp_directory_path() / "cpp-context-path-cache-test";
  std::error_code error;
  std::filesystem::remove_all(base, error);
  require(std::filesystem::create_directories(base / "real", error));
  std::filesystem::create_directory_symlink(base / "real", base / "alias", error);
  require(!error);

  unsigned calls = 0;
  cpp_context::CanonicalPathCache cache(
      [&calls](const Path &path, std::error_code &canonicalError) {
        ++calls;
        return std::filesystem::weakly_canonical(path, canonicalError);
      });
  const auto raw = base / "alias" / ".." / "alias" / "file.hpp";
  const auto expected = base / "real" / "file.hpp";
  require(cache.canonical(raw) == expected);
  require(cache.canonical(raw) == expected);
  require(calls == 1);

  std::filesystem::remove_all(base, error);
}

void testCacheLifetimeIsInstanceLocal() {
  unsigned calls = 0;
  const auto resolver = [&calls](const Path &path, std::error_code &error) {
    ++calls;
    error.clear();
    return path;
  };
  {
    cpp_context::CanonicalPathCache cache(resolver);
    require(cache.canonical("one") == Path("one"));
    require(cache.canonical("two") == Path("two"));
    require(cache.size() == 2);
  }
  cpp_context::CanonicalPathCache next(resolver);
  require(next.canonical("one") == Path("one"));
  require(calls == 3);
}

} // namespace

int main() {
  testCachesSuccessAndFallbackByExactRawPath();
  testCanonicalizesRelativeAndSymlinkPaths();
  testCacheLifetimeIsInstanceLocal();
}
