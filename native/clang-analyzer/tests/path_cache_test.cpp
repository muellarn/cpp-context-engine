#include "path_cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {
bool countAllocations = false;
std::size_t allocations = 0;
}

void *operator new(std::size_t size) {
  if (countAllocations)
    ++allocations;
  if (void *memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

namespace {

using Path = std::filesystem::path;

void require(bool condition) {
  if (!condition)
    throw std::runtime_error("path cache assertion failed");
}

void testCachesSuccessAndFallbackByExactRawPath() {
  std::map<Path::string_type, unsigned> calls;
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
  require(calls[Path("source.cpp").native()] == 1);

  require(cache.canonical("missing/../broken") == Path("broken"));
  require(cache.canonical("missing/../broken") == Path("broken"));
  require(calls[Path("missing/../broken").native()] == 1);

  require(cache.canonical("./source.cpp") == Path("/canonical/./source.cpp"));
  require(calls[Path("./source.cpp").native()] == 1);
  require(calls[Path("source.cpp").native()] == 1);
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

void testWarmCacheHitsDoNotAllocate() {
  unsigned calls = 0;
  cpp_context::CanonicalPathCache cache(
      [&calls](const Path &path, std::error_code &error) {
        ++calls;
        error.clear();
        return path;
      });
  // Avoid small-string optimization; only already-populated hits are measured.
  const Path raw = Path("/canonical") / std::string(128, 'x') / "source.cpp";
  const Path expected = cache.canonical(raw);
  bool identical = true;
  allocations = 0;
  countAllocations = true;
  for (unsigned index = 0; index < 16; ++index) {
    const auto &cached = cache.canonical(raw);
    identical = identical && cached == expected;
  }
  countAllocations = false;
  std::printf("warm cache: hits=16 allocations=%zu resolver_calls=%u identical=%d\n",
              allocations, calls, identical);
  std::fflush(stdout);
  require(identical && calls == 1);
  require(allocations == 0);
}

void testReferencesOwnTemporaryInputsAndSurviveGrowth() {
  unsigned calls = 0;
  cpp_context::CanonicalPathCache cache(
      [&calls](const Path &path, std::error_code &error) {
        ++calls;
        error.clear();
        return path;
      });
  const Path expected = Path("/canonical") / std::string(128, 'y') / "first.cpp";
  // The input temporary dies here; the result must belong to the cache.
  const auto &first = cache.canonical(Path(expected));
  const auto *address = &first;
  require(first == expected && calls == 1);
  for (unsigned index = 0; index < 512; ++index)
    cache.canonical(Path("/growth") / std::to_string(index));
  require(cache.size() == 513 && calls == 513);
  const auto &again = cache.canonical(Path(expected));
  require(&again == address && first == expected && calls == 513);
}

} // namespace

int main() {
  testCachesSuccessAndFallbackByExactRawPath();
  testCanonicalizesRelativeAndSymlinkPaths();
  testCacheLifetimeIsInstanceLocal();
  testWarmCacheHitsDoNotAllocate();
  testReferencesOwnTemporaryInputsAndSurviveGrowth();
}
