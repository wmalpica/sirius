/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#if __has_include(<ctrack.hpp>)
#include <ctrack.hpp>

#include <optional>
#define SIRIUS_TIMELINE_HAS_CTRACK 1
#endif

namespace sirius::diagnostics::timeline {
// Diagnostic-only, bounded per-thread recording. No file I/O or shared locks on
// the event path after first use. Records are published once and never overwritten.
// Dump at normal process exit; a crash may leave no dump. Dropped records are explicit.
struct event {
  uint64_t ns, cpu_ns, id, parent, owner, a, b;
  int tid, cpu;
  char const* name;
};
inline uint64_t now() noexcept
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}
inline uint64_t thread_cpu() noexcept
{
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
inline char const* path() noexcept
{
  static char const* p = std::getenv("SIRIUS_PREPARATION_TIMELINE");
  return p;
}
inline uint64_t next_id() noexcept
{
  static std::atomic<uint64_t> counter{1};
  return path() ? counter.fetch_add(1, std::memory_order_relaxed) : 0;
}
inline thread_local uint64_t context = 0;
struct buffer {
  static constexpr size_t capacity = 4194304;
  event* rows                      = static_cast<event*>(std::malloc(capacity * sizeof(event)));
  std::atomic<size_t> count{0}, dropped{0};
};
struct registry {
  std::mutex mutex;
  std::vector<buffer*> buffers;
  std::unordered_map<uint64_t, std::string> labels;
  uint64_t mono  = now();
  uint64_t epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
};
inline registry& state()
{
  static auto* p = new registry;
  return *p;
}
inline uint64_t label(std::string const& text)
{
  if (!path()) return 0;
  auto id = static_cast<uint64_t>(std::hash<std::string>{}(text));
  auto& s = state();
  std::lock_guard lock(s.mutex);
  s.labels.try_emplace(id, text);
  return id;
}
inline void dump() noexcept
{
  if (!path()) return;
  auto& s = state();
  auto* f = std::fopen(path(), "w");
  if (!f) {
    std::fprintf(stderr, "preparation timeline: cannot open %s\n", path());
    return;
  }
  std::fprintf(
    f, "# mono_ns=%llu epoch_ns=%llu\n", (unsigned long long)s.mono, (unsigned long long)s.epoch);
  std::fprintf(f, "ns,cpu_ns,id,parent,owner,a,b,tid,cpu,event\n");
  size_t dropped = 0;
  std::lock_guard lock(s.mutex);
  for (auto* b : s.buffers) {
    auto n = b->count.load(std::memory_order_acquire);
    dropped += b->dropped.load();
    for (size_t i = 0; i < n; ++i) {
      auto const& e = b->rows[i];
      std::fprintf(f,
                   "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%s\n",
                   (unsigned long long)e.ns,
                   (unsigned long long)e.cpu_ns,
                   (unsigned long long)e.id,
                   (unsigned long long)e.parent,
                   (unsigned long long)e.owner,
                   (unsigned long long)e.a,
                   (unsigned long long)e.b,
                   e.tid,
                   e.cpu,
                   e.name);
    }
  }
  for (auto const& [id, text] : s.labels)
    std::fprintf(f, "# label=%llu %s\n", (unsigned long long)id, text.c_str());
  std::fprintf(f, "# dropped=%zu\n", dropped);
  std::fclose(f);
#ifdef SIRIUS_TIMELINE_HAS_CTRACK
  if (std::getenv("SIRIUS_PREPARATION_CTRACK")) ctrack::result_print();
#endif
}
inline buffer& local()
{
  static bool const registered = [] {
    (void)state();
    std::atexit(dump);
    return true;
  }();
  (void)registered;
  static thread_local buffer* b = [] {
    auto* p = new buffer;
    auto& s = state();
    std::lock_guard lock(s.mutex);
    s.buffers.push_back(p);
    return p;
  }();
  return *b;
}
inline void record(char const* name,
                   uint64_t id       = 0,
                   void const* owner = nullptr,
                   uint64_t a        = 0,
                   uint64_t b        = 0,
                   uint64_t parent   = 0) noexcept
{
  if (!path()) return;
  try {
    auto& out = local();
    auto n    = out.count.load(std::memory_order_relaxed);
    if (!out.rows || n == buffer::capacity) {
      out.dropped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    static thread_local int const tid = static_cast<int>(syscall(SYS_gettid));
    out.rows[n]                       = {now(),
                                         thread_cpu(),
                                         id,
                   parent ? parent : context,
                                         reinterpret_cast<uintptr_t>(owner),
                                         a,
                                         b,
                                         tid,
                                         sched_getcpu(),
                                         name};
    out.count.store(n + 1, std::memory_order_release);
  } catch (...) { /* Diagnostics must not change query exception handling. */
  }
}
struct span {
  uint64_t id, previous;
  char const* name;
  void const* owner;
#ifdef SIRIUS_TIMELINE_HAS_CTRACK
  std::optional<ctrack::EventHandler> ctrack_scope;
#endif
  span(
    char const* n, void const* o = nullptr, uint64_t a = 0, uint64_t b = 0, uint64_t existing = 0)
    : id(existing ? existing : next_id()), previous(context), name(n), owner(o)
  {
    record("begin", id, o, a, b);
    context = id;
    record(n, id, o, a, b);
#ifdef SIRIUS_TIMELINE_HAS_CTRACK
    if (std::getenv("SIRIUS_PREPARATION_CTRACK")) ctrack_scope.emplace(0, "preparation", n);
#endif
  }
  ~span()
  {
    record("end", id, owner);
    context = previous;
  }
  span(span const&)            = delete;
  span& operator=(span const&) = delete;
};
}  // namespace sirius::diagnostics::timeline
