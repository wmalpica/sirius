/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "log/logging.hpp"
#include "log/preparation_timeline.hpp"

#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>

namespace sirius::diagnostics {
// Opt-in investigation probe. Thread CPU time distinguishes computation from
// off-CPU waits; faults/context switches help explain the latter. No affinity changes.
class preparation_probe {
 public:
  preparation_probe(char const* name, void const* owner, long id = -1)
    : timeline_(name, owner, id), name_(name), owner_(owner), id_(id)
  {
    static bool const enabled = std::getenv("SIRIUS_PREPARATION_PROBE") != nullptr;
    enabled_                  = enabled;
    if (enabled_) reset();
  }
  void mark(char const* stage)
  {
    timeline::record(stage, timeline_.id, owner_);
    if (!enabled_) return;
    auto end = std::chrono::steady_clock::now();
    rusage usage{};
    getrusage(RUSAGE_THREAD, &usage);
    auto wall = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    if (wall >= 1000) {
      SIRIUS_LOG_INFO(
        "[prep_probe] scope={} stage={} owner={} id={} tid={} cpu_start={} "
        "cpu_end={} wall_us={} cpu_us={} minor_faults={} major_faults={} "
        "voluntary_cs={} involuntary_cs={}",
        name_,
        stage,
        owner_,
        id_,
        syscall(SYS_gettid),
        cpu_,
        sched_getcpu(),
        wall,
        cpu_us(usage) - cpu_us(usage_),
        usage.ru_minflt - usage_.ru_minflt,
        usage.ru_majflt - usage_.ru_majflt,
        usage.ru_nvcsw - usage_.ru_nvcsw,
        usage.ru_nivcsw - usage_.ru_nivcsw);
    }
    reset();
  }
  ~preparation_probe()
  {
    if (enabled_) {
      try {
        mark("finish");
      } catch (...) {
      }
    }
  }

 private:
  static long cpu_us(rusage const& r)
  {
    return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1000000L + r.ru_utime.tv_usec +
           r.ru_stime.tv_usec;
  }
  void reset()
  {
    getrusage(RUSAGE_THREAD, &usage_);
    cpu_   = sched_getcpu();
    start_ = std::chrono::steady_clock::now();
  }
  timeline::span timeline_;
  char const* name_;
  void const* owner_;
  long id_;
  bool enabled_ = false;
  int cpu_      = -1;
  rusage usage_{};
  std::chrono::steady_clock::time_point start_;
};
}  // namespace sirius::diagnostics
