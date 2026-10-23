/*
 * Copyright 2026, Sirius Contributors.
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

#include "exec/wake_event.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace sirius::creator {

/// Waiting edges may point only to earlier scan ranks. This proves that filter
/// waits cannot create a cycle, including mutual optional-filter dependencies.
inline bool can_wait_for_build(std::size_t candidate,
                               std::span<std::optional<std::size_t> const> build_scans)
{
  if (build_scans.empty()) return false;
  for (auto rank : build_scans) {
    if (!rank || *rank >= candidate) return false;
  }
  return true;
}

/// One warm-up per scan. Failed/deferred attempts remain retryable; in-flight
/// claims exclude duplicate creation. Request destruction releases every claim.
class lookahead_candidates {
 public:
  explicit lookahead_candidates(std::size_t count) : _slots(count) {}
  std::vector<std::size_t> eligible() const
  {
    std::lock_guard lock(_mutex);
    std::vector<std::size_t> result;
    for (std::size_t n = 0; n < _slots.size(); ++n) {
      auto i = (_cursor + n) % _slots.size();
      if (!_slots[i].issued && !_slots[i].inflight) result.push_back(i);
    }
    return result;
  }
  bool claim(std::size_t i)
  {
    std::lock_guard lock(_mutex);
    auto& slot = _slots.at(i);
    if (slot.issued || slot.inflight) return false;
    slot.inflight = true;
    _cursor       = (i + 1) % _slots.size();
    return true;
  }
  void finish(std::size_t i,
              bool created,
              std::weak_ptr<exec::wake_event> const& wake = {},
              std::uint64_t claimed_generation            = 0) noexcept
  {
    {
      std::lock_guard lock(_mutex);
      auto& slot    = _slots[i];
      slot.inflight = false;
      slot.issued   = slot.issued || created;
    }
    // A producer can notify while this claim still hides the candidate. If the
    // scheduler already consumed that wake, give it another after releasing the
    // claim. With no intervening event, do not retry a failed creation in a loop.
    if (!created) {
      if (auto signal = wake.lock(); signal && signal->generation() != claimed_generation)
        signal->notify();
    }
  }
  /// Returns true once per deferral episode, for useful non-polling trace output.
  bool defer(std::size_t i)
  {
    std::lock_guard lock(_mutex);
    auto& start = _slots.at(i).deferred_since;
    if (start) return false;
    start = std::chrono::steady_clock::now();
    return true;
  }
  std::optional<double> release_deferral(std::size_t i)
  {
    std::lock_guard lock(_mutex);
    auto& start = _slots.at(i).deferred_since;
    if (!start) return {};
    auto ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *start).count();
    start.reset();
    return ms;
  }

 private:
  struct slot {
    bool issued{false};
    bool inflight{false};
    std::optional<std::chrono::steady_clock::time_point> deferred_since;
  };
  mutable std::mutex _mutex;
  std::vector<slot> _slots;
  std::size_t _cursor{0};
};

}  // namespace sirius::creator
