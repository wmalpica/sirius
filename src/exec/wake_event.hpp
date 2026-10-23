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

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace sirius::exec {

/// A generation-counted notification shared by queues and optional policy producers.
/// Capture generation BEFORE inspecting work; a notification before wait is not lost.
/// No callbacks run here, so producers may notify while holding their own state locks.
class wake_event {
 public:
  std::uint64_t generation() const noexcept
  {
    std::lock_guard lock(_mutex);
    return _generation;
  }
  void notify() noexcept
  {
    {
      std::lock_guard lock(_mutex);
      ++_generation;
    }
    _cv.notify_all();
  }
  bool wait(std::uint64_t observed) const
  {
    std::unique_lock lock(_mutex);
    _cv.wait(lock, [&] { return _generation != observed || _interrupted; });
    return !_interrupted;
  }
  void interrupt() noexcept
  {
    {
      std::lock_guard lock(_mutex);
      _interrupted = true;
    }
    _cv.notify_all();
  }

 private:
  mutable std::mutex _mutex;
  mutable std::condition_variable _cv;
  std::uint64_t _generation{0};
  bool _interrupted{false};
};

}  // namespace sirius::exec
