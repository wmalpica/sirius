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

#include "catch.hpp"
#include "creator/lookahead_policy.hpp"
#include "creator/task_creator.hpp"
#include "exec/channel.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "exec/wake_event.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using sirius::creator::lookahead_candidates;
using sirius::op::sirius_dynamic_filter_set;

namespace {
class host_filter final : public sirius::op::sirius_dynamic_filter {
 public:
  sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::BLOOM;
  }
};
}  // namespace

TEST_CASE("lookahead defers only acyclic independently earlier builds", "[lookahead][policy]")
{
  using sirius::creator::can_wait_for_build;
  std::vector<std::optional<std::size_t>> earlier{0, 2};
  CHECK(can_wait_for_build(3, earlier));
  CHECK_FALSE(can_wait_for_build(2, earlier));  // depends on itself
  CHECK_FALSE(can_wait_for_build(1, earlier));  // later build could create a wait cycle
  CHECK_FALSE(can_wait_for_build(3, {}));
  earlier.push_back({});
  CHECK_FALSE(can_wait_for_build(3, earlier));  // unknown/manual/cyclic source
}

TEST_CASE("deferred lookahead candidates remain retryable without blocking other scans",
          "[lookahead][policy]")
{
  lookahead_candidates candidates(3);
  CHECK(candidates.defer(0));
  CHECK_FALSE(candidates.defer(0));
  CHECK(candidates.eligible() == std::vector<std::size_t>{0, 1, 2});
  CHECK(candidates.claim(1));
  CHECK_FALSE(candidates.claim(1));
  CHECK(candidates.eligible() == std::vector<std::size_t>{2, 0});
  candidates.finish(1, true);
  CHECK(candidates.eligible() == std::vector<std::size_t>{2, 0});
  CHECK(candidates.claim(0));
  candidates.finish(0, false);  // request was cancelled or recheck deferred it
  CHECK(candidates.claim(0));
  candidates.finish(0, true);
  CHECK(candidates.eligible() == std::vector<std::size_t>{2});
  CHECK(candidates.release_deferral(0).has_value());
  CHECK_FALSE(candidates.release_deferral(0).has_value());
  lookahead_candidates empty(0);
  CHECK(empty.eligible().empty());
}

TEST_CASE("concurrent lookahead checks claim a scan once", "[lookahead][policy]")
{
  lookahead_candidates candidates(1);
  std::atomic<int> claims{0};
  std::vector<std::thread> workers;
  for (int n = 0; n < 8; ++n)
    workers.emplace_back([&] {
      if (candidates.claim(0)) ++claims;
    });
  for (auto& worker : workers)
    worker.join();
  CHECK(claims.load() == 1);
  candidates.finish(0, false);
  CHECK(candidates.claim(0));
}

TEST_CASE("filter-aware lookahead waits for all relevant producers", "[lookahead][channel]")
{
  sirius_dynamic_filter_set channel;
  auto first     = channel.register_producer({0});
  auto second    = channel.register_producer({0});
  auto unrelated = channel.register_producer({1});
  std::array<std::size_t, 1> const columns{0};
  CHECK_FALSE(channel.readiness_for(columns).registration_frozen);
  channel.freeze_registration();
  CHECK(channel.num_producers() == 3);
  CHECK(channel.readiness_for(columns).relevant_producers == 2);
  CHECK(channel.readiness_for(columns).pending_producers == 2);
  REQUIRE(first.push_filter(0, std::make_shared<host_filter>()));
  first.finish(sirius_dynamic_filter_set::completion::PUBLISHED);
  auto partial = channel.readiness_for(columns);
  CHECK(partial.published_filters == 1);
  CHECK(partial.pending_producers == 1);  // first publication alone must not release the scan
  second.finish(sirius_dynamic_filter_set::completion::SKIPPED);
  auto ready = channel.readiness_for(columns);
  CHECK(ready.pending_producers == 0);
  CHECK(ready.published_filters == 1);
  CHECK_FALSE(channel.snapshot().terminal());  // unrelated producer must not delay this scan
}

TEST_CASE("declined failed cancelled and abandoned producers release lookahead",
          "[lookahead][channel]")
{
  for (auto completion : {sirius_dynamic_filter_set::completion::SKIPPED,
                          sirius_dynamic_filter_set::completion::FAILED,
                          sirius_dynamic_filter_set::completion::CANCELLED}) {
    sirius_dynamic_filter_set channel;
    auto producer = channel.register_producer({4});
    std::array<std::size_t, 1> columns{4};
    channel.freeze_registration();
    producer.finish(completion);
    auto state = channel.readiness_for(columns);
    CHECK(state.registration_frozen);
    CHECK(state.pending_producers == 0);
    CHECK(state.published_filters == 0);
  }
  sirius_dynamic_filter_set channel;
  {
    auto abandoned = channel.register_producer({4});
  }
  channel.freeze_registration();
  std::array<std::size_t, 1> columns{4};
  CHECK(channel.readiness_for(columns).pending_producers == 0);
}

TEST_CASE("ignored and closed filter targets cannot strand lookahead", "[lookahead][channel]")
{
  sirius_dynamic_filter_set channel;
  auto producer = channel.register_producer({0, 1});
  channel.freeze_registration();
  std::array<std::size_t, 1> columns{0};
  CHECK(channel.readiness_for(columns).pending_producers == 1);
  channel.ignore_columns({0});
  CHECK(channel.readiness_for(columns).pending_producers == 0);
  std::array<std::size_t, 1> other{1};
  CHECK(channel.readiness_for(other).pending_producers == 1);
  channel.close_for_new_filters();
  CHECK(channel.readiness_for(other).pending_producers == 0);
}

TEST_CASE("filter completion between inspection and sleep cannot lose a wakeup",
          "[lookahead][wakeup]")
{
  auto wake = std::make_shared<sirius::exec::wake_event>();
  sirius_dynamic_filter_set channel;
  auto producer = channel.register_producer({0});
  channel.freeze_registration();
  channel.observe_wakeup(wake);
  channel.observe_wakeup(wake);  // duplicate subscription must not duplicate notifications
  auto generation = wake->generation();
  producer.finish(sirius_dynamic_filter_set::completion::SKIPPED);
  CHECK(wake->generation() == generation + 1);
  std::promise<bool> result;
  auto future = result.get_future();
  std::thread waiter([&] { result.set_value(wake->wait(generation)); });
  CHECK(future.wait_for(1s) == std::future_status::ready);
  wake->interrupt();
  waiter.join();
}

TEST_CASE("filter publication and cancellation wake a sleeping scheduler", "[lookahead][wakeup]")
{
  auto wake = std::make_shared<sirius::exec::wake_event>();
  sirius_dynamic_filter_set channel;
  auto producer = channel.register_producer({0});
  channel.freeze_registration();
  channel.observe_wakeup(wake);
  auto generation = wake->generation();
  std::promise<bool> result;
  auto future = result.get_future();
  std::thread waiter([&] { result.set_value(wake->wait(generation)); });
  CHECK(producer.push_filter(0, std::make_shared<host_filter>()));
  CHECK(future.wait_for(1s) == std::future_status::ready);
  wake->interrupt();
  waiter.join();
  // The channel must not own the scheduler or any query work while waiting.
  std::weak_ptr<sirius::exec::wake_event> weak = wake;
  wake.reset();
  CHECK(weak.expired());
  CHECK_NOTHROW(producer.finish(sirius_dynamic_filter_set::completion::CANCELLED));
}

TEST_CASE("queue pushes and device readiness share the lookahead wake signal",
          "[lookahead][wakeup]")
{
  auto wake = std::make_shared<sirius::exec::wake_event>();
  struct item {
    int id;
  };
  sirius::exec::multi_index_priority_queue<item> queue(
    [](item const&) {
      return sirius::exec::index_keys{0, sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, 0, -1};
    },
    wake);
  auto generation = wake->generation();
  CHECK(queue.push(std::make_unique<item>(item{1})));
  CHECK(wake->generation() != generation);  // direct downgrade return, no channel event
  CHECK(queue.pop()->id == 1);
  sirius::exec::channel<std::unique_ptr<item>> requests(wake);
  auto publisher = requests.make_publisher();
  generation     = wake->generation();
  CHECK(publisher.send(std::make_unique<item>(item{2})));
  CHECK(wake->generation() != generation);
  CHECK(requests.get()->id == 2);
}

TEST_CASE("shutdown interrupts a lookahead wait without polling", "[lookahead][wakeup]")
{
  auto wake       = std::make_shared<sirius::exec::wake_event>();
  auto generation = wake->generation();
  std::promise<bool> result;
  auto future = result.get_future();
  std::thread waiter([&] { result.set_value(wake->wait(generation)); });
  wake->interrupt();
  CHECK(future.wait_for(1s) == std::future_status::ready);
  waiter.join();
  CHECK_FALSE(future.get());
}

TEST_CASE("destroying a queued lookahead request releases its candidate claim",
          "[lookahead][request_lifetime]")
{
  auto progress = std::make_shared<lookahead_candidates>(1);
  REQUIRE(progress->claim(0));
  {
    auto request              = std::make_unique<sirius::creator::task_creation_request>();
    request->finish_lookahead = [progress](bool created) { progress->finish(0, created); };
    request->type             = sirius::creator::request_type::lookahead;
    // The same destructor runs on failed creation, queue drain and cancellation.
  }
  CHECK(progress->claim(0));
  {
    auto request               = std::make_unique<sirius::creator::task_creation_request>();
    request->finish_lookahead  = [progress](bool created) { progress->finish(0, created); };
    request->lookahead_created = true;
  }
  CHECK(progress->eligible().empty());
}

TEST_CASE("completion while a lookahead claim is hidden is retried after release",
          "[lookahead][wakeup]")
{
  auto wake = std::make_shared<sirius::exec::wake_event>();
  lookahead_candidates candidates(1);
  REQUIRE(candidates.claim(0));
  auto claimed = wake->generation();
  wake->notify();  // publication happened, but candidate is still in flight
  CHECK(candidates.eligible().empty());
  auto scheduler_observed = wake->generation();
  candidates.finish(0, false, wake, claimed);
  CHECK(wake->generation() != scheduler_observed);
  CHECK(candidates.claim(0));
  auto retry_generation = wake->generation();
  candidates.finish(0, false, wake, retry_generation);
  CHECK(wake->generation() == retry_generation);  // no busy retry without a new event
}
