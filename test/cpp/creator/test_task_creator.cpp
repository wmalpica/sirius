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

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "exec/config.hpp"
#include "gpu_context.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_pipeline_hashmap.hpp"

#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb.hpp>
#include <duckdb/main/connection.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::creator;
using namespace sirius::exec;
using namespace sirius::parallel;
using namespace sirius::pipeline;
using namespace sirius::op::scan;
using namespace std::chrono_literals;
using namespace sirius::op;
using namespace sirius;
using sirius::sirius_pipeline_hashmap;

//===----------------------------------------------------------------------===//
// Mock GPU Physical Operator
//===----------------------------------------------------------------------===//

/**
 * @brief A mock GPU physical operator for testing get_operator_for_next_task.
 *
 * This mock allows configuring the hint that get_next_task_hint() returns,
 * enabling controlled testing of different scheduling scenarios.
 */
class mock_sirius_physical_operator : public sirius_physical_operator {
 public:
  mock_sirius_physical_operator(
    SiriusPhysicalOperatorType op_type = SiriusPhysicalOperatorType::PROJECTION)
    : sirius_physical_operator(op_type, {}, 0), _use_custom_hint(false), _custom_hint(std::nullopt)
  {
  }

  /**
   * @brief Enable custom hint mode and set the hint to return.
   *
   * When custom hint mode is enabled, get_next_task_hint() returns the
   * configured hint instead of computing one from ports.
   */
  void set_custom_hint(std::optional<sirius::op::task_creation_hint> hint)
  {
    _use_custom_hint = true;
    _custom_hint     = std::move(hint);
  }

  /**
   * @brief Disable custom hint mode (use default port-based behavior).
   */
  void clear_custom_hint() { _use_custom_hint = false; }

  /**
   * @brief Override to return configured hint when in custom mode.
   */
  std::optional<task_creation_hint> get_next_task_hint() override
  {
    if (_use_custom_hint) { return _custom_hint; }
    // Fall back to parent implementation
    return sirius_physical_operator::get_next_task_hint();
  }

 private:
  bool _use_custom_hint;
  std::optional<sirius::op::task_creation_hint> _custom_hint;
};

/**
 * @brief A mock DuckDB scan operator for testing get_next_task_hint behavior.
 *
 * This mock allows controlling the exhausted state to test different scenarios
 * in get_next_task_hint() which checks the exhausted flag.
 */
class mock_sirius_physical_duckdb_scan : public sirius_physical_duckdb_scan {
 public:
  mock_sirius_physical_duckdb_scan()
    : sirius_physical_duckdb_scan(
        {duckdb::LogicalType::BIGINT},                    // types - at least one type
        duckdb::TableFunction(),                          // function
        nullptr,                                          // bind_data
        {duckdb::LogicalType::BIGINT},                    // returned_types
        {duckdb::ColumnIndex(0)},                         // column_ids - at least one column
        {0},                                              // projection_ids
        {"col0"},                                         // names
        nullptr,                                          // table_filters
        0,                                                // estimated_cardinality
        duckdb::ExtraOperatorInfo(),                      // extra_info
        {},                                               // parameters
        {})                                               // virtual_columns
  {
  }

  /**
   * @brief Set the exhausted state for testing.
   *
   * @param value true if the scan should be exhausted, false otherwise
   */
  void set_exhausted(bool value) { exhausted.store(value); }

  /**
   * @brief Get the current exhausted state.
   *
   * @return true if exhausted, false otherwise
   */
  bool is_exhausted() const { return exhausted.load(); }
};

/**
 * @brief A mock GPU pipeline for testing FULL barrier scenarios.
 *
 * This class allows controlling the return value of is_pipeline_finished()
 * for testing purposes.
 */
class mock_gpu_pipeline : public sirius_pipeline {
 public:
  explicit mock_gpu_pipeline(sirius_engine& engine) : sirius_pipeline(engine), _finished(false) {}

  void set_finished(bool finished) { _finished = finished; }

  bool is_pipeline_finished() const override { return _finished; }

  void add_operators(std::vector<std::shared_ptr<sirius_physical_operator>> ops) {
    for (auto& op : ops) {
      operators.push_back(*op);
    }
    if (!ops.empty()) {
      sink = ops.back().get();
    }
  }

 private:
  bool _finished;
};

/**
 * @brief A mock GPU pipeline for testing.
 *
 * This requires a sirius_engine reference, so we use a factory pattern
 * to create test pipelines when we have the necessary context.
 */
class mock_pipeline_builder {
 public:
  /**
   * @brief Create a mock pipeline with specified source and operators.
   *
   * Since sirius_pipeline requires sirius_engine, we set up ports directly
   * on operators to control get_next_task_hint() behavior.
   */
  static void setup_operator_with_pipeline_port(mock_sirius_physical_operator& op,
                                                const std::string& port_id,
                                                MemoryBarrierType barrier_type,
                                                cucascade::shared_data_repository* repo,
                                                duckdb::shared_ptr<sirius_pipeline> src_pipeline,
                                                duckdb::shared_ptr<sirius_pipeline> dest_pipeline)
  {
    auto port           = std::make_unique<sirius_physical_operator::port>();
    port->type          = barrier_type;
    port->repo          = repo;
    port->src_pipeline  = src_pipeline;
    port->dest_pipeline = dest_pipeline;
    op.add_port(port_id, std::move(port));
  }
};

//===----------------------------------------------------------------------===//
// Pipeline Setup Utility
//===----------------------------------------------------------------------===//

/**
 * @brief Utility structure to hold the result of create_connected_pipelines.
 */
struct connected_pipelines_result {
  // The four pipelines
  duckdb::shared_ptr<mock_gpu_pipeline> pipe_a;
  duckdb::shared_ptr<mock_gpu_pipeline> pipe_b;
  duckdb::shared_ptr<mock_gpu_pipeline> pipe_c;
  duckdb::shared_ptr<mock_gpu_pipeline> pipe_d;

  // The four operators (one per pipeline)
  std::shared_ptr<mock_sirius_physical_duckdb_scan> op_a;
  std::shared_ptr<mock_sirius_physical_operator> op_b;
  std::shared_ptr<mock_sirius_physical_duckdb_scan> op_c;
  std::shared_ptr<mock_sirius_physical_operator> op_d;

  // Data repositories for each port (need to be kept alive)
  std::unique_ptr<cucascade::shared_data_repository> repo_ab;
  std::unique_ptr<cucascade::shared_data_repository> repo_bd;
  std::unique_ptr<cucascade::shared_data_repository> repo_cd;
 
};

//===----------------------------------------------------------------------===//
// Testable Task Creator
//===----------------------------------------------------------------------===//

/**
 * @brief A testable subclass of task_creator that tracks scheduled tasks.
 *
 * This class overrides schedule() to record what operators objects
 * were scheduled, allowing tests to verify correct scheduling behavior.
 */
class testable_task_creator : public task_creator {
 public:
  testable_task_creator(int num_threads,
                        sirius_pipeline_hashmap& gpu_pipeline_map,
                        duckdb::ClientContext& client_context,
                        pipeline_executor& pipeline_executor,
                        sirius::memory::sirius_memory_reservation_manager& mem_res_mgr)
    : task_creator(
        exec::thread_pool_config{.num_threads = num_threads, .thread_name_prefix = "task_creator"},
        mem_res_mgr)
  {
    this->set_client_context(client_context);
    this->set_pipeline_executor(pipeline_executor);
  }

  void schedule(op::sirius_physical_operator* request) override
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    if (request) {
      _scheduled_nodes.push_back(request);
      _scheduled_pipelines.push_back(request->get_pipeline());
    }
    _schedule_count++;
  }

  size_t get_schedule_count() const { return _schedule_count.load(); }

  std::vector<sirius_physical_operator*> get_scheduled_nodes()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    return _scheduled_nodes;
  }

  std::vector<duckdb::shared_ptr<sirius_pipeline>> get_scheduled_pipelines()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    return _scheduled_pipelines;
  }

  void clear_scheduled()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    _scheduled_nodes.clear();
    _scheduled_pipelines.clear();
    _schedule_count.store(0);
  }

  [[nodiscard]] bool is_running() const { return _running.load(); }

  // Expose protected method for testing
  using task_creator::get_operator_for_next_task;

 private:
  std::atomic<size_t> _schedule_count{0};
  std::vector<sirius_physical_operator*> _scheduled_nodes;
  std::vector<duckdb::shared_ptr<sirius_pipeline>> _scheduled_pipelines;
  std::mutex _scheduled_mutex;
};

//===----------------------------------------------------------------------===//
// Test Fixture Helper
//===----------------------------------------------------------------------===//

/**
 * @brief Helper class to set up minimal test infrastructure.
 */
class test_fixture {
 public:
  test_fixture()
    : db(nullptr),
      con(db),
      gpu_context(*con.context),
      engine(*con.context, gpu_context),
      memory_manager([] {
        cucascade::memory::reservation_manager_configurator builder;
        const size_t gpu_capacity  = 2ull << 27;
        const double limit_ratio   = 0.75;
        const size_t host_capacity = 4ull << 27;

        builder.set_number_of_gpus(1)
          .set_gpu_usage_limit(gpu_capacity)
          .set_reservation_fraction_per_gpu(limit_ratio)
          .set_per_host_capacity(host_capacity)
          .use_host_per_gpu()
          .set_reservation_fraction_per_host(limit_ratio);

        // Build configuration with topology detection
        auto space_configs = builder.build();
        return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
          std::move(space_configs));
      }()),
      pipeline_exec(exec::thread_pool_config{.num_threads = 1},
                    exec::thread_pool_config{.num_threads = 2},
                    *memory_manager),
      empty_pipelines(),
      pipeline_map(empty_pipelines)
  {
  }

  /**
   * @brief Create a mock GPU pipeline with controllable finished state.
   */
  duckdb::shared_ptr<mock_gpu_pipeline> create_mock_pipeline()
  {
    return duckdb::make_shared_ptr<mock_gpu_pipeline>(engine);
  }

  duckdb::DuckDB db;
  duckdb::Connection con;
  duckdb::GPUContext gpu_context;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager;
  sirius_engine engine;
  pipeline_executor pipeline_exec;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> empty_pipelines;
  sirius::sirius_pipeline_hashmap pipeline_map;
};


/**
 * @brief Create a connected pipeline structure for testing.
 *
 * Creates 4 pipelines with the following structure:
 *   A -> B -> D
 *        C -> D
 *
 * - pipe_b has port port_ab (from A)
 * - pipe_d has two ports: port_bd (from B) and port_cd (from C)
 * - Each pipeline has one operator
 *
 * @param fixture The test fixture (used to create mock pipelines)
 * @param port_ab_barrier_type Barrier type for port_ab
 * @param port_ab_num_batches Number of data batches in port_ab's repository
 * @param port_bd_barrier_type Barrier type for port_bd
 * @param port_bd_num_batches Number of data batches in port_bd's repository
 * @param port_cd_barrier_type Barrier type for port_cd
 * @param port_cd_num_batches Number of data batches in port_cd's repository
 * @return connected_pipelines_result containing all created objects
 */
 inline connected_pipelines_result create_connected_pipelines(
  test_fixture& fixture,
  MemoryBarrierType port_ab_barrier_type,
  size_t port_ab_num_batches,
  MemoryBarrierType port_bd_barrier_type,
  size_t port_bd_num_batches,
  MemoryBarrierType port_cd_barrier_type,
  size_t port_cd_num_batches)
{
  connected_pipelines_result result;

  // Create the four operators
  result.op_a = std::make_shared<mock_sirius_physical_duckdb_scan>();
  result.op_a->set_exhausted(false);
  result.op_b = std::make_shared<mock_sirius_physical_operator>();
  result.op_c = std::make_shared<mock_sirius_physical_duckdb_scan>();
  result.op_c->set_exhausted(false);
  result.op_d = std::make_shared<mock_sirius_physical_operator>();


  // Create the four pipelines
  result.pipe_a = fixture.create_mock_pipeline();
  result.pipe_a->add_operators({result.op_a});
  result.pipe_b = fixture.create_mock_pipeline();
  result.pipe_b->add_operators({result.op_b});
  result.pipe_c = fixture.create_mock_pipeline();
  result.pipe_c->add_operators({result.op_c});
  result.pipe_d = fixture.create_mock_pipeline();
  result.pipe_d->add_operators({result.op_d});

  // Create data repositories for the three ports
  result.repo_ab = std::make_unique<cucascade::shared_data_repository>();
  result.repo_bd = std::make_unique<cucascade::shared_data_repository>();
  result.repo_cd = std::make_unique<cucascade::shared_data_repository>();

  // Populate repo_ab with the specified number of batches
  for (size_t i = 0; i < port_ab_num_batches; ++i) {
    auto batch = std::make_shared<cucascade::data_batch>(1, nullptr);
    result.repo_ab->add_data_batch(batch);
  }

  // Populate repo_bd with the specified number of batches
  for (size_t i = 0; i < port_bd_num_batches; ++i) {
    auto batch = std::make_shared<cucascade::data_batch>(1, nullptr);
    result.repo_bd->add_data_batch(batch);
  }

  // Populate repo_cd with the specified number of batches
  for (size_t i = 0; i < port_cd_num_batches; ++i) {
    auto batch = std::make_shared<cucascade::data_batch>(1, nullptr);
    result.repo_cd->add_data_batch(batch);
  }

  // Set up port_ab on op_b (A -> B)
  mock_pipeline_builder::setup_operator_with_pipeline_port(*result.op_b,
                                                           "port_ab",
                                                           port_ab_barrier_type,
                                                           result.repo_ab.get(),
                                                           result.pipe_a,
                                                           result.pipe_b);

  // Set up port_bd on op_d (B -> D)
  mock_pipeline_builder::setup_operator_with_pipeline_port(*result.op_d,
                                                           "port_bd",
                                                           port_bd_barrier_type,
                                                           result.repo_bd.get(),
                                                           result.pipe_b,
                                                           result.pipe_d);

  // Set up port_cd on op_d (C -> D)
  mock_pipeline_builder::setup_operator_with_pipeline_port(*result.op_d,
                                                           "port_cd",
                                                           port_cd_barrier_type,
                                                           result.repo_cd.get(),
                                                           result.pipe_c,
                                                           result.pipe_d);

  return result;
}


//===----------------------------------------------------------------------===//
// task_creator Thread Pool Tests
//===----------------------------------------------------------------------===//

TEST_CASE("task_creator thread pool starts and stops", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  SECTION("Creator starts not running") { REQUIRE_FALSE(creator.is_running()); }

  SECTION("start_thread_pool creates threads")
  {
    creator.start_thread_pool();
    REQUIRE(creator.is_running());

    // Give threads time to start
    std::this_thread::sleep_for(10ms);

    creator.stop_thread_pool();
    REQUIRE_FALSE(creator.is_running());
  }

  SECTION("stop_thread_pool joins threads gracefully")
  {
    creator.start_thread_pool();

    // Stop should complete without hanging
    auto start_time = std::chrono::steady_clock::now();
    creator.stop_thread_pool();
    auto duration = std::chrono::steady_clock::now() - start_time;

    REQUIRE(duration < std::chrono::seconds(5));
    REQUIRE_FALSE(creator.is_running());
  }
}

TEST_CASE("task_creator thread pool is idempotent", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  SECTION("Multiple start_thread_pool calls don't create extra threads")
  {
    creator.start_thread_pool();
    creator.start_thread_pool();
    creator.start_thread_pool();

    REQUIRE(creator.is_running());

    creator.stop_thread_pool();
  }

  SECTION("Multiple stop_thread_pool calls don't crash")
  {
    creator.start_thread_pool();

    REQUIRE_NOTHROW(creator.stop_thread_pool());
    REQUIRE_NOTHROW(creator.stop_thread_pool());
    REQUIRE_NOTHROW(creator.stop_thread_pool());
  }

  SECTION("Can restart after stop")
  {
    creator.start_thread_pool();
    creator.stop_thread_pool();
    REQUIRE_FALSE(creator.is_running());

    creator.start_thread_pool();
    REQUIRE(creator.is_running());

    creator.stop_thread_pool();
  }
}

TEST_CASE("task_creator destructor stops thread pool", "[task_creator]")
{
  test_fixture fixture;

  {
    testable_task_creator creator(2,
                                  fixture.pipeline_map,
                                  *fixture.con.context,
                                  fixture.pipeline_exec,
                                  *fixture.memory_manager);
    creator.start_thread_pool();
    // Destructor should stop threads
  }

  // If we get here without hanging, the destructor worked
  SUCCEED("Destructor completed without hanging");
}

//===----------------------------------------------------------------------===//
// get_operator_for_next_task Tests
//===----------------------------------------------------------------------===//

TEST_CASE("get_operator_for_next_task for operator with data returns the operator",
          "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  // Create the source operator that we will call process_next_task on
  auto source_op = std::make_unique<mock_sirius_physical_operator>();

  // Create the hint operator that should be scheduled
  auto hint_op = std::make_unique<mock_sirius_physical_operator>();

  // Create a data repository for the port
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Set up the hint operator with a "default" port that has a dest_pipeline
  // For this test, we'll set the dest_pipeline to nullptr since we're testing
  // through the testable_task_creator which captures what gets scheduled
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    *hint_op,
    "default",
    MemoryBarrierType::PIPELINE,
    data_repo.get(),
    nullptr,  // src_pipeline
    nullptr   // dest_pipeline - will be captured by schedule()
  );

  // Configure source_op to return hint_op as the hint
  source_op->set_custom_hint(
    task_creation_hint{.hint = TaskCreationHint::READY, .producer = hint_op.get()});

  // Call process_next_task - this should attempt to schedule with hint_op
  // Note: This will try to access hint_op->get_port("default")->dest_pipeline
  // which we've set up above
  auto next_op = creator.get_operator_for_next_task(source_op.get());

  REQUIRE(next_op == hint_op.get());

  // // Verify that schedule was called with the hint_op
  // auto scheduled_nodes = creator.get_scheduled_nodes();
  // REQUIRE(creator.get_schedule_count() == 1);
  // REQUIRE(scheduled_nodes.size() == 1);
  // REQUIRE(scheduled_nodes[0] == hint_op.get());
}

TEST_CASE("process_next_task with 4 pipeline graph all FULL barriers", "[task_creator]")
{
test_fixture fixture;

// Create connected pipelines with specific configurations
auto result = create_connected_pipelines(fixture,
                                        MemoryBarrierType::FULL,
                                        2,  // 2 batches in port_ab
                                        MemoryBarrierType::FULL,
                                        1,  // 1 batch in port_bd
                                        MemoryBarrierType::FULL,
                                        3);  // 3 batches in port_cd
// none of the pipelines are finished. 
// all operators next task should point at the scan operators A or C
result.pipe_a->set_finished(false);
result.pipe_b->set_finished(false);
result.pipe_c->set_finished(false);
result.pipe_d->set_finished(false);


testable_task_creator creator(
  2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

auto next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == result.op_a.get());
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == result.op_a.get());
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == result.op_c.get());
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_c.get());

// now set the pipeline A to finished
result.pipe_a->set_finished(true);
result.op_a->set_exhausted(true);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == result.op_b.get());
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == result.op_c.get());
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_c.get());

// now set the pipeline C to finished
result.pipe_c->set_finished(true);
result.op_c->set_exhausted(true);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == result.op_b.get());
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_b.get());

// WSM TODO: pop data form op b and see

}



// TEST_CASE("process_next_task with pipeline hint recurses to inner operator", "[task_creator]")
// {
//   test_fixture fixture;

//   // This test verifies the recursive behavior when the hint is a sirius_pipeline.
//   // When hint is a pipeline, process_next_task should call itself with
//   // pipeline->GetInnerOperators()[0]
//   //
//   // Since sirius_pipeline requires sirius_engine and complex setup, we test this
//   // behavior indirectly by verifying that the source operator's
//   // get_next_task_hint() is called and the scheduling logic follows through.

//   testable_task_creator creator(
//     2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
//     *fixture.memory_manager);

//   // Create an operator chain: source_op returns a custom hint that is monostate
//   // (simulating the end of recursion)
//   auto source_op = std::make_unique<mock_sirius_physical_operator>();
//   source_op->set_custom_hint(sirius::creator::task_creation_hint(std::monostate{}));

//   // Call process_next_task - with monostate and no priority_scans, nothing scheduled
//   creator.process_next_task(source_op.get());
//   REQUIRE(creator.get_schedule_count() == 0);

//   // Now test with an operator hint that returns itself (simulating ready operator)
//   auto ready_op  = std::make_unique<mock_sirius_physical_operator>();
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Set up the port so get_port("default") works
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     *ready_op, "default", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

//   // Configure ready_op to return itself as hint (all ports ready)
//   ready_op->set_custom_hint(sirius::creator::task_creation_hint(ready_op.get()));

//   creator.clear_scheduled();
//   creator.process_next_task(ready_op.get());

//   // Should have scheduled the ready_op
//   REQUIRE(creator.get_schedule_count() == 1);
//   auto nodes = creator.get_scheduled_nodes();
//   REQUIRE(nodes.size() == 1);
//   REQUIRE(nodes[0] == ready_op.get());
// }
// WSM TODO continue to fix tests
// TEST_CASE("process_next_task operator hint follows dest_pipeline", "[task_creator]")
// {
//   test_fixture fixture;

//   testable_task_creator creator(
//     2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
//     *fixture.memory_manager);

//   // Create operators
//   auto source_op = std::make_unique<mock_sirius_physical_operator>();
//   auto target_op = std::make_unique<mock_sirius_physical_operator>();

//   // Create data repository
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Set up target_op with a default port
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     *target_op, "default", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

//   // Source returns target as hint
//   source_op->set_custom_hint(sirius::creator::task_creation_hint(target_op.get()));

//   creator.process_next_task(source_op.get());

//   // Verify target_op was scheduled
//   auto nodes = creator.get_scheduled_nodes();
//   REQUIRE(nodes.size() == 1);
//   REQUIRE(nodes[0] == target_op.get());
// }

// TEST_CASE("process_next_task hint traversal chain", "[task_creator]")
// {
//   test_fixture fixture;

//   testable_task_creator creator(
//     2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
//     *fixture.memory_manager);

//   // Test a chain where:
//   // op1 returns hint pointing to op2
//   // op2 will be scheduled

//   auto op1       = std::make_unique<mock_sirius_physical_operator>();
//   auto op2       = std::make_unique<mock_sirius_physical_operator>();
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Set up op2 with default port
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     *op2, "default", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

//   // op1 returns op2 as hint
//   op1->set_custom_hint(sirius::creator::task_creation_hint(op2.get()));

//   creator.process_next_task(op1.get());

//   // op2 should be scheduled
//   auto nodes = creator.get_scheduled_nodes();
//   REQUIRE(nodes.size() == 1);
//   REQUIRE(nodes[0] == op2.get());
// }

// TEST_CASE("task_creator start/stop lifecycle", "[task_creator]")
// {
//   test_fixture fixture;

//   testable_task_creator creator(
//     2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
//     *fixture.memory_manager);

//   SECTION("start() calls start_thread_pool()")
//   {
//     creator.start();
//     REQUIRE(creator.is_running());
//     creator.stop();
//     REQUIRE_FALSE(creator.is_running());
//   }

//   SECTION("stop() calls stop_thread_pool()")
//   {
//     creator.start();
//     creator.stop();
//     REQUIRE_FALSE(creator.is_running());
//   }
// }

// TEST_CASE("task_creator get_next_task_id increments", "[task_creator]")
// {
//   test_fixture fixture;

//   testable_task_creator creator(
//     1, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
//     *fixture.memory_manager);

//   // The task_id is protected, but we can verify behavior indirectly
//   // by checking that the creator can be constructed and used
//   REQUIRE_FALSE(creator.is_running());
// }

// //===----------------------------------------------------------------------===//
// // sirius_physical_operator::get_next_task_hint() Tests
// //===----------------------------------------------------------------------===//

// TEST_CASE("get_next_task_hint returns monostate when no ports", "[get_next_task_hint]")
// {
//   // An operator with no ports should return monostate
//   mock_sirius_physical_operator op;

//   auto hint = op.get_next_task_hint();

//   REQUIRE(std::holds_alternative<std::monostate>(hint));
// }

// TEST_CASE("get_next_task_hint PIPELINE barrier with empty repo returns src_pipeline",
//           "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator op;
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Create a mock src_pipeline - we use nullptr for simplicity but wrap in shared_ptr
//   // In real usage this would be a valid pipeline
//   auto src_pipeline = duckdb::shared_ptr<sirius_pipeline>(nullptr);

//   // Create a simple mock pipeline to return as src_pipeline hint
//   // Since we can't easily create a real sirius_pipeline, we'll test the path
//   // where src_pipeline is nullptr (returns monostate)
//   mock_pipeline_builder::setup_operator_with_pipeline_port(op,
//                                                            "input",
//                                                            MemoryBarrierType::PIPELINE,
//                                                            data_repo.get(),
//                                                            nullptr,  // src_pipeline is nullptr
//                                                            nullptr   // dest_pipeline
//   );

//   // repo is empty, src_pipeline is nullptr → should return monostate
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<std::monostate>(hint));
// }

// TEST_CASE("get_next_task_hint PIPELINE barrier with data returns this", "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator op;
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Add data to the repository so it's not empty
//   auto batch = std::make_shared<cucascade::data_batch>(1, nullptr);
//   data_repo->add_data_batch(batch);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

//   // repo has data → all ports ready → should return this operator
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
//   REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
// }

// TEST_CASE("get_next_task_hint multiple PIPELINE ports all ready returns this",
//           "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // Add data to both repositories
//   data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));
//   data_repo2->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input1", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input2", MemoryBarrierType::PIPELINE, data_repo2.get(), nullptr, nullptr);

//   // Both repos have data → all ports ready → should return this operator
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
//   REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
// }

// TEST_CASE("get_next_task_hint multiple PIPELINE ports one empty returns monostate",
//           "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // Add data only to first repository
//   data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));
//   // data_repo2 is empty

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input1", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(op,
//                                                            "input2",
//                                                            MemoryBarrierType::PIPELINE,
//                                                            data_repo2.get(),
//                                                            nullptr,  // no src_pipeline
//                                                            nullptr);

//   // One repo is empty with no src_pipeline → should return monostate
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<std::monostate>(hint));
// }

// TEST_CASE("get_next_task_hint uses custom hint when set on mock", "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator source_op;
//   mock_sirius_physical_operator target_op;

//   // Set custom hint to return target_op
//   source_op.set_custom_hint(sirius::creator::task_creation_hint(&target_op));

//   auto hint = source_op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
//   REQUIRE(std::get<sirius_physical_operator*>(hint) == &target_op);
// }

// TEST_CASE("get_next_task_hint custom hint monostate", "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator op;

//   // Set custom hint to monostate
//   op.set_custom_hint(sirius::creator::task_creation_hint(std::monostate{}));

//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<std::monostate>(hint));
// }

// TEST_CASE("get_next_task_hint clear_custom_hint falls back to default", "[get_next_task_hint]")
// {
//   mock_sirius_physical_operator op;

//   // Set and then clear custom hint
//   op.set_custom_hint(sirius::creator::task_creation_hint(&op));
//   op.clear_custom_hint();

//   // With no ports and no custom hint, should return monostate (default behavior)
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<std::monostate>(hint));
// }

// //===----------------------------------------------------------------------===//
// // FULL MemoryBarrierType Tests
// //===----------------------------------------------------------------------===//

// TEST_CASE("get_next_task_hint FULL barrier with unfinished pipeline returns src_pipeline",
//           "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Create a mock pipeline that is NOT finished
//   auto mock_pipeline = fixture.create_mock_pipeline();
//   mock_pipeline->set_finished(false);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op,
//     "input",
//     MemoryBarrierType::FULL,
//     data_repo.get(),
//     mock_pipeline,  // src_pipeline is not finished
//     nullptr);

//   // src_pipeline is not finished → should return the src_pipeline
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
//   REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == mock_pipeline);
// }

// TEST_CASE("get_next_task_hint FULL barrier with finished pipeline returns this",
//           "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo = std::make_unique<cucascade::shared_data_repository>();

//   // Create a mock pipeline that IS finished
//   auto mock_pipeline = fixture.create_mock_pipeline();
//   mock_pipeline->set_finished(true);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op,
//     "input",
//     MemoryBarrierType::FULL,
//     data_repo.get(),
//     mock_pipeline,  // src_pipeline is finished
//     nullptr);

//   // src_pipeline is finished → all ports ready → should return this operator
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
//   REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
// }

// TEST_CASE("get_next_task_hint multiple FULL barriers all finished returns this",
//           "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // Create two mock pipelines that are both finished
//   auto mock_pipeline1 = fixture.create_mock_pipeline();
//   auto mock_pipeline2 = fixture.create_mock_pipeline();
//   mock_pipeline1->set_finished(true);
//   mock_pipeline2->set_finished(true);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input1", MemoryBarrierType::FULL, data_repo1.get(), mock_pipeline1, nullptr);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input2", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline2, nullptr);

//   // Both pipelines finished → all ports ready → should return this operator
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
//   REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
// }

// TEST_CASE("get_next_task_hint multiple FULL barriers one unfinished returns src_pipeline",
//           "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // Create two mock pipelines: one finished, one not
//   auto mock_pipeline1 = fixture.create_mock_pipeline();
//   auto mock_pipeline2 = fixture.create_mock_pipeline();
//   mock_pipeline1->set_finished(true);
//   mock_pipeline2->set_finished(false);  // This one is not finished

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input1", MemoryBarrierType::FULL, data_repo1.get(), mock_pipeline1, nullptr);

//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "input2", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline2, nullptr);

//   // One pipeline is not finished → should return that unfinished pipeline
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
//   REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == mock_pipeline2);
// }

// TEST_CASE("get_next_task_hint mixed PIPELINE and FULL barriers", "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // Add data to PIPELINE port's repo
//   data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));

//   // Create a mock pipeline that is finished for FULL barrier
//   auto mock_pipeline = fixture.create_mock_pipeline();
//   mock_pipeline->set_finished(true);

//   // PIPELINE barrier with data
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "pipeline_input", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

//   // FULL barrier with finished pipeline
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "full_input", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline, nullptr);

//   // Both ports are ready (PIPELINE has data, FULL is finished) → return this
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
//   REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
// }

// TEST_CASE("get_next_task_hint mixed barriers with FULL unfinished", "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // Add data to PIPELINE port's repo
//   data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));

//   // Create a mock pipeline that is NOT finished for FULL barrier
//   auto mock_pipeline = fixture.create_mock_pipeline();
//   mock_pipeline->set_finished(false);

//   // PIPELINE barrier with data (ready)
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "pipeline_input", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

//   // FULL barrier with unfinished pipeline (not ready)
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "full_input", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline, nullptr);

//   // FULL port is not ready → should return the unfinished src_pipeline
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
//   REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == mock_pipeline);
// }

// TEST_CASE("get_next_task_hint mixed barriers with PIPELINE empty", "[get_next_task_hint]")
// {
//   test_fixture fixture;
//   mock_sirius_physical_operator op;
//   auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
//   auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

//   // data_repo1 is empty for PIPELINE barrier

//   // Create a mock pipeline for PIPELINE's src_pipeline (to return when empty)
//   auto pipeline_src = fixture.create_mock_pipeline();

//   // Create a mock pipeline that is finished for FULL barrier
//   auto full_src = fixture.create_mock_pipeline();
//   full_src->set_finished(true);

//   // PIPELINE barrier with empty repo (not ready) - has src_pipeline to return
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op,
//     "pipeline_input",
//     MemoryBarrierType::PIPELINE,
//     data_repo1.get(),
//     pipeline_src,  // src_pipeline to return when empty
//     nullptr);

//   // FULL barrier with finished pipeline (ready)
//   mock_pipeline_builder::setup_operator_with_pipeline_port(
//     op, "full_input", MemoryBarrierType::FULL, data_repo2.get(), full_src, nullptr);

//   // PIPELINE port is empty → should return its src_pipeline
//   auto hint = op.get_next_task_hint();
//   REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
//   REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == pipeline_src);
// }

// //===----------------------------------------------------------------------===//
// // Concurrent Operation Tests
// //===----------------------------------------------------------------------===//

// TEST_CASE("task_creator handles concurrent schedule calls", "[task_creator]")
// {
//   test_fixture fixture;

//   testable_task_creator creator(
//     4, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
//     *fixture.memory_manager);

//   const int num_calls = 100;
//   std::atomic<int> completed{0};

//   // Create mock operators
//   std::vector<std::unique_ptr<mock_sirius_physical_operator>> operators;
//   for (int i = 0; i < num_calls; ++i) {
//     operators.push_back(std::make_unique<mock_sirius_physical_operator>());
//   }

//   // Spawn threads to call process_next_task concurrently
//   std::vector<std::thread> threads;
//   for (int i = 0; i < num_calls; ++i) {
//     threads.emplace_back([&creator, &operators, i, &completed]() {
//       creator.process_next_task(operators[i].get());
//       completed.fetch_add(1);
//     });
//   }

//   // Wait for all threads
//   for (auto& t : threads) {
//     t.join();
//   }

//   REQUIRE(completed.load() == num_calls);
// }

//===----------------------------------------------------------------------===//
// Connected Pipelines Utility Tests
//===----------------------------------------------------------------------===//

TEST_CASE("create_connected_pipelines utility creates correct structure", "[task_creator][utility]")
{
  test_fixture fixture;

  // Create connected pipelines with specific configurations
  auto result = create_connected_pipelines(fixture,
                                          MemoryBarrierType::PIPELINE,
                                          2,  // 2 batches in port_ab
                                          MemoryBarrierType::FULL,
                                          1,  // 1 batch in port_bd
                                          MemoryBarrierType::PARTIAL,
                                          3);  // 3 batches in port_cd

  // Verify we have 4 pipelines
  REQUIRE(result.pipe_a != nullptr);
  REQUIRE(result.pipe_b != nullptr);
  REQUIRE(result.pipe_c != nullptr);
  REQUIRE(result.pipe_d != nullptr);

  // Verify we have 4 operators
  REQUIRE(result.op_a != nullptr);
  REQUIRE(result.op_b != nullptr);
  REQUIRE(result.op_c != nullptr);
  REQUIRE(result.op_d != nullptr);

  // Verify op_b has one port (port_ab)
  auto op_b_ports = result.op_b->get_port_ids();
  REQUIRE(op_b_ports.size() == 1);
  REQUIRE(std::string(op_b_ports[0]) == "port_ab");

  // Verify op_d has two ports (port_bd and port_cd)
  auto op_d_ports = result.op_d->get_port_ids();
  REQUIRE(op_d_ports.size() == 2);
  std::vector<std::string> port_names;
  for (const auto& port_id : op_d_ports) {
    port_names.push_back(std::string(port_id));
  }
  REQUIRE(std::find(port_names.begin(), port_names.end(), "port_bd") != port_names.end());
  REQUIRE(std::find(port_names.begin(), port_names.end(), "port_cd") != port_names.end());

  // Verify op_b's pipeline is pipe_b (through port dest_pipeline)
  REQUIRE(result.op_b->get_pipeline() == result.pipe_b);

  // Verify op_d's pipeline is pipe_d
  REQUIRE(result.op_d->get_pipeline() == result.pipe_d);

  // Verify data repositories have correct number of batches
  REQUIRE(result.repo_ab->size() == 2);
  REQUIRE(result.repo_bd->size() == 1);
  REQUIRE(result.repo_cd->size() == 3);
}
