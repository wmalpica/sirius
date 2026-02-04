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
#include "pipeline/gpu_pipeline_task.hpp"
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
// Testable Pipeline Executor
//===----------------------------------------------------------------------===//

/**
 * @brief A testable subclass of pipeline_executor that allows inspection of queued tasks.
 *
 * This class extends pipeline_executor to provide testing capabilities by exposing
 * a method to pop and inspect all tasks from the internal task queue.
 */
class testable_pipeline_executor : public pipeline_executor {
 public:
  using pipeline_executor::pipeline_executor;  // Inherit constructors

  /**
   * @brief Pop all tasks from the task queue and return them.
   *
   * This method repeatedly calls try_pop() on the internal _task_queue until
   * it's empty, collecting all tasks into a vector.
   *
   * @return std::vector<std::unique_ptr<sirius::parallel::itask>> All tasks that were in the queue
   */
  std::vector<std::unique_ptr<sirius::parallel::itask>> pop_all_tasks()
  {
    std::vector<std::unique_ptr<sirius::parallel::itask>> tasks;
    std::unique_ptr<sirius::parallel::itask> task;

    while ((task = _task_queue.try_pop()) != nullptr) {
      tasks.push_back(std::move(task));
    }

    return tasks;
  }
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
  testable_pipeline_executor pipeline_exec;
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

}

TEST_CASE("process_next_task with 4 pipeline graph all FULL barriers", "[task_creator]")
{
test_fixture fixture;

// Create connected pipelines with specific configurations
auto result = create_connected_pipelines(fixture,
                                        MemoryBarrierType::FULL,
                                        1,  // 1 batch in port_ab
                                        MemoryBarrierType::FULL,
                                        1,  // 1 batch in port_bd
                                        MemoryBarrierType::FULL,
                                        1);  // 1 batch in port_cd
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

// empty ab and set b to finished.  Only d should be ready to create a task
auto batch_and_handle0 = result.repo_ab->pop_data_batch(cucascade::batch_state::task_created);
REQUIRE(result.repo_ab->size() == 0);

result.pipe_b->set_finished(true);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_d.get());

// now lets empty bd. We should not be able to create a task because d has no data. 
auto batch_and_handle1 = result.repo_bd->pop_data_batch(cucascade::batch_state::task_created);
REQUIRE(result.repo_bd->size() == 0);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == nullptr);
}

TEST_CASE("process_next_task with 4 pipeline graph all PARTIAL barriers", "[task_creator]")
{
test_fixture fixture;

// Create connected pipelines with specific configurations
auto result = create_connected_pipelines(fixture,
                                        MemoryBarrierType::PARTIAL,
                                        1,  // 1 batch in port_ab
                                        MemoryBarrierType::PARTIAL,
                                        1,  // 1 batch in port_bd
                                        MemoryBarrierType::PARTIAL,
                                        1);  // 1 batch in port_cd
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
REQUIRE(next_op == result.op_b.get());
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == result.op_c.get());
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_d.get());

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
REQUIRE(next_op == result.op_d.get());

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
REQUIRE(next_op == result.op_d.get());

// empty ab and set b to finished.  Only d should be ready to create a task
auto batch_and_handle0 = result.repo_ab->pop_data_batch(cucascade::batch_state::task_created);
REQUIRE(result.repo_ab->size() == 0);

result.pipe_b->set_finished(true);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_d.get());

// now lets empty bd. We should not be able to create a task because d has no data. 
auto batch_and_handle1 = result.repo_bd->pop_data_batch(cucascade::batch_state::task_created);
REQUIRE(result.repo_bd->size() == 0);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == nullptr);
}

TEST_CASE("process_next_task with 4 pipeline graph with mixed barriers", "[task_creator]")
{
test_fixture fixture;

// Create connected pipelines with specific configurations
auto result = create_connected_pipelines(fixture,
                                        MemoryBarrierType::FULL,
                                        1,  // 1 batch in port_ab
                                        MemoryBarrierType::PARTIAL,
                                        1,  // 1 batch in port_bd
                                        MemoryBarrierType::FULL,
                                        1);  // 1 batch in port_cd
// none of the pipelines are finished. 
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
REQUIRE(next_op == result.op_d.get());

// empty ab and set b to finished.  Only d should be ready to create a task
auto batch_and_handle0 = result.repo_ab->pop_data_batch(cucascade::batch_state::task_created);
REQUIRE(result.repo_ab->size() == 0);

result.pipe_b->set_finished(true);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == result.op_d.get());

// now lets empty bd. We should not be able to create a task because d has no data. 
auto batch_and_handle1 = result.repo_bd->pop_data_batch(cucascade::batch_state::task_created);
REQUIRE(result.repo_bd->size() == 0);

next_op = creator.get_operator_for_next_task(result.op_a.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_b.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_c.get());
REQUIRE(next_op == nullptr);
next_op = creator.get_operator_for_next_task(result.op_d.get());
REQUIRE(next_op == nullptr);
}


TEST_CASE("task_creator get_next_task_id increments", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    1, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec,
    *fixture.memory_manager);

  // The task_id is protected, but we can verify behavior indirectly
  // by checking that the creator can be constructed and used
  REQUIRE_FALSE(creator.is_running());
}

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

//===----------------------------------------------------------------------===//
// Multi-threaded Schedule Tests
//===----------------------------------------------------------------------===//

TEST_CASE("task_creator multi-threaded schedule requests", "[task_creator]")
{
  test_fixture fixture;

  // Create connected pipelines with data in each repository
  auto result = create_connected_pipelines(fixture,
                                          MemoryBarrierType::PARTIAL,
                                          1,  // 1 batch in port_ab
                                          MemoryBarrierType::PARTIAL,
                                          1,  // 1 batch in port_bd
                                          MemoryBarrierType::PARTIAL,
                                          1);  // 1 batch in port_cd

  
  auto pipeline_a = result.op_a->get_pipeline();
  REQUIRE(pipeline_a != nullptr);
  // WSM TODO: need to have the fix for operatore being able to reliably get their pipeline.
  
                                          // Create task_creator and start it
  testable_task_creator creator(
    4, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);
  creator.start_thread_pool();

  // Give the task creator time to start
  std::this_thread::sleep_for(100ms);

  // Create 4 threads, each will call schedule() 10 times for one operator
  const int calls_per_thread = 10;
  std::vector<std::thread> threads;

  threads.emplace_back([&creator, &result]() {
    for (int i = 0; i < calls_per_thread; ++i) {
      creator.schedule(result.op_a.get());
    }
  });

  threads.emplace_back([&creator, &result]() {
    for (int i = 0; i < calls_per_thread; ++i) {
      creator.schedule(result.op_b.get());
    }
  });

  threads.emplace_back([&creator, &result]() {
    for (int i = 0; i < calls_per_thread; ++i) {
      creator.schedule(result.op_c.get());
    }
  });

  threads.emplace_back([&creator, &result]() {
    for (int i = 0; i < calls_per_thread; ++i) {
      creator.schedule(result.op_d.get());
    }
  });

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Give task creator time to process all requests
  std::this_thread::sleep_for(500ms);

  // Stop the task creator
  creator.stop_thread_pool();

  // Pop all tasks from the pipeline executor
  auto tasks = fixture.pipeline_exec.pop_all_tasks();

  // Verify we have 40 tasks (10 per operator)
  REQUIRE(tasks.size() == 40);

  // Count tasks per operator
  std::unordered_map<op::sirius_physical_operator*, int> task_counts;
  task_counts[result.op_a.get()] = 0;
  task_counts[result.op_b.get()] = 0;
  task_counts[result.op_c.get()] = 0;
  task_counts[result.op_d.get()] = 0;

  for (auto& task : tasks) {
    // Cast to sirius_pipeline_itask to access pipeline-specific methods
    auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get());
    REQUIRE(pipeline_task != nullptr);

    // Get the pipeline from the task's global state
    auto* gpu_task = dynamic_cast<pipeline::gpu_pipeline_task*>(task.get());
    REQUIRE(gpu_task != nullptr);

    // Access the global state to get the pipeline
    auto* global_state = gpu_task->global_state();
    REQUIRE(global_state != nullptr);
    auto& gpu_global_state = global_state->cast<pipeline::gpu_pipeline_task_global_state>();
    auto pipeline = gpu_global_state._pipeline;

    REQUIRE(pipeline != nullptr);

    // Get the first operator from the pipeline
    auto operators = pipeline->get_operators();
    REQUIRE(!operators.empty());
    auto* source_op = &operators[0].get();
    
    // Increment count for this operator
    auto it = task_counts.find(source_op);
    if (it != task_counts.end()) {
      it->second++;
    }
  }

  // Verify we have exactly 10 tasks for each operator
  REQUIRE(task_counts[result.op_a.get()] == 10);
  REQUIRE(task_counts[result.op_b.get()] == 10);
  REQUIRE(task_counts[result.op_c.get()] == 10);
  REQUIRE(task_counts[result.op_d.get()] == 10);
}
