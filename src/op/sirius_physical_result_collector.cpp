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

// sirius

#include <data/sirius_converter_registry.hpp>
#include <op/result/host_table_chunk_reader.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_interface.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstdio>

namespace sirius {
namespace op {

sirius_physical_result_collector::sirius_physical_result_collector(
  ::sirius::sirius_prepared_statement_data& data)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::RESULT_COLLECTOR, {duckdb::LogicalType::BOOLEAN}, 0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.sirius_physical_plan),
    names(data.prepared->names)
{
  this->types = data.prepared->types;
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_result_collector::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches)
{
  return input_batches;
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_result_collector::get_children() const
{
  return {plan};
}

void sirius_physical_result_collector::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // operator is a sink, build a pipeline
  sink_state.reset();

  D_ASSERT(children.empty());

  // single operator: the operator becomes the data source of the current pipeline
  auto& state = meta_pipeline.get_state();
  state.set_pipeline_source(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(plan);
}

sirius_physical_materialized_collector::sirius_physical_materialized_collector(
  ::sirius::sirius_prepared_statement_data& data, duckdb::ClientContext& client_ctx)
  : sirius_physical_result_collector(data),
    _client_ctx(client_ctx),
    result_collection(duckdb::make_uniq<duckdb::ColumnDataCollection>(client_ctx, types))
{
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_materialized_collector::get_result(
  duckdb::GlobalSinkState& state)
{
  (void)state;  // Silence unused parameter warning

  auto props = _client_ctx.GetClientProperties();

  std::lock_guard<std::mutex> guard(lock);
  // Return an empty result collection if the result_collection is null (from a move)
  if (!result_collection) {
    result_collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, types);
  }

  return duckdb::make_uniq<duckdb::MaterializedQueryResult>(
    statement_type, properties, names, std::move(result_collection), props);
}

void sirius_physical_materialized_collector::sink(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches)
{
  using host_table_chunk_reader = ::sirius::op::result::host_table_chunk_reader;

  printf("sirius_physical_materialized_collector::sink entry, input_batches.size()=%zu\n",
         input_batches.size());

  if (input_batches.empty()) {
    printf("sirius_physical_materialized_collector::sink early return (empty input)\n");
    return;  // todo(kevin) we should handle this case properly
    throw duckdb::InvalidInputException("[GPUPhysicalMaterializedCollector] input_batches is null");
  }

  auto sink_single_batch = [this](std::shared_ptr<cucascade::data_batch> const& input_batch) {
    printf("sirius_physical_materialized_collector::sink processing single batch\n");
    auto* data = input_batch->get_data();
    std::shared_ptr<cucascade::data_batch> clone_batch;
    if (!data) {
      printf("sirius_physical_materialized_collector::sink ERROR: data_batch has no data\n");
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] data_batch has no data representation");
    }
    printf("sirius_physical_materialized_collector::sink got data, tier=%d\n",
           static_cast<int>(data->get_current_tier()));

    // If data is in GPU tier, convert to HOST tier first
    if (data->get_current_tier() == cucascade::memory::Tier::GPU) {
      printf("sirius_physical_materialized_collector::sink converting GPU to HOST\n");
      // Make the HOST memory reservation
      auto sirius_ctx  = _client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      auto& memory_mgr = sirius_ctx->get_memory_manager();
      /// TODO: Find the closest memory space, not just any memory space, in HOST tier
      auto reservation = memory_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        data->get_size_in_bytes());
      if (!reservation) {
        printf("sirius_physical_materialized_collector::sink ERROR: failed to reserve host memory\n");
        throw duckdb::InternalException(
          "[GPUPhysicalMaterializedCollector] Failed to reserve host memory for result collection");
      }

      // Convert to host representation
      auto& registry      = sirius::converter_registry::get();
      auto& mem_space     = reservation->get_memory_space();
      auto& data_repo_mgr = sirius_ctx->get_data_repository_manager();
      auto next_batch_id  = data_repo_mgr.get_next_data_batch_id();
      clone_batch         = input_batch->clone(next_batch_id);
      printf("sirius_physical_materialized_collector::sink cloned batch, converting to host\n");
      clone_batch->convert_to<cucascade::host_table_representation>(
        registry, &mem_space, rmm::cuda_stream_default);
      data = clone_batch->get_data();
      printf("sirius_physical_materialized_collector::sink GPU->HOST conversion done\n");
    } else if (data->get_current_tier() != cucascade::memory::Tier::HOST) {
      // Data must be in HOST tier (i.e., cannot currently reside in DISK tier)
      printf("sirius_physical_materialized_collector::sink ERROR: tier is not HOST (%d)\n",
             static_cast<int>(data->get_current_tier()));
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] Expected host_table_representation in HOST tier");
    }

    // Only accepting host_table_representations for now
    if (dynamic_cast<cucascade::host_table_representation*>(data) == nullptr) {
      printf("sirius_physical_materialized_collector::sink ERROR: data is not host_table_representation\n");
    }
    assert(dynamic_cast<cucascade::host_table_representation*>(data) != nullptr);

    // Push chunks to result collection
    auto const& host_table = data->cast<cucascade::host_table_representation>();
    // host_table_chunk_reader expects get_host_table() and ->allocation to be non-null;
    // otherwise it will dereference a null unique_ptr (e.g. in column_reader::initialize).
    auto const* ht = host_table.get_host_table().get();
    if (!ht) {
      printf("sirius_physical_materialized_collector::sink ERROR: get_host_table() is null\n");
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] host_table_representation has null get_host_table()");
    }
    if (!ht->allocation) {
      printf("sirius_physical_materialized_collector::sink ERROR: host_table->allocation is null\n");
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] host_table allocation is null (cannot read chunks)");
    }
    printf("sirius_physical_materialized_collector::sink creating chunk_reader, reading chunks\n");
    host_table_chunk_reader chunk_reader(_client_ctx, host_table, types);

    printf("sirius_physical_materialized_collector::sink made chunk_reader \n");

    // Push chunks to result collection
    duckdb::DataChunk chunk;
    int chunk_count = 0;
    while (chunk_reader.get_next_chunk(chunk)) {
      printf("sirius_physical_materialized_collector::sink got chunk %d \n", chunk_count);
      std::lock_guard<std::mutex> guard(lock);
      // Initialize result collection if it is null (from a move)
      if (!result_collection) {
        printf("sirius_physical_materialized_collector::sink (re)initializing result_collection\n");
        result_collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, types);
      }
      result_collection->Append(chunk);
      chunk_count++;
    }
    printf("sirius_physical_materialized_collector::sink finished batch (%d chunks appended)\n",
           chunk_count);
  };

  printf("sirius_physical_materialized_collector::sink calling for_each over %zu batches\n",
         input_batches.size());
  std::for_each(input_batches.begin(), input_batches.end(), sink_single_batch);
  printf("sirius_physical_materialized_collector::sink exit\n");
}

}  // namespace op
}  // namespace sirius
