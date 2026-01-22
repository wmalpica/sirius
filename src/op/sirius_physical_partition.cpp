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

#include "op/sirius_physical_partition.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n.hpp"

namespace sirius {
namespace op {

sirius_physical_partition::sirius_physical_partition(duckdb::vector<duckdb::LogicalType> types,
                                                     duckdb::idx_t estimated_cardinality,
                                                     sirius_physical_operator* parent_op,
                                                     bool is_build)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARTITION, std::move(types), estimated_cardinality)
{
  _num_partitions = (estimated_cardinality + PARTITION_SIZE - 1) / PARTITION_SIZE;
  _parent_op      = parent_op;
  _is_build       = is_build;
  get_partition_keys_and_type(parent_op, is_build);
}

std::string sirius_physical_partition::get_name() const { return "PARTITION"; }

bool sirius_physical_partition::is_source() const { return true; }

bool sirius_physical_partition::is_sink() const { return true; }

void sirius_physical_partition::get_partition_keys_and_type(sirius_physical_operator* op, bool is_build)
{
  _partition_keys.clear();
  _partition_type = PartitionType::NONE;
  if (op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    _partition_type = PartitionType::HASH;
    auto& hash_join_op = op->Cast<sirius_physical_hash_join>();
    if (is_build) {
      for (duckdb::idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
        auto& condition = hash_join_op.conditions[cond_idx];
        if (condition.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(
            condition.right->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }
    } else {
      for (duckdb::idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
        auto& condition = hash_join_op.conditions[cond_idx];
        if (condition.left->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(condition.left->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }
    }
} else if (op->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    _partition_type = PartitionType::HASH;
    auto& grouped_aggregate_op = op->Cast<sirius_physical_grouped_aggregate>();
    for (duckdb::idx_t i = 0; i < grouped_aggregate_op.groupings.size(); i++) {
      auto& grouping = grouped_aggregate_op.groupings[i];
      for (auto& group_idx : grouped_aggregate_op.grouping_sets[i]) {
        auto& group = grouped_aggregate_op.grouped_aggregate_data.groups[group_idx];
        if (group->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(group->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::ORDER_BY) {
    _partition_type = PartitionType::RANGE;
    auto& order_by_op = op->Cast<sirius_physical_order>();
    for (size_t order_idx = 0; order_idx < order_by_op.orders.size(); order_idx++) {
      auto& expr = order_by_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<duckdb::BoundReferenceExpression>().index);
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::TOP_N) {
    _partition_type = PartitionType::CUSTOM;
    auto& top_n_op = op->Cast<sirius_physical_top_n>();
    for (size_t order_idx = 0; order_idx < top_n_op.orders.size(); order_idx++) {
      auto& expr = top_n_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<duckdb::BoundReferenceExpression>().index);
      }
    }
  }
}

bool sirius_physical_partition::is_build_partition() { return _is_build; }


std::vector<std::shared_ptr<::cucascade::data_batch>> sirius_physical_partition::execute(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches) override {

    std::vector<std::shared_ptr<cucascade::data_batch>>  partitioned_results;
    switch (_partition_type) {
      case PartitionType::HASH:
      partitioned_results = gpu_partition_impl::hash_partition(input_batches, _partition_keys, _num_partitions);
      case PartitionType::RANGE:
        throw std::runtime_error("Range partitioning is not implemented yet");
      case PartitionType::EVENLY:
      partitioned_results = gpu_partition_impl::evenly_partition(input_batches, _partition_keys, _num_partitions);
      case PartitionType::CUSTOM:
        throw std::runtime_error("Custom partitioning is not implemented yet");
      default:
        throw std::runtime_error("Unsupported partition type: " + std::to_string(_partition_type));
  }

  return {};
}

void sirius_physical_partition::sink(const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches) override {
  
  for (auto& batch : output_batches) {
    for (auto& [next_op, port_id] : next_port_after_sink) {
      // the next operator is a partition consumer operator, so we need to push the batch into the specific partition
      auto partition_consumer_op = dynamic_cast<sirius_physical_partition_consumer_operator*>(next_op);
      if (partition_consumer_op) {
        partition_consumer_op->push_data_batch_partitioned(port_id, batch, partition_id);
      } else {
        throw std::runtime_error("Next operator is not a partition consumer operator");
      }
    }
  }

  if (!creator) {
    throw std::runtime_error(
      "sirius_physical_operator creator is null in sink_execute for operator " + get_name());
  }
  if (next_port_after_sink.size() > 0) {
    auto current_pipeline =
      next_port_after_sink[0].first->get_port(next_port_after_sink[0].second)->src_pipeline;
    current_pipeline->update_pipeline_status();
  }
  for (auto& [next_op, port_id] : next_port_after_sink) {
    if (next_op) { creator->process_next_task(next_op); }
  }
}

}  // namespace op
}  // namespace sirius
