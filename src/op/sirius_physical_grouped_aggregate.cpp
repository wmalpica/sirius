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

#include "op/sirius_physical_grouped_aggregate.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"

namespace sirius {
namespace op {

static duckdb::vector<duckdb::LogicalType> create_group_chunk_types(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups)
{
  duckdb::set<duckdb::idx_t> group_indices;

  if (groups.empty()) { return {}; }

  for (auto& group : groups) {
    D_ASSERT(group->type == duckdb::ExpressionType::BOUND_REF);
    auto& bound_ref = group->Cast<duckdb::BoundReferenceExpression>();
    group_indices.insert(bound_ref.index);
  }
  duckdb::idx_t highest_index = *group_indices.rbegin();
  duckdb::vector<duckdb::LogicalType> types(highest_index + 1, duckdb::LogicalType::SQLNULL);
  for (auto& group : groups) {
    auto& bound_ref        = group->Cast<duckdb::BoundReferenceExpression>();
    types[bound_ref.index] = bound_ref.return_type;
  }
  return types;
}

sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_grouped_aggregate(
      context, std::move(types), std::move(expressions), {}, estimated_cardinality)
{
}

sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_grouped_aggregate(context,
                                      std::move(types),
                                      std::move(expressions),
                                      std::move(groups_p),
                                      {},
                                      {},
                                      estimated_cardinality,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)
{
}

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> grouping_functions_p,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType group_validity,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  // get a list of all aggregates to be computed
  const duckdb::idx_t group_count = groups_p.size();
  if (grouping_sets.empty()) {
    duckdb::GroupingSet set;
    for (duckdb::idx_t i = 0; i < group_count; i++) {
      set.insert(i);
    }
    grouping_sets.push_back(std::move(set));
  }
  input_group_types = create_group_chunk_types(groups_p);

  grouped_aggregate_data.InitializeGroupby(
    std::move(groups_p), std::move(expressions), std::move(grouping_functions_p));

  auto& aggregates = grouped_aggregate_data.aggregates;
  // filter_indexes must be pre-built, not lazily instantiated in parallel...
  // Because everything that lives in this class should be read-only at execution time
  idx_t aggregate_input_idx = 0;
  for (idx_t i = 0; i < aggregates.size(); i++) {
    auto& aggregate = aggregates[i];
    auto& aggr      = aggregate->Cast<duckdb::BoundAggregateExpression>();
    aggregate_input_idx += aggr.children.size();
    if (aggr.aggr_type == duckdb::AggregateType::DISTINCT) {
      distinct_filter.push_back(i);
    } else if (aggr.aggr_type == duckdb::AggregateType::NON_DISTINCT) {
      non_distinct_filter.push_back(i);
    } else {  // LCOV_EXCL_START
      throw duckdb::NotImplementedException(
        "AggregateType not implemented in PhysicalHashAggregate");
    }  // LCOV_EXCL_STOP
  }

  for (idx_t i = 0; i < aggregates.size(); i++) {
    auto& aggregate = aggregates[i];
    auto& aggr      = aggregate->Cast<duckdb::BoundAggregateExpression>();
    if (aggr.filter) {
      auto& bound_ref_expr = aggr.filter->Cast<duckdb::BoundReferenceExpression>();
      if (!filter_indexes.count(aggr.filter.get())) {
        // Replace the bound reference expression's index with the corresponding index of the
        // payload chunk
        // TODO: Still not quite sure why duckdb replace the index
        filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
        bound_ref_expr.index              = aggregate_input_idx;
      }
      aggregate_input_idx++;
    }
  }

  distinct_collection_info =
    duckdb::DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

  for (idx_t i = 0; i < grouping_sets.size(); i++) {
    groupings.emplace_back(grouping_sets[i],
                           grouped_aggregate_data,
                           distinct_collection_info,
                           group_validity,
                           distinct_validity);
  }

  // The output of groupby is ordered as the grouping columns first followed by the aggregate
  // columns See RadixHTLocalSourceState::Scan for more details
  idx_t total_output_columns = 0;
  for (auto& aggregate : aggregates) {
    auto& aggr = aggregate->Cast<duckdb::BoundAggregateExpression>();
    total_output_columns++;
  }
  total_output_columns += grouped_aggregate_data.GroupCount();



// Convert the grouped aggregate data to cudf compute definitions
  {
    // 1. Extract group_idx from grouped_aggregate_data.groups
    for (const auto& group : grouped_aggregate_data.groups) {
      D_ASSERT(group->type == duckdb::ExpressionType::BOUND_REF);
      auto& bound_ref = group->Cast<duckdb::BoundReferenceExpression>();
      group_idx.push_back(static_cast<int>(bound_ref.index));
    }

    // 2. Extract aggregates (cudf::aggregation::Kind) from grouped_aggregate_data.aggregates
    for (const auto& aggregate : grouped_aggregate_data.aggregates) {
      auto& aggr = aggregate->Cast<duckdb::BoundAggregateExpression>();
      
      // Convert DuckDB aggregate function name to cudf::aggregation::Kind
      cudf::aggregation::Kind agg_kind;
      if (aggr.function.name == "sum" || aggr.function.name == "sum_no_overflow") {
        agg_kind = cudf::aggregation::Kind::SUM;
      } else if (aggr.function.name == "count") {
        agg_kind = cudf::aggregation::Kind::COUNT_VALID;
      } else if (aggr.function.name == "count_star") {
        agg_kind = cudf::aggregation::Kind::COUNT_ALL;
      } else if (aggr.function.name == "min") {
        agg_kind = cudf::aggregation::Kind::MIN;
      } else if (aggr.function.name == "max") {
        agg_kind = cudf::aggregation::Kind::MAX;
      } else {
        throw std::runtime_error("Unsupported aggregate function: " + aggr.function.name);
      }
      cudf_aggregates.push_back(agg_kind);
      
      // 3. Extract aggregate_idx from the children of the aggregate expression
      if (aggr.children.empty()) {
        // COUNT(*) has no children - use 0 as a placeholder (will be handled by COUNT_ALL)
        if (aggr.function.name == "count_star") {
          cudf_aggregate_idx.push_back(0);
        } else {
          throw std::runtime_error("Unsupported aggregate function: " + aggr.function.name + " with no children");
        }
      } else {
        if (aggr.children.size() == 1) {
          // Extract the column index from the first child (most aggregates have one child)
          D_ASSERT(aggr.children[0]->type == duckdb::ExpressionType::BOUND_REF);
          auto& bound_ref = aggr.children[0]->Cast<duckdb::BoundReferenceExpression>();
          cudf_aggregate_idx.push_back(static_cast<int>(bound_ref.index));
        } else {
          throw std::runtime_error("Unsupported aggregate function: " + aggr.function.name + " with " + std::to_string(aggr.children.size()) + " children");
        }
      }
    }
  }
}

std::vector<std::shared_ptr<::cucascade::data_batch>> sirius_physical_grouped_aggregate::execute(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches) override {

    std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  for (auto& input_batch : input_batches) {
    auto result = gpu_aggregate_impl::local_grouped_aggregate(
      input_batch,
      group_idx,
      cudf_aggregates,
      cudf_aggregate_idx,
      cudf::get_default_stream(),
      *input_batch->get_memory_space()
    );
    results.push_back(std::move(result));
  
  }
  return results;
}
}  // namespace op
}  // namespace sirius
