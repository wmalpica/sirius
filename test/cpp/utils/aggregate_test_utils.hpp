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

#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <string>
#include <vector>

namespace sirius {
namespace test {

/**
 * @brief Result structure for create_aggregate_expressions
 */
struct AggregateExpressionResult {
  duckdb::vector<duckdb::LogicalType> output_types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
};

/**
 * @brief Helper to create a dummy AggregateFunction for testing
 * 
 * Creates a minimal AggregateFunction with just name and types,
 * suitable for GPU operator testing where full aggregate logic isn't needed.
 */
inline duckdb::AggregateFunction MakeDummyAggregate(
  const std::string& name,
  const duckdb::vector<duckdb::LogicalType>& args,
  const duckdb::LogicalType& ret_type)
{
  return duckdb::AggregateFunction(
    name, args, ret_type, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

/**
 * @brief Create DuckDB aggregate expressions for testing
 *
 * This utility function creates the necessary DuckDB expressions for grouped aggregations,
 * which are commonly used in physical operator tests.
 *
 * @tparam Traits Type traits class providing logical_type() method
 * @param group_indexes Column indices for GROUP BY expressions
 * @param aggregations Names of aggregation functions (e.g., "sum", "count", "avg")
 * @param agg_indexes Column indices for aggregation input expressions
 * @return AggregateExpressionResult containing output types, group expressions, and aggregate expressions
 */
template <typename Traits>
AggregateExpressionResult create_aggregate_expressions(
  const std::vector<std::size_t>& group_indexes,
  const std::vector<std::string>& aggregations,
  const std::vector<std::size_t>& agg_indexes)
{
  AggregateExpressionResult result;

  // Create output types: first the group by column types, then the aggregate result types
  for (std::size_t group_idx : group_indexes) {
    result.output_types.push_back(Traits::logical_type());
  }
  for (std::size_t i = 0; i < aggregations.size(); ++i) {
    result.output_types.push_back(Traits::logical_type());
  }

  // Create group by expressions
  for (std::size_t group_idx : group_indexes) {
    result.groups.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(Traits::logical_type(), group_idx));
  }

  // Create aggregate expressions
  for (std::size_t i = 0; i < aggregations.size(); ++i) {
    const std::string& agg_name = aggregations[i];
    std::size_t agg_idx = agg_indexes[i];

    // Create children for the aggregate (the column to aggregate)
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_children;
    agg_children.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(Traits::logical_type(), agg_idx));

    // Create the dummy aggregate function
    duckdb::AggregateFunction agg_function = 
      MakeDummyAggregate(agg_name, {Traits::logical_type()}, Traits::logical_type());

    // Create the BoundAggregateExpression
    auto agg_expr = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
      agg_function,
      std::move(agg_children),
      nullptr, // filter
      nullptr, // bind_info
      duckdb::AggregateType::NON_DISTINCT);

    result.aggregates.push_back(std::move(agg_expr));
  }

  return result;
}

}  // namespace test
}  // namespace sirius
