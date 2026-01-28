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

 #include "operator_test_utils.hpp"
 #include "operator_type_traits.hpp"
 #include "utils/aggregate_test_utils.hpp"
 #include "utils/data_utils.hpp"
 #include "utils/test_validation_utility.hpp"
 #include "data/data_batch_utils.hpp"

 #include <catch.hpp>
 #include <op/sirius_physical_grouped_aggregate.hpp>
 #include "op/sirius_physical_grouped_aggregate_merge.hpp"
 
 #include <duckdb.hpp>
 #include <duckdb/parser/query_error_context.hpp>
 #include <duckdb/planner/expression/bound_aggregate_expression.hpp>
 #include <duckdb/planner/expression/bound_reference_expression.hpp>
 
 #include <cudf/table/table.hpp>
 
 #include <algorithm>
 #include <numeric>
 
 using namespace duckdb;
 using namespace sirius::op;
 using namespace cucascade;
 using namespace cucascade::memory;
 
namespace {

using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;
}  // namespace
 
 TEMPLATE_TEST_CASE("sirius_physical_grouped_aggregate grouped aggregates data_batch with single partition key, multiple aggregations on numerics",
                    "[physical_grouped_aggregate]",
                    int32_t,
                    int64_t,
                    float,
                    double,
                    int16_t,
                    decimal64_tag)
 {
   using Traits = gpu_type_traits<TestType>;
 
   auto memory_manager = initialize_memory_manager();
   auto* space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
   REQUIRE(space != nullptr);

   
   std::size_t num_groups = 1000;
   std::vector<int32_t> group_sizes(num_groups);
   std::iota(group_sizes.begin(), group_sizes.end(), 1);
   std::size_t total_num_values = std::accumulate(group_sizes.begin(), group_sizes.end(), 0);


   std::vector<typename Traits::type> key_values(total_num_values);
   std::vector<typename Traits::type> value_values(total_num_values);
   // group by column and 3 aggregations: min, max, count
   std::vector<typename Traits::type> expected_group_by(num_groups);
   std::vector<typename Traits::type> expected_min(num_groups);
   std::vector<typename Traits::type> expected_max(num_groups);
   std::vector<int32_t> expected_count(num_groups);
   
   std::size_t offset = 0;
   for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
    std::size_t num_values = group_sizes[group_idx];
    
    // set all group keys to the same value
    std::fill(key_values.begin() + offset, key_values.begin() + offset + num_values, static_cast<typename Traits::type>(group_idx));
    
    // values for each group will go from -group_idx/2 to (group_idx-1)/2
    std::iota(value_values.begin() + offset, value_values.begin() + offset + num_values, static_cast<typename Traits::type>(-group_idx)/2);
    
    // set the expected values
    expected_group_by[group_idx] = group_idx; // key
    expected_min[group_idx] = *std::min_element(value_values.begin() + offset, value_values.begin() + offset + num_values); //min
    expected_max[group_idx] = *std::max_element(value_values.begin() + offset, value_values.begin() + offset + num_values); //max
    expected_count[group_idx] = num_values; //count

    offset += num_values;
  }
  
  auto mr = get_resource_ref(*space);
  auto stream = default_stream();
  
  // Create input columns
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.push_back(vector_to_cudf_column<Traits>(key_values, stream, mr));
  input_columns.push_back(vector_to_cudf_column<Traits>(value_values, stream, mr));

  // Create input table
  std::unique_ptr<cudf::table> input_table = std::make_unique<cudf::table>(std::move(input_columns));
  std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(std::move(input_table), *space);

   // Create DuckDB context for aggregate function binding
   duckdb::DuckDB db(nullptr);
   duckdb::Connection con(db);
   auto& context = *con.context;

   // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
   auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
     {0},      // group_indexes: GROUP BY column 0
     {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
     {1, 1, 1}       // agg_indexes: MIN(column 1), MAX(column 1), COUNT(column 1)
   );

   // Create the grouped aggregate merge operator
   sirius_physical_grouped_aggregate grouped_aggregator(
     context,
     std::move(agg_result.output_types),
     std::move(agg_result.aggregates),
     std::move(agg_result.groups),
     num_groups);
   
   auto outputs = grouped_aggregator.execute({input_batch});

   // Verify we got one output batch
   REQUIRE(outputs.size() == 1);

   // Create expected cuDF table from expected vectors using vector_to_cudf_column
   
   std::vector<std::unique_ptr<cudf::column>> expected_cols;
   
   // Column 0: group key
   expected_cols.push_back(sirius::test::vector_to_cudf_column<Traits>(expected_group_by, stream, mr));
   
   // Column 1: min
   expected_cols.push_back(sirius::test::vector_to_cudf_column<Traits>(expected_min, stream, mr));
   
   // Column 2: max
   expected_cols.push_back(sirius::test::vector_to_cudf_column<Traits>(expected_max, stream, mr));
   
   // Column 3: count (always int32 for count aggregation)
   expected_cols.push_back(sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(expected_count, stream, mr));
   
   // Create expected table
   auto expected_table = std::make_unique<cudf::table>(std::move(expected_cols));
   
   // Compare output with expected using the validation utility
   // Sort both tables before comparison since aggregation order is not guaranteed
   bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
     outputs[0], expected_table->view(), true);
   REQUIRE(tables_match);

}



TEMPLATE_TEST_CASE("sirius_physical_grouped_aggregate grouped aggregates data_batch with multiple partition key, multiple aggregations",
  "[physical_grouped_aggregate]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag,
  string_tag,
  timestamp_us_tag,
  date32_tag)
{
using Traits = gpu_type_traits<TestType>;

auto memory_manager = initialize_memory_manager();
auto* space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
REQUIRE(space != nullptr);


std::size_t num_groups = 1000;
std::vector<int32_t> group_sizes(num_groups);
std::iota(group_sizes.begin(), group_sizes.end(), 1);
std::size_t total_num_values = std::accumulate(group_sizes.begin(), group_sizes.end(), 0);


std::vector<typename Traits::type> key_values0(total_num_values);
std::vector<std::string> key_values1(total_num_values);
std::vector<typename Traits::type> value_values(total_num_values);
// group by column and 3 aggregations: min, max, count
std::vector<typename Traits::type> expected_group_by0(num_groups);
std::vector<std::string> expected_group_by1(num_groups);
std::vector<typename Traits::type> expected_min(num_groups);
std::vector<typename Traits::type> expected_max(num_groups);
std::vector<int32_t> expected_count(num_groups);

std::size_t offset = 0;
for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
std::size_t num_values = group_sizes[group_idx];

// set all group keys to the same value
if constexpr (Traits::is_string) {
  std::fill(key_values0.begin() + offset, key_values0.begin() + offset + num_values, std::to_string(group_idx));
} else {
  std::fill(key_values0.begin() + offset, key_values0.begin() + offset + num_values, static_cast<typename Traits::type>(group_idx));
}
std::fill(key_values1.begin() + offset, key_values1.begin() + offset + num_values, std::to_string(group_idx));

// values for each group will go from -group_idx/2 to (group_idx-1)/2
if constexpr (Traits::is_string) {
  std::fill(value_values.begin() + offset, value_values.begin() + offset + num_values, std::to_string(group_idx));
} else {
  std::iota(value_values.begin() + offset, value_values.begin() + offset + num_values, static_cast<typename Traits::type>(-group_idx)/2);
}

// set the expected values
if constexpr (Traits::is_string) {
  expected_group_by0[group_idx] = std::to_string(group_idx); // key
} else {
  expected_group_by0[group_idx] = group_idx; // key
}
expected_group_by1[group_idx] = std::to_string(group_idx); // key
expected_min[group_idx] = *std::min_element(value_values.begin() + offset, value_values.begin() + offset + num_values); //min
expected_max[group_idx] = *std::max_element(value_values.begin() + offset, value_values.begin() + offset + num_values); //max
expected_count[group_idx] = num_values; //count

offset += num_values;
}

auto mr = get_resource_ref(*space);
auto stream = default_stream();

// Create input columns
std::vector<std::unique_ptr<cudf::column>> input_columns;
input_columns.push_back(vector_to_cudf_column<Traits>(key_values0, stream, mr));
input_columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(key_values1, stream, mr));
input_columns.push_back(vector_to_cudf_column<Traits>(value_values, stream, mr));

// Create input table
std::unique_ptr<cudf::table> input_table = std::make_unique<cudf::table>(std::move(input_columns));
std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(std::move(input_table), *space);

// Create DuckDB context for aggregate function binding
duckdb::DuckDB db(nullptr);
duckdb::Connection con(db);
auto& context = *con.context;

// Create aggregate expressions: GROUP BY column 0, SUM(column 1)
auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
{0, 1},      // group_indexes: GROUP BY column 0 and 1
{"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
{2, 2, 2}       // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
);

// Create the grouped aggregate merge operator
sirius_physical_grouped_aggregate grouped_aggregator(
context,
std::move(agg_result.output_types),
std::move(agg_result.aggregates),
std::move(agg_result.groups),
num_groups);

auto outputs = grouped_aggregator.execute({input_batch});

// Verify we got one output batch
REQUIRE(outputs.size() == 1);

// Create expected cuDF table from expected vectors using vector_to_cudf_column

std::vector<std::unique_ptr<cudf::column>> expected_cols;

// Column 0: group key
expected_cols.push_back(sirius::test::vector_to_cudf_column<Traits>(expected_group_by0, stream, mr));

// Column 1: group key
expected_cols.push_back(sirius::test::vector_to_cudf_column<gpu_type_traits<string_tag>>(expected_group_by1, stream, mr));

// Column 2: min
expected_cols.push_back(sirius::test::vector_to_cudf_column<Traits>(expected_min, stream, mr));

// Column 3: max
expected_cols.push_back(sirius::test::vector_to_cudf_column<Traits>(expected_max, stream, mr));

// Column 4: count (always int32 for count aggregation)
expected_cols.push_back(sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(expected_count, stream, mr));

// Create expected table
auto expected_table = std::make_unique<cudf::table>(std::move(expected_cols));

// Compare output with expected using the validation utility
// Sort both tables before comparison since aggregation order is not guaranteed
bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
outputs[0], expected_table->view(), true);
REQUIRE(tables_match);

}

