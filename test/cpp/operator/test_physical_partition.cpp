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
 
 #include <catch.hpp>
 #include <op/sirius_physical_partition.hpp>
 #include "op/sirius_physical_grouped_aggregate_merge.hpp"
 
 #include <duckdb.hpp>
 #include <duckdb/catalog/catalog.hpp>
 #include <duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp>
 #include <duckdb/parser/query_error_context.hpp>
 #include <duckdb/planner/expression/bound_aggregate_expression.hpp>
 #include <duckdb/planner/expression/bound_reference_expression.hpp>
 
 #include <numeric>
 
 using namespace duckdb;
 using namespace sirius::op;
 using namespace cucascade;
 using namespace cucascade::memory;
 
 namespace {
 
 using namespace sirius::test::operator_utils;
 }  // namespace

 // Helper to create a dummy AggregateFunction since we only need the name and types for the GPU
// operator
AggregateFunction MakeDummyAggregate(const std::string& name,
  const duckdb::vector<LogicalType>& args,
  const LogicalType& ret_type)
{
return AggregateFunction(
name, args, ret_type, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}
 
 TEMPLATE_TEST_CASE("sirius_physical_partition partitions data_batch with single partition key",
                    "[physical_partition]",
                    int32_t,
                    int64_t,
                    float,
                    double,
                    int16_t,
                    bool,
                    decimal64_tag,
                    string_tag,
                    timestamp_us_tag,
                    date32_tag)
 {
   using Traits = gpu_type_traits<TestType>;
 
   auto memory_manager = initialize_memory_manager();
   auto* space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
   REQUIRE(space != nullptr);

   std::size_t num_values = 10000;

   std::vector<typename Traits::type> values(num_values);
   if constexpr (Traits::is_string) {
     std::vector<std::string> string_values = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
     for (int i = 0; i < num_values; ++i) {
       values[i] = string_values[i % string_values.size()];
     }     
   } else if constexpr (Traits::is_decimal) {
     for (int i = 0; i < num_values; ++i) {
       values[i] = static_cast<typename Traits::type>(i * 100);
     }
   } else if constexpr (Traits::is_ts) {
     for (int i = 0; i < num_values; ++i) {
       values[i] = static_cast<typename Traits::type>(i * 1'000'000);
     }
   } else if constexpr (std::is_same_v<typename Traits::type, int32_t> ||
                        std::is_same_v<typename Traits::type, int64_t> ||
                        std::is_same_v<typename Traits::type, int16_t>) {
     std::iota(values.begin(), values.end(), static_cast<typename Traits::type>(0));
   } else if constexpr (std::is_same_v<typename Traits::type, float> ||
                        std::is_same_v<typename Traits::type, double>) {
     for (int i = 0; i < num_values; ++i) {
       values[i] = static_cast<typename Traits::type>(i);
     }
   } else if constexpr (std::is_same_v<typename Traits::type, bool>) {
     for (int i = 0; i < num_values; ++i) {
       values[i] = (i % 2 == 0);
     }
   }
   std::shared_ptr<data_batch> input_batch0; // batch for the aggregation key
   if constexpr (Traits::is_string) {
     input_batch0 = make_string_batch(*space, values);
   } else if constexpr (Traits::is_decimal) {
     input_batch0 = make_decimal64_batch(*space, values, Traits::scale);
   } else if constexpr (Traits::is_ts) {
     input_batch0 = make_timestamp_batch(*space, values, Traits::cudf_type);
   } else {
     input_batch0 = make_numeric_batch<typename Traits::type>(*space, values, Traits::cudf_type);
   }

   // create a batch for the aggregation value to just be a int32_t
   std::shared_ptr<data_batch> input_batch1;
   input_batch1 = make_numeric_batch<int32_t>(*space, std::vector<int32_t>(num_values, 1), cudf::type_id::INT32);

   // Concatenate the two batches horizontally (side by side columns)
   auto input_batch = concatenate_batches_horizontal({input_batch0, input_batch1}, *space);

   std::cout<<"gen input batches"<<std::endl;
   std::size_t estimated_cardinality = 100000000; // 100 million rows = PARTITION_SIZE x 10
 
   // Create DuckDB context for aggregate function binding
   duckdb::DuckDB db(nullptr);
   duckdb::Connection con(db);
   auto& context = *con.context;

   std::cout<<"gen context"<<std::endl;

   // Create output types: [group_by_column_type, sum_result_type]
   duckdb::vector<duckdb::LogicalType> agg_output_types;
   agg_output_types.push_back(Traits::logical_type()); // group by key type
   agg_output_types.push_back(Traits::logical_type()); // sum result type

   std::cout<<"gen agg output types"<<std::endl;

   // Create group by expression: GROUP BY column 0
   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
   groups.push_back(
     duckdb::make_uniq<duckdb::BoundReferenceExpression>(Traits::logical_type(), 0));

   std::cout<<"gen groups"<<std::endl;

   // Create aggregate expression: SUM(column 0)
   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
   
       
   std::cout<<"gen sum function"<<std::endl;

   // Create children for the aggregate (the column to sum)
   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> sum_children;
   sum_children.push_back(
     duckdb::make_uniq<duckdb::BoundReferenceExpression>(Traits::logical_type(), 1));
   
   std::cout<<"gen sum children"<<std::endl;

   AggregateFunction sum_function = MakeDummyAggregate("sum", {Traits::logical_type()}, Traits::logical_type());

   // Create the BoundAggregateExpression for SUM
   auto sum_aggregate = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
     sum_function,
     std::move(sum_children),
     nullptr, // filter
     nullptr, // bind_info
     duckdb::AggregateType::NON_DISTINCT);
   
   std::cout<<"gen sum aggregate"<<std::endl;

   aggregates.push_back(std::move(sum_aggregate));

   std::cout<<"finish arg setup"<<std::endl;

   // Create the grouped aggregate merge operator
   sirius_physical_grouped_aggregate_merge grouped_aggregator(
     context,
     std::move(agg_output_types),
     std::move(aggregates),
     std::move(groups),
     estimated_cardinality);

   // Create partitioner types (just the group by key for partitioning)
   duckdb::vector<duckdb::LogicalType> partitioner_types = agg_output_types;
   
   sirius_physical_partition partitioner(
     std::move(partitioner_types),
     estimated_cardinality,
     &grouped_aggregator,
     false);

   std::cout<<"finish grouped aggregator"<<std::endl;
 
   auto outputs = partitioner.execute({input_batch});

   std::cout<<"finish partitioner"<<std::endl;

   std::size_t partition_size = 10000000; // from sirius_physical_partition.hpp

   std::size_t expected_num_partitions = (estimated_cardinality + partition_size - 1) / partition_size;

   REQUIRE(outputs.size() == expected_num_partitions);

   // count the number of rows in each output and make sure it's the same and the initial inputs
   std::size_t total_num_rows = 0;
   for (auto& output : outputs) {
    total_num_rows += output->get_data()->cast<gpu_table_representation>().get_table().num_rows();
   }
   REQUIRE(total_num_rows == num_values);
   

 }