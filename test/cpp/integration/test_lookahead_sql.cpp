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
#include "operator/mgpu_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <filesystem>
#include <fstream>
#include <string>

TEST_CASE("lookahead policies preserve pinned join and UNION results",
          "[integration][isolated_context][lookahead_sql]")
{
  if (!sirius::test::has_gpus(1)) return;
  sirius::test::scratch_dir scratch("lookahead_sql");
  auto surface = std::filesystem::path(scratch.file("surface"));
  sirius::test::mgpu::generate_parquet_surface(
    surface, "SELECT range::INTEGER AS k, (range*2)::BIGINT AS v FROM range(20000)", 2);
  auto glob = sirius::test::mgpu::parquet_glob(surface);
  for (auto const* policy : {"active", "unguarded", "guarded"}) {
    INFO("policy=" << policy);
    auto yaml = std::filesystem::path(scratch.file(std::string(policy) + ".yaml"));
    {
      std::ofstream out(yaml);
      out << "sirius:\n"
             "  topology: { num_gpus: 1 }\n"
             "  memory:\n"
             "    gpu: { usage_limit_bytes: 256Mi }\n"
             "    host: { capacity_bytes: 512Mi, initial_number_pools: 1, pool_size: 8, "
             "block_size: 1048576 }\n"
             "  executor:\n"
             "    task_creator:\n"
          << "      strategy: " << (std::string(policy) == "active" ? "active" : "lookahead")
          << "\n"
          << "      lookahead_wait_for_dynamic_filters: "
          << (std::string(policy) == "unguarded" ? "false" : "true")
          << "\n"
             "  operator_params:\n"
             "    scan_task_batch_size: 64Ki\n"
             "  telemetry: { enable_quent: false }\n";
      REQUIRE(out.good());
    }
    sirius::test::mgpu::scoped_mgpu_env env(yaml);
    auto con = env.make_connection();
    for (auto const* name : {"a", "b", "c"}) {
      auto view = con.Query("CREATE VIEW " + std::string(name) + " AS SELECT * FROM read_parquet(" +
                            sirius::test::sql_literal(glob) + ");");
      REQUIRE(view);
      REQUIRE_FALSE(view->HasError());
    }
    auto pin =
      con.Query("CALL pin_table(" + sirius::test::sql_literal(glob) + ", tier='host', name='a');");
    REQUIRE(pin);
    REQUIRE_FALSE(pin->HasError());
    for (auto const* sql :
         {"SELECT SUM(a.v) FROM a JOIN b ON a.k=b.k JOIN c ON b.k=c.k WHERE c.k<64",
          "SELECT SUM(a.v) FROM a JOIN b ON a.k=b.k WHERE b.k>=20000",
          "SELECT SUM(v) AS s FROM a WHERE k<64 UNION ALL SELECT SUM(v) AS s FROM b WHERE "
          "k>=19936"}) {
      sirius::test::mgpu::require_gpu_matches_cpu(con, sql, /*force_cpu_reference=*/true);
    }
    auto unpin = con.Query("CALL unpin_table('a');");
    REQUIRE(unpin);
    REQUIRE_FALSE(unpin->HasError());
  }
}
