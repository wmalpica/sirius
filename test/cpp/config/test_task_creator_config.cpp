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
#include "sirius_config.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <source_location>
#include <string>
namespace fs = std::filesystem;

TEST_CASE("Sirius configuration exposes task creation strategy", "[sirius][config][lookahead]")
{
  auto const data_dir =
    fs::path(std::source_location::current().file_name()).parent_path() / "data";
  SECTION("invalid strategy is rejected")
  {
    REQUIRE_THROWS(
      sirius::parsed_sirius_config::from_file(data_dir / "invalid_task_creator_strategy.yaml"));
  }
  SECTION("omitted strategy preserves demand-driven creation")
  {
    auto parsed = sirius::parsed_sirius_config::from_file(
      data_dir / "valid_task_creator_strategy_omitted.yaml");
    CHECK(parsed.get_task_creator_config().strategy == sirius::creator::request_type::active);
    CHECK(parsed.get_task_creator_config().lookahead_wait_for_dynamic_filters);
    CHECK(parsed.get_task_creator_config().priority == sirius::creator::priority_order::source);
  }
  SECTION("active and guarded lookahead are accepted")
  {
    auto active =
      sirius::parsed_sirius_config::from_file(data_dir / "valid_task_creator_active.yaml");
    auto lookahead =
      sirius::parsed_sirius_config::from_file(data_dir / "valid_task_creator_lookahead.yaml");
    CHECK(active.get_task_creator_config().strategy == sirius::creator::request_type::active);
    CHECK(lookahead.get_task_creator_config().strategy == sirius::creator::request_type::lookahead);
    CHECK(lookahead.get_task_creator_config().lookahead_wait_for_dynamic_filters);
  }
  SECTION("unguarded lookahead is an explicit experiment")
  {
    auto parsed = sirius::parsed_sirius_config::from_file(
      data_dir / "valid_task_creator_unrestricted_lookahead.yaml");
    CHECK(parsed.get_task_creator_config().strategy == sirius::creator::request_type::lookahead);
    CHECK_FALSE(parsed.get_task_creator_config().lookahead_wait_for_dynamic_filters);
  }
  SECTION("invalid strategy types and filter-wait values are rejected")
  {
    for (auto const* text : {"strategy: []",
                             "strategy: true",
                             "strategy: 42",
                             "lookahead_wait_for_dynamic_filters: maybe",
                             "lookahead_wait_for_dynamic_filters: []"}) {
      auto node = YAML::Load(std::string("sirius:\n  executor:\n    task_creator:\n      ") + text);
      CHECK_THROWS(sirius::parsed_sirius_config::from_node(node));
    }
  }
  SECTION("priority_order remains unsupported")
  {
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(
                          data_dir / "invalid_task_creator_priority_order.yaml"),
                        Catch::Matchers::ContainsSubstring("priority_order") &&
                          Catch::Matchers::ContainsSubstring("removed"));
  }
}
