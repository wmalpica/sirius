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
#include "exec/starrocks_substrait_compat.hpp"
#include "substrait/plan.pb.h"

namespace sp = ::substrait;

TEST_CASE("StarRocks fetch adapter preserves literal limits and offsets", "[starrocks_fetch]")
{
  sp::Plan input;
  auto* fetch = input.add_relations()->mutable_root()->mutable_input()->mutable_fetch();
  fetch->mutable_input()->mutable_read()->mutable_named_table()->add_names("orders");
  fetch->mutable_count_expr()->mutable_literal()->set_i64(7);
  fetch->mutable_offset_expr()->mutable_literal()->set_i64(3);
  sp::Plan output;
  REQUIRE(output.ParseFromString(
    sirius::exec::starrocks_compatible_substrait(input.SerializeAsString())));
  auto const& actual = output.relations(0).root().input().fetch();
  CHECK(actual.count() == 7);
  CHECK(actual.offset() == 3);
  CHECK(actual.input().read().named_table().names(0) == "orders");
}

TEST_CASE("StarRocks fetch adapter handles zero unlimited and invalid bounds", "[starrocks_fetch]")
{
  sp::Plan input;
  auto* fetch = input.add_relations()->mutable_root()->mutable_input()->mutable_fetch();
  SECTION("unset count is unlimited") {}
  SECTION("null count is unlimited")
  {
    fetch->mutable_count_expr()->mutable_literal()->mutable_null();
  }
  SECTION("zero count remains zero") { fetch->mutable_count_expr()->mutable_literal()->set_i64(0); }
  SECTION("negative count fails")
  {
    fetch->mutable_count_expr()->mutable_literal()->set_i64(-1);
    REQUIRE_THROWS(sirius::exec::starrocks_compatible_substrait(input.SerializeAsString()));
    return;
  }
  SECTION("nonliteral count fails")
  {
    fetch->mutable_count_expr()->mutable_scalar_function();
    REQUIRE_THROWS(sirius::exec::starrocks_compatible_substrait(input.SerializeAsString()));
    return;
  }
  auto const zero = fetch->has_count_expr() && fetch->count_expr().literal().has_i64();
  sp::Plan output;
  REQUIRE(output.ParseFromString(
    sirius::exec::starrocks_compatible_substrait(input.SerializeAsString())));
  CHECK(output.relations(0).root().input().fetch().count() == (zero ? 0 : -1));
}
