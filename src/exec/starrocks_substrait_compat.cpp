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

#include "exec/starrocks_substrait_compat.hpp"

#include "sirius/exception.hpp"
#include "substrait/plan.pb.h"

#include <google/protobuf/message.h>

#include <cstdint>
#include <vector>

namespace sirius::exec {
namespace {
namespace pb = duckdb::google::protobuf;
namespace sp = ::substrait;

std::int64_t bound(const sp::Expression& expression, std::int64_t null_value)
{
  if (!expression.has_literal()) {
    throw sirius::invalid_input_exception("StarRocks fetch bound must be a literal");
  }
  auto const& literal = expression.literal();
  std::int64_t value;
  switch (literal.literal_type_case()) {
    case sp::Expression_Literal::kNull: return null_value;
    case sp::Expression_Literal::kI8: value = literal.i8(); break;
    case sp::Expression_Literal::kI16: value = literal.i16(); break;
    case sp::Expression_Literal::kI32: value = literal.i32(); break;
    case sp::Expression_Literal::kI64: value = literal.i64(); break;
    default: throw sirius::invalid_input_exception("StarRocks fetch bound must be an integer");
  }
  if (value < 0) {
    throw sirius::invalid_input_exception("StarRocks fetch bound must be nonnegative");
  }
  return value;
}

void normalize(pb::Message& message)
{
  if (message.GetDescriptor()->full_name() == "substrait.FetchRel") {
    auto& fetch = static_cast<sp::FetchRel&>(message);
    if (fetch.has_count_expr()) {
      fetch.set_count(bound(fetch.count_expr(), -1));
    } else if (!fetch.has_count()) {
      fetch.set_count(-1);
    }
    if (fetch.has_offset_expr()) { fetch.set_offset(bound(fetch.offset_expr(), 0)); }
  }
  auto const* reflection = message.GetReflection();
  std::vector<const pb::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (auto const* field : fields) {
    if (field->cpp_type() != pb::FieldDescriptor::CPPTYPE_MESSAGE) { continue; }
    if (field->is_repeated()) {
      for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
        normalize(*reflection->MutableRepeatedMessage(&message, field, i));
      }
    } else {
      normalize(*reflection->MutableMessage(&message, field));
    }
  }
}
}  // namespace

std::string starrocks_compatible_substrait(const std::string& bytes)
{
  sp::Plan plan;
  if (!plan.ParseFromString(bytes)) {
    throw sirius::invalid_input_exception("Invalid StarRocks Substrait plan");
  }
  normalize(plan);
  return plan.SerializeAsString();
}
}  // namespace sirius::exec
