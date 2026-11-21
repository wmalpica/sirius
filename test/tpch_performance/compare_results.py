#!/usr/bin/env python3
# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Compare DuckDB CLI result tables with float tolerance."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FLOAT_TOLERANCE = 1e-10


@dataclass
class ParsedResult:
    headers: list[str]
    types: list[str]
    rows: list[list[str]]


def _split_box_row(line: str) -> list[str]:
    if not (line.startswith("│") and line.endswith("│")):
        raise ValueError(f"invalid DuckDB boxed row: {line!r}")
    return [cell.strip() for cell in line[1:-1].split("│")]


def parse_result(path: Path) -> ParsedResult:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_rows = [line for line in lines if line.startswith("│") and line.endswith("│")]
    if len(table_rows) < 2:
        raise ValueError(f"{path}: expected DuckDB boxed table output")

    headers = _split_box_row(table_rows[0])
    types = _split_box_row(table_rows[1])
    rows = [_split_box_row(line) for line in table_rows[2:]]

    width = len(headers)
    if len(types) != width:
        raise ValueError(f"{path}: header/type column count mismatch")

    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{path}: row {idx} has {len(row)} columns, expected {width}"
            )

    return ParsedResult(headers=headers, types=types, rows=rows)


def is_float_type(type_name: str) -> bool:
    normalized = type_name.strip().lower()
    return normalized in {"float", "double"}


def compare_results(
    lhs: ParsedResult, rhs: ParsedResult, float_tolerance: float
) -> tuple[bool, str]:
    if lhs.headers != rhs.headers:
        return False, f"header mismatch: {lhs.headers!r} != {rhs.headers!r}"

    if lhs.types != rhs.types:
        return False, f"type mismatch: {lhs.types!r} != {rhs.types!r}"

    if len(lhs.rows) != len(rhs.rows):
        return False, f"row count mismatch: {len(lhs.rows)} != {len(rhs.rows)}"

    float_columns = [is_float_type(type_name) for type_name in lhs.types]
    lhs_rows = sorted(lhs.rows)
    rhs_rows = sorted(rhs.rows)

    for row_idx, (lhs_row, rhs_row) in enumerate(zip(lhs_rows, rhs_rows)):
        for col_idx, (lhs_value, rhs_value) in enumerate(zip(lhs_row, rhs_row)):
            if float_columns[col_idx]:
                try:
                    lhs_float = float(lhs_value)
                    rhs_float = float(rhs_value)
                except ValueError:
                    if lhs_value != rhs_value:
                        column = lhs.headers[col_idx]
                        return False, (
                            f"row {row_idx} column {column} parse mismatch: "
                            f"{lhs_value!r} != {rhs_value!r}"
                        )
                    continue

                if not math.isclose(
                    lhs_float, rhs_float, rel_tol=0.0, abs_tol=float_tolerance
                ):
                    column = lhs.headers[col_idx]
                    diff = abs(lhs_float - rhs_float)
                    return False, (
                        f"row {row_idx} column {column} float mismatch: "
                        f"{lhs_float} != {rhs_float} (abs diff {diff} > {float_tolerance})"
                    )
            elif lhs_value != rhs_value:
                column = lhs.headers[col_idx]
                return False, (
                    f"row {row_idx} column {column} mismatch: "
                    f"{lhs_value!r} != {rhs_value!r}"
                )

    return True, ""


def compare_values(
    expected: ParsedResult,
    actual_rows: list[list],
    float_tolerance: float,
    relative_tolerance: float = 0.0,
) -> tuple[bool, str]:
    """Compare ordered SQL values across frontends with different type/header spellings.

    Integer and decimal values are exact. Floating values use the declared tolerances.
    The reference's DuckDB types decide which columns are numeric; text containing
    digits remains text. Both raw schemas should be retained by the caller.
    """
    if len(expected.rows) != len(actual_rows):
        return False, f"row count mismatch: {len(expected.rows)} != {len(actual_rows)}"
    for row_index, (lhs, rhs) in enumerate(zip(expected.rows, actual_rows)):
        if len(lhs) != len(rhs):
            return False, f"row {row_index}: column count mismatch"
        for column, (reference, value, kind) in enumerate(
            zip(lhs, rhs, expected.types)
        ):
            actual = "NULL" if value is None else str(value)
            if reference == "NULL" or actual == "NULL":
                same = reference == actual
            elif is_float_type(kind):
                same = math.isclose(
                    float(reference),
                    float(actual),
                    rel_tol=relative_tolerance,
                    abs_tol=float_tolerance,
                )
            elif kind.lower().startswith(
                (
                    "decimal",
                    "tinyint",
                    "smallint",
                    "integer",
                    "bigint",
                    "hugeint",
                    "utinyint",
                    "usmallint",
                    "uinteger",
                    "ubigint",
                    "uhugeint",
                )
            ):
                try:
                    same = Decimal(reference) == Decimal(actual)
                except InvalidOperation:
                    same = False
            else:
                same = reference == actual
            if not same:
                return (
                    False,
                    f"row {row_index} column {column}: {reference!r} != {actual!r} ({kind})",
                )
    return True, ""


def with_json_rows(reference: ParsedResult, path: Path) -> ParsedResult:
    """Use DuckDB JSON values to preserve whitespace lost in boxed display padding."""
    objects = json.loads(path.read_text(), parse_float=Decimal)
    rows = [
        ["NULL" if row[name] is None else str(row[name]) for name in reference.headers]
        for row in objects
    ]
    return ParsedResult(reference.headers, reference.types, rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two DuckDB CLI result tables with float tolerance."
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument(
        "--expected-rows-json",
        type=Path,
        help="untrimmed DuckDB -json rows; column types come from expected boxed result",
    )
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="compare ordered values across frontend schemas",
    )
    parser.add_argument(
        "--actual-format",
        choices=("duckdb-box", "starrocks-json"),
        default="duckdb-box",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=DEFAULT_FLOAT_TOLERANCE,
        help=f"absolute tolerance for float/double columns (default: {DEFAULT_FLOAT_TOLERANCE})",
    )
    args = parser.parse_args()

    try:
        expected = parse_result(args.expected)
        if args.expected_rows_json:
            expected = with_json_rows(expected, args.expected_rows_json)
        if args.actual_format == "starrocks-json":
            if not args.values_only:
                parser.error("starrocks-json requires --values-only")
            actual_rows = json.loads(args.actual.read_text())["rows"]
        else:
            actual = parse_result(args.actual)
            actual_rows = actual.rows
    except Exception as exc:  # pragma: no cover - CLI error path
        print(exc, file=sys.stderr)
        return 2

    if args.values_only:
        match, message = compare_values(
            expected, actual_rows, args.float_tolerance, args.relative_tolerance
        )
    else:
        match, message = compare_results(expected, actual, args.float_tolerance)
    if match:
        return 0

    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
